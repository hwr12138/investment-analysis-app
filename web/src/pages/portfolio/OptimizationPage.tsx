import { BarChart, DonutChart } from "@mantine/charts";
import {
  Alert,
  Box,
  Button,
  Card,
  Grid,
  Group,
  NumberInput,
  Progress,
  SimpleGrid,
  Skeleton,
  Slider,
  Stack,
  TagsInput,
  Tabs,
  Text,
} from "@mantine/core";
import { DatePickerInput } from "@mantine/dates";
import { IconAlertTriangle, IconCalendar } from "@tabler/icons-react";
import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router";

import { api, unwrap, type Schemas } from "../../api/client";
import { Stat } from "../../components/Stat";
import { fixed, isoDay, MISSING, percent } from "../../format";
import {
  assetColor,
  cssColor,
  DEFAULT_START,
  defaultRiskFreePct,
  joined,
  Notices,
  numberParam,
  PageIntro,
  RiskFreeInput,
  split,
  today,
  useRiskFreeRate,
} from "./common";

type Strategy = Schemas["StrategyResult"]["strategy"];
type Range = [string | null, string | null];

/** One run: what the URL holds once Optimise is pressed. Rates in percent. */
type Request = { tickers: string[]; start: string; end: string; rf: number; vol: number; kelly: number };

const DEFAULT_TICKERS = ["QQQ", "GLD", "XLE"];
const DEFAULT_VOL = 25;
const DEFAULT_KELLY = 0.5;

function fromSearch(params: URLSearchParams): Request | null {
  const tickers = split(params.get("tickers"));
  const start = params.get("start");
  const end = params.get("end");
  const rf = numberParam(params.get("rf"));
  if (tickers.length < 2 || !start || !end || rf === null) return null;
  return {
    tickers,
    start,
    end,
    rf,
    vol: numberParam(params.get("vol")) ?? DEFAULT_VOL,
    kelly: numberParam(params.get("kelly")) ?? DEFAULT_KELLY,
  };
}

function toSearch(request: Request): URLSearchParams {
  return new URLSearchParams({
    tickers: joined(request.tickers),
    start: request.start,
    end: request.end,
    rf: String(request.rf),
    vol: String(request.vol),
    kelly: String(request.kelly),
  });
}

export function OptimizationPage() {
  const [params, setParams] = useSearchParams();
  const submitted = useMemo(() => fromSearch(params), [params]);

  const [tickers, setTickers] = useState<string[]>(submitted?.tickers ?? DEFAULT_TICKERS);
  const [range, setRange] = useState<Range>([submitted?.start ?? DEFAULT_START, submitted?.end ?? today()]);
  const [rfDraft, setRfDraft] = useState<number | "" | null>(submitted?.rf ?? null);
  const [vol, setVol] = useState<number | "">(submitted?.vol ?? DEFAULT_VOL);
  const [kelly, setKelly] = useState(submitted?.kelly ?? DEFAULT_KELLY);
  const rate = useRiskFreeRate();
  const rf = rfDraft ?? defaultRiskFreePct(rate.data) ?? "";
  const [start, end] = range;

  const problem =
    tickers.length < 2
      ? "Add at least two assets."
      : !start || !end
        ? "Pick both ends of the history."
        : rf === ""
          ? "Enter a risk-free rate."
          : vol === "" || vol <= 0
            ? "Give the volatility cap as a positive percentage."
            : null;
  const draft: Request | null = problem
    ? null
    : { tickers, start: start!, end: end!, rf: rf as number, vol: vol as number, kelly };
  const edited = submitted !== null && draft !== null && toSearch(draft).toString() !== toSearch(submitted).toString();

  return (
    <Stack gap="lg" maw={1440} mx="auto">
      <PageIntro
        title="Optimization"
        description="Four ways to allocate across the same assets, each read off their price history: the best Sharpe ratio on arithmetic and on compound returns, the best return under a volatility cap, and a fraction of the Kelly bet."
      />

      <Card>
        <Grid gap="lg" align="flex-end">
          <Grid.Col span={{ base: 12, md: 6 }}>
            <Stack gap="md">
              <TagsInput
                label="Assets"
                description="US stocks and ETFs covered by Sharadar. Press Enter or a comma after each."
                placeholder={tickers.length ? "" : "Add a ticker"}
                value={tickers}
                onChange={(values) => setTickers([...new Set(values.map((v) => v.trim().toUpperCase()).filter(Boolean))])}
                splitChars={[",", " "]}
                maxTags={20}
                clearable
              />
              <DatePickerInput
                type="range"
                label="History"
                value={range}
                onChange={setRange}
                maxDate={today()}
                valueFormat="MMM D, YYYY"
                leftSection={<IconCalendar size={16} />}
              />
            </Stack>
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <Stack gap="md">
              <SimpleGrid cols={2} spacing="md">
                <RiskFreeInput value={rf} onChange={setRfDraft} />
                <NumberInput
                  label="Volatility cap"
                  description="For the capped strategy"
                  suffix="%"
                  min={1}
                  max={200}
                  step={1}
                  allowNegative={false}
                  value={vol}
                  onChange={(value) => setVol(typeof value === "number" ? value : "")}
                />
              </SimpleGrid>
              <Box>
                <Group justify="space-between">
                  <Text size="sm" fw={500}>
                    Kelly fraction
                  </Text>
                  <Text size="sm" fw={600} className="num">
                    {fixed(kelly, 1)}×
                  </Text>
                </Group>
                <Slider
                  min={0.1}
                  max={1}
                  step={0.1}
                  value={kelly}
                  onChange={setKelly}
                  label={(value) => `${fixed(value, 1)}×`}
                  marks={[{ value: 0.5 }]}
                  mt={6}
                  mb={10}
                />
              </Box>
            </Stack>
          </Grid.Col>
        </Grid>
        <Group justify="flex-end" mt="lg" gap="md">
          <Text size="xs" c={problem ? "red" : "dimmed"}>
            {problem ?? (edited ? "The results below are for the last inputs you optimised." : "")}
          </Text>
          <Button disabled={draft === null} onClick={() => draft && setParams(toSearch(draft))}>
            Optimise
          </Button>
        </Group>
      </Card>

      {submitted && <Results request={submitted} />}
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

type Copy = { tab: string; describe: (r: Request) => string; labels: [string, string, string]; hint?: string };

const COPY: Record<Strategy, Copy> = {
  sharpe: {
    tab: "Max Sharpe",
    describe: () =>
      "The long-only mix with the highest Sharpe ratio on arithmetic mean returns. The return shown is that mix's compound rate over the history.",
    labels: ["Expected CAGR", "Annual volatility", "Sharpe ratio"],
  },
  geometric_sharpe: {
    tab: "Max geometric Sharpe",
    describe: () =>
      "The long-only mix with the highest Sharpe ratio on compound returns, which charges for the drag volatility puts on growth.",
    labels: ["Expected CAGR", "Annual volatility", "Geometric Sharpe"],
  },
  volatility_target: {
    tab: "Volatility cap",
    describe: (r) => `The long-only mix with the highest expected return whose volatility stays under ${fixed(r.vol, 1)}%.`,
    labels: ["Expected annual return", "Annual volatility", "Sharpe ratio"],
    hint: "The cap may be below what any long-only mix of these assets can reach.",
  },
  kelly: {
    tab: "Fractional Kelly",
    describe: (r) =>
      `The growth-maximising Kelly mix scaled by ${fixed(r.kelly, 1)}×, with the rest held in cash at the ${fixed(r.rf, 2)}% risk-free rate.`,
    labels: ["Expected annual return", "Annual volatility", "Cash"],
  },
};

function Results({ request }: { request: Request }) {
  const result = useQuery({
    queryKey: ["portfolio-optimization", request],
    queryFn: () =>
      unwrap(
        api.GET("/api/portfolio/optimization", {
          params: {
            query: {
              tickers: request.tickers,
              start: request.start,
              end: request.end,
              risk_free: request.rf / 100,
              vol_target: request.vol / 100,
              kelly_fraction: request.kelly,
            },
          },
        }),
      ),
  });

  if (result.isPending) {
    return (
      <Stack gap="lg">
        <Skeleton h={36} w={520} radius="md" />
        <SimpleGrid cols={{ base: 1, sm: 3 }} spacing="lg">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} h={96} radius="md" />
          ))}
        </SimpleGrid>
        <Skeleton h={340} radius="md" />
      </Stack>
    );
  }
  if (result.isError) {
    return (
      <Alert variant="light" color="red" icon={<IconAlertTriangle size={18} />} title="Couldn't optimise">
        {result.error.message}
      </Alert>
    );
  }

  const data = result.data;
  const order = (ticker: string) => data.tickers.indexOf(ticker);
  return (
    <Stack gap="lg">
      <Notices>
        {data.missing.length > 0 && (
          <Alert variant="light" color="yellow" icon={<IconAlertTriangle size={18} />}>
            No prices for {data.missing.join(", ")} in this range, so the optimisers allocate across{" "}
            {data.tickers.join(", ")} only.
          </Alert>
        )}
      </Notices>
      <Tabs defaultValue="sharpe" keepMounted={false} variant="pills" radius="md">
        <Group justify="space-between" align="center" gap="sm">
          <Tabs.List>
            {data.strategies.map((s) => (
              <Tabs.Tab key={s.strategy} value={s.strategy}>
                {COPY[s.strategy].tab}
              </Tabs.Tab>
            ))}
          </Tabs.List>
          <Text size="xs" c="dimmed">
            Prices from {isoDay(data.first_session)} to {isoDay(data.last_session)}
          </Text>
        </Group>
        {data.strategies.map((s) => (
          <Tabs.Panel key={s.strategy} value={s.strategy} pt="lg">
            <StrategyPanel result={s} request={request} order={order} />
          </Tabs.Panel>
        ))}
      </Tabs>
    </Stack>
  );
}

function StrategyPanel({
  result,
  request,
  order,
}: {
  result: Schemas["StrategyResult"];
  request: Request;
  order: (ticker: string) => number;
}) {
  const copy = COPY[result.strategy];
  if (result.reason !== null) {
    return (
      <Alert variant="light" color="yellow" icon={<IconAlertTriangle size={18} />} title="No allocation">
        {result.reason}
        {copy.hint && (
          <Text inherit mt={4}>
            {copy.hint}
          </Text>
        )}
      </Alert>
    );
  }

  const [returnLabel, volatilityLabel, thirdLabel] = copy.labels;
  const cash = result.allocation.find((w) => w.ticker === "CASH")?.weight ?? 0;
  const show = (value: number | null, format: (v: number) => string) => (value === null ? MISSING : format(value));
  return (
    <Stack gap="lg">
      <Text size="sm" c="dimmed" maw={760}>
        {copy.describe(request)}
      </Text>
      <SimpleGrid cols={{ base: 1, sm: 3 }} spacing="lg">
        <Stat label={returnLabel} value={show(result.expected_return, (v) => percent(v))} />
        <Stat label={volatilityLabel} value={show(result.volatility, (v) => percent(v))} />
        {result.strategy === "kelly" ? (
          <Stat
            label={thirdLabel}
            value={percent(cash, 1)}
            hint={cash > 0 ? `Earning ${fixed(request.rf, 2)}%` : undefined}
          />
        ) : (
          <Stat label={thirdLabel} value={show(result.ratio, (v) => fixed(v, 3))} />
        )}
      </SimpleGrid>
      <Grid gap="lg">
        <Grid.Col span={{ base: 12, lg: 5 }}>
          <AllocationCard allocation={result.allocation} order={order} />
        </Grid.Col>
        <Grid.Col span={{ base: 12, lg: 7 }}>
          <YearlyCard
            years={result.yearly_returns}
            title={result.strategy === "kelly" ? "Yearly returns of the invested part" : "Yearly returns"}
          />
        </Grid.Col>
      </Grid>
    </Stack>
  );
}

function AllocationCard({ allocation, order }: { allocation: Schemas["Weight"][]; order: (t: string) => number }) {
  const held = allocation.filter((w) => w.weight >= 0.0005).sort((a, b) => b.weight - a.weight);
  const unheld = allocation.filter((w) => w.weight < 0.0005).map((w) => w.ticker);
  const color = (ticker: string) => assetColor(order(ticker), ticker);
  return (
    <Card h="100%">
      <Text fw={600} mb="md">
        Allocation
      </Text>
      <Group align="center" gap="xl" wrap="wrap" justify="center">
        <DonutChart
          size={168}
          thickness={24}
          paddingAngle={2}
          withTooltip={false}
          data={held.map((w) => ({ name: w.ticker, value: w.weight, color: color(w.ticker) }))}
        />
        <Stack gap={10} style={{ flex: 1, minWidth: 180 }}>
          {held.map((w) => (
            <div key={w.ticker}>
              <Group justify="space-between" gap="xs">
                <Group gap={8}>
                  <Box w={10} h={10} style={{ borderRadius: 3, background: cssColor(color(w.ticker)) }} />
                  <Text size="sm" fw={600}>
                    {w.ticker}
                  </Text>
                </Group>
                <Text size="sm" className="num">
                  {percent(w.weight, 1)}
                </Text>
              </Group>
              <Progress value={w.weight * 100} color={color(w.ticker)} size={4} mt={4} />
            </div>
          ))}
          {unheld.length > 0 && (
            <Text size="xs" c="dimmed">
              Not held: {unheld.join(", ")}
            </Text>
          )}
        </Stack>
      </Group>
    </Card>
  );
}

function YearlyCard({ years, title }: { years: Schemas["YearReturn"][]; title: string }) {
  const data = years
    .filter((y) => y.value !== null)
    .map((y) => ({ year: String(y.year), value: (y.value as number) * 100 }));
  return (
    <Card h="100%">
      <Text fw={600}>{title}</Text>
      <Text size="xs" c="dimmed" mb="md">
        The first and last years cover only the sessions inside the history.
      </Text>
      <BarChart
        h={240}
        data={data}
        dataKey="year"
        series={[{ name: "value", label: "Return" }]}
        getBarColor={(value) => (value >= 0 ? "teal.6" : "red.6")}
        valueFormatter={(value) => `${fixed(value, 1)}%`}
        withLegend={false}
        gridAxis="x"
        tickLine="none"
        yAxisProps={{ tickFormatter: (value: number) => `${value}%` }}
        barProps={{ radius: 4 }}
      />
    </Card>
  );
}

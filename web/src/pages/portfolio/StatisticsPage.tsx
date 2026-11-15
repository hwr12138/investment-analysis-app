import {
  ActionIcon,
  Alert,
  Badge,
  Box,
  Button,
  Card,
  Grid,
  Group,
  NumberInput,
  Progress,
  ScrollArea,
  SegmentedControl,
  Skeleton,
  Stack,
  Table,
  Text,
  TextInput,
  Tooltip,
} from "@mantine/core";
import { DatePickerInput } from "@mantine/dates";
import { IconAlertTriangle, IconCalendar, IconInfoCircle, IconPlus, IconX } from "@tabler/icons-react";
import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router";

import { api, unwrap, type Schemas } from "../../api/client";
import { fixed, isoDay, MISSING, percent } from "../../format";
import {
  assetColor,
  assetHue,
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

type Holding = { id: number; ticker: string; weight: number | "" };
type Range = [string | null, string | null];

/** One calculation: what the URL holds once Calculate is pressed. */
type Request = { tickers: string[]; weights: number[]; start: string; end: string; rf: number };

const DEFAULT_HOLDINGS: [string, number][] = [
  ["QQQ", 55],
  ["GLD", 30],
  ["XLE", 15],
];

function fromSearch(params: URLSearchParams): Request | null {
  const tickers = split(params.get("tickers"));
  const weights = split(params.get("weights")).map(Number);
  const start = params.get("start");
  const end = params.get("end");
  const rf = numberParam(params.get("rf"));
  if (!tickers.length || weights.length !== tickers.length || weights.some((w) => !Number.isFinite(w))) return null;
  if (!start || !end || rf === null) return null;
  return { tickers, weights, start, end, rf };
}

function toSearch(request: Request): URLSearchParams {
  return new URLSearchParams({
    tickers: joined(request.tickers),
    weights: joined(request.weights),
    start: request.start,
    end: request.end,
    rf: String(request.rf),
  });
}

let nextId = 0;
const holding = (ticker: string, weight: number | ""): Holding => ({ id: nextId++, ticker, weight });

export function StatisticsPage() {
  const [params, setParams] = useSearchParams();
  const submitted = useMemo(() => fromSearch(params), [params]);

  // The form starts from the last calculation in the URL, or the defaults.
  const [holdings, setHoldings] = useState<Holding[]>(() =>
    submitted
      ? submitted.tickers.map((t, i) => holding(t, submitted.weights[i]!))
      : DEFAULT_HOLDINGS.map(([t, w]) => holding(t, w)),
  );
  const [range, setRange] = useState<Range>([submitted?.start ?? DEFAULT_START, submitted?.end ?? today()]);
  // null until the reader types, so the fetched T-bill yield shows through.
  const [rfDraft, setRfDraft] = useState<number | "" | null>(submitted?.rf ?? null);
  const rate = useRiskFreeRate();
  const rf = rfDraft ?? defaultRiskFreePct(rate.data) ?? "";

  const listed = holdings.filter((h) => h.ticker.trim());
  const tickers = listed.map((h) => h.ticker.trim().toUpperCase());
  const duplicates = new Set(tickers.filter((t, i) => tickers.indexOf(t) !== i));
  const total = listed.reduce((sum, h) => sum + (h.weight === "" ? 0 : h.weight), 0);
  const [start, end] = range;

  const problem = !listed.length
    ? "Add at least one holding."
    : duplicates.size
      ? "Each ticker can be listed only once."
      : listed.some((h) => h.weight === "")
        ? "Give every holding a weight."
        : total <= 0
          ? "The weights must add up to more than zero."
          : !start || !end
            ? "Pick both ends of the window."
            : rf === ""
              ? "Enter a risk-free rate."
              : null;

  const draft: Request | null = problem
    ? null
    : { tickers, weights: listed.map((h) => h.weight as number), start: start!, end: end!, rf: rf as number };
  const edited = submitted !== null && draft !== null && toSearch(draft).toString() !== toSearch(submitted).toString();

  const update = (id: number, change: Partial<Holding>) =>
    setHoldings((rows) => rows.map((row) => (row.id === id ? { ...row, ...change } : row)));

  return (
    <Stack gap="lg" maw={1440} mx="auto">
      <PageIntro
        title="Statistics"
        description="Risk and return for a fixed-weight portfolio over one to ten years and a window you choose, and how its holdings move together."
      />

      <Grid gap="lg">
        <Grid.Col span={{ base: 12, md: 7 }}>
          <Card h="100%">
            <Text fw={600}>Holdings</Text>
            <Text size="xs" c="dimmed" mb="md">
              US stocks and ETFs covered by Sharadar, each with its share of the portfolio.
            </Text>
            <Stack gap="xs">
              {holdings.map((row, index) => {
                const ticker = row.ticker.trim().toUpperCase();
                return (
                  <Group key={row.id} gap="xs" wrap="nowrap" align="flex-start">
                    <Box w={10} h={36} style={{ display: "flex", alignItems: "center" }}>
                      <Box w={10} h={10} style={{ borderRadius: 3, background: cssColor(assetColor(index)) }} />
                    </Box>
                    <TextInput
                      aria-label="Ticker"
                      placeholder="Ticker"
                      value={row.ticker}
                      onChange={(event) => update(row.id, { ticker: event.currentTarget.value.toUpperCase() })}
                      error={duplicates.has(ticker) ? "Listed twice" : undefined}
                      style={{ flex: 1 }}
                    />
                    <NumberInput
                      aria-label="Weight"
                      placeholder="Weight"
                      suffix="%"
                      min={0}
                      allowNegative={false}
                      decimalScale={2}
                      value={row.weight}
                      onChange={(value) => update(row.id, { weight: typeof value === "number" ? value : "" })}
                      w={120}
                    />
                    <ActionIcon
                      variant="subtle"
                      color="gray"
                      size={36}
                      aria-label="Remove holding"
                      disabled={holdings.length === 1}
                      onClick={() => setHoldings((rows) => rows.filter((r) => r.id !== row.id))}
                    >
                      <IconX size={16} />
                    </ActionIcon>
                  </Group>
                );
              })}
            </Stack>
            <Button
              variant="subtle"
              leftSection={<IconPlus size={16} />}
              onClick={() => setHoldings((rows) => [...rows, holding("", "")])}
              mt="xs"
              w="fit-content"
              disabled={holdings.length >= 20}
            >
              Add holding
            </Button>

            <Box mt="md">
              <Progress.Root size={12} radius="xl">
                {holdings.map((row, index) =>
                  row.weight !== "" && row.weight > 0 && total > 0 ? (
                    <Tooltip key={row.id} label={`${row.ticker || "Unnamed"} ${fixed(row.weight, 2)}%`}>
                      <Progress.Section value={(row.weight / total) * 100} color={assetColor(index)} />
                    </Tooltip>
                  ) : null,
                )}
              </Progress.Root>
              <Group justify="space-between" mt={6}>
                <Text size="xs" c="dimmed">
                  Total{" "}
                  <Text span fw={600} c={total === 100 ? "teal" : "yellow.7"} inherit className="num">
                    {fixed(total, 2)}%
                  </Text>
                </Text>
                {total !== 100 && total > 0 && (
                  <Text size="xs" c="dimmed">
                    Weights are scaled to 100% when calculated.
                  </Text>
                )}
              </Group>
            </Box>
          </Card>
        </Grid.Col>

        <Grid.Col span={{ base: 12, md: 5 }}>
          <Card h="100%">
            <Text fw={600} mb="md">
              Settings
            </Text>
            <Stack gap="md">
              <DatePickerInput
                type="range"
                label="Custom window"
                description="Reported beside the 1, 3, 5 and 10-year windows, which end on the same day."
                value={range}
                onChange={setRange}
                maxDate={today()}
                valueFormat="MMM D, YYYY"
                leftSection={<IconCalendar size={16} />}
              />
              <RiskFreeInput value={rf} onChange={setRfDraft} />
              <Box mt="auto">
                <Button
                  fullWidth
                  disabled={draft === null}
                  onClick={() => draft && setParams(toSearch(draft))}
                >
                  Calculate
                </Button>
                <Text size="xs" c={problem ? "red" : "dimmed"} mt={6} ta="center" mih={18}>
                  {problem ?? (edited ? "The results below are for the last inputs you calculated." : "")}
                </Text>
              </Box>
            </Stack>
          </Card>
        </Grid.Col>
      </Grid>

      {submitted && <Results request={submitted} />}
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

function Results({ request }: { request: Request }) {
  const result = useQuery({
    queryKey: ["portfolio-statistics", request],
    queryFn: () =>
      unwrap(
        api.GET("/api/portfolio/statistics", {
          params: {
            query: {
              tickers: request.tickers,
              weights: request.weights,
              start: request.start,
              end: request.end,
              risk_free: request.rf / 100,
            },
          },
        }),
      ),
  });

  if (result.isPending) {
    return (
      <Grid gap="lg">
        <Grid.Col span={{ base: 12, lg: 7 }}>
          <Skeleton h={420} radius="md" />
        </Grid.Col>
        <Grid.Col span={{ base: 12, lg: 5 }}>
          <Skeleton h={420} radius="md" />
        </Grid.Col>
      </Grid>
    );
  }
  if (result.isError) {
    return (
      <Alert variant="light" color="red" icon={<IconAlertTriangle size={18} />} title="Couldn't calculate">
        {result.error.message}
      </Alert>
    );
  }

  const data = result.data;
  // Colours follow the order the holdings were entered, as in the form.
  const order = (ticker: string) => request.tickers.indexOf(ticker);
  return (
    <Stack gap="lg">
      <Notices>
        {data.missing.length > 0 && (
          <Alert variant="light" color="yellow" icon={<IconAlertTriangle size={18} />}>
            No prices for {data.missing.join(", ")}, so {data.missing.length === 1 ? "it is" : "they are"} left out and
            the rest are rescaled.
          </Alert>
        )}
        {data.rescaled && (
          <Alert variant="light" color="blue" icon={<IconInfoCircle size={18} />}>
            The weights summed to {fixed(data.entered_total, 2)}%, so each was scaled in proportion to make 100%.
          </Alert>
        )}
        {!data.benchmark_priced && (
          <Alert variant="light" color="yellow" icon={<IconAlertTriangle size={18} />}>
            No {data.benchmark} prices came back, so alpha, beta and Treynor are not reported rather than measured
            against a flat market.
          </Alert>
        )}
      </Notices>

      <Grid gap="lg">
        <Grid.Col span={{ base: 12, lg: 7 }}>
          <MetricsCard data={data} request={request} order={order} />
        </Grid.Col>
        <Grid.Col span={{ base: 12, lg: 5 }}>
          <RelationshipsCard relationships={data.relationships} request={request} />
        </Grid.Col>
      </Grid>
    </Stack>
  );
}

function MetricsCard({
  data,
  request,
  order,
}: {
  data: Schemas["PortfolioStatistics"];
  request: Request;
  order: (ticker: string) => number;
}) {
  return (
    <Card h="100%">
      <Group justify="space-between" align="flex-start" mb="md" gap="xs">
        <div>
          <Text fw={600}>Risk and return</Text>
          <Text size="xs" c="dimmed">
            Against {data.benchmark} for alpha, beta and Treynor.
          </Text>
        </div>
        <Group gap={6}>
          {data.holdings.map((h) => (
            <Badge
              key={h.ticker}
              variant="light"
              color={assetHue(order(h.ticker))}
              radius="sm"
              className="num"
            >
              {h.ticker} {percent(h.weight, 1)}
            </Badge>
          ))}
        </Group>
      </Group>
      <ScrollArea type="auto">
        <Table striped highlightOnHover verticalSpacing={8} miw={560}>
          <Table.Thead>
            <Table.Tr>
              <Table.Th />
              {data.windows.map((w) => (
                <Table.Th key={w} ta="right">
                  {w}
                </Table.Th>
              ))}
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {data.metrics.map((row) => (
              <Table.Tr key={row.name}>
                <Table.Td fw={500}>{row.name}</Table.Td>
                {row.values.map((value, i) => (
                  <Table.Td key={data.windows[i]} ta="right" className="num">
                    {value === null ? (
                      <Text span c="dimmed" inherit>
                        {MISSING}
                      </Text>
                    ) : row.unit === "percent" ? (
                      percent(value)
                    ) : (
                      fixed(value, 3)
                    )}
                  </Table.Td>
                ))}
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </ScrollArea>
      <Text size="xs" c="dimmed" mt="sm">
        Custom covers {isoDay(request.start)} – {isoDay(request.end)}. {MISSING} means the window is longer than the
        price history, or the figure is undefined (no volatility, no benchmark).
      </Text>
    </Card>
  );
}

function RelationshipsCard({
  relationships,
  request,
}: {
  relationships: Schemas["PortfolioStatistics"]["relationships"];
  request: Request;
}) {
  const [kind, setKind] = useState<"correlation" | "covariance">("correlation");
  return (
    <Card h="100%">
      <Group justify="space-between" mb="md" gap="xs">
        <div>
          <Text fw={600}>How they move together</Text>
          <Text size="xs" c="dimmed">
            Daily returns over the custom window.
          </Text>
        </div>
        <SegmentedControl
          size="xs"
          value={kind}
          onChange={(value) => setKind(value as typeof kind)}
          data={[
            { label: "Correlation", value: "correlation" },
            { label: "Covariance", value: "covariance" },
          ]}
          disabled={!relationships}
        />
      </Group>
      {relationships ? (
        <Heatmap
          matrix={relationships[kind]}
          digits={kind === "correlation" ? 2 : 4}
          order={(t) => request.tickers.indexOf(t)}
        />
      ) : (
        <Text size="sm" c="dimmed">
          {request.tickers.length < 2
            ? "Add a second holding to see how they move together."
            : "Too few shared days in the custom window to compare the holdings."}
        </Text>
      )}
      {relationships && kind === "covariance" && (
        <Text size="xs" c="dimmed" mt="sm">
          Annualised.
        </Text>
      )}
    </Card>
  );
}

/** A matrix as a table whose cells are shaded by value: indigo for moving
 *  together, orange for moving apart, stronger as the value grows. */
function Heatmap({
  matrix,
  digits,
  order,
}: {
  matrix: Schemas["Matrix"];
  digits: number;
  order: (ticker: string) => number;
}) {
  const largest = Math.max(...matrix.values.flat().map((v) => Math.abs(v ?? 0)), Number.EPSILON);
  return (
    <ScrollArea type="auto">
      <Table withColumnBorders={false} verticalSpacing={10} horizontalSpacing={8} miw={matrix.tickers.length * 72 + 72}>
        <Table.Thead>
          <Table.Tr>
            <Table.Th />
            {matrix.tickers.map((t) => (
              <Table.Th key={t} ta="center">
                <TickerLabel ticker={t} index={order(t)} />
              </Table.Th>
            ))}
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {matrix.values.map((row, i) => (
            <Table.Tr key={matrix.tickers[i]}>
              <Table.Th>
                <TickerLabel ticker={matrix.tickers[i]!} index={order(matrix.tickers[i]!)} />
              </Table.Th>
              {row.map((value, j) => {
                const strength = value === null ? 0 : Math.round((Math.abs(value) / largest) * 55);
                const hue = value !== null && value < 0 ? "orange" : "indigo";
                return (
                  <Table.Td
                    key={matrix.tickers[j]}
                    ta="center"
                    className="num"
                    style={{
                      background: `color-mix(in srgb, var(--mantine-color-${hue}-6) ${strength}%, transparent)`,
                      borderRadius: 6,
                    }}
                  >
                    {value === null ? (
                      <Text span c="dimmed" inherit>
                        {MISSING}
                      </Text>
                    ) : (
                      fixed(value, digits)
                    )}
                  </Table.Td>
                );
              })}
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </ScrollArea>
  );
}

function TickerLabel({ ticker, index }: { ticker: string; index: number }) {
  return (
    <Group gap={6} wrap="nowrap" justify="center">
      <Box w={8} h={8} style={{ borderRadius: 2, background: cssColor(assetColor(index)) }} />
      <Text size="sm" fw={600}>
        {ticker}
      </Text>
    </Group>
  );
}

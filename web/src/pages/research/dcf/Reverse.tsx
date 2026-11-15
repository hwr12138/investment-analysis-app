// The reverse DCF: hold value at the market price and solve for the year-1
// revenue growth that would justify it, under both terminal methods.
import {
  Accordion,
  Alert,
  Card,
  Loader,
  SimpleGrid,
  Stack,
  Table,
  Text,
  Title,
  useComputedColorScheme,
  useMantineTheme,
} from "@mantine/core";
import type { Data, Layout } from "plotly.js";
import { useMemo, useState } from "react";

import { chartPalette } from "../../../components/chartPalette";
import { PlotlyChart } from "../../../components/PlotlyChart";
import { fixed, isoDay, percent } from "../../../format";
import { Stat } from "../../../components/Stat";
import { price, useDcfWhatIf } from "../common";
import type { Overrides } from "./overrides";
import { Caption, type Company, Md, NumberField, type Panel, ProvenanceLine, type Valuation } from "./parts";

type Implied = Valuation["implied"];
type Solve = Implied["solves"][number];

type Props = { ticker: string; overrides: Overrides; valuation: Valuation; panel: Panel; company: Company };

const LEVEL_COLOR = { info: "blue", warning: "yellow", error: "red" } as const;

export function Reverse({ ticker, overrides, valuation, panel, company }: Props) {
  const implied = valuation.implied;
  const history = implied.history;
  const currency = company.valuation_currency;
  const shape = implied.faded
    ? `a year-1 revenue growth rate, held for ${implied.hold} year(s) and then faded toward the terminal rate over the rest of the ${implied.years}-year forecast, exactly as yours is`
    : `a flat revenue growth rate across all ${implied.years} forecast years`;

  return (
    <Stack gap="lg">
      <Caption>
        {`Everything else on this tab holds the assumptions fixed and solves for value. This holds value fixed at the market price and solves for the one assumption that would justify it: ${shape}, with the other drivers exactly as set. It is solved under both terminal methods, whichever one is the headline.`}
      </Caption>
      <SimpleGrid cols={{ base: 2, md: 3, lg: 5 }} spacing="md">
        {implied.solves.map((solve) => (
          <Stat
            key={solve.method}
            label={`Implied — ${solve.method === "gordon" ? "Gordon" : "exit multiple"}`}
            value={solve.growth !== null ? percent(solve.growth) : solve.roots.length > 1 ? "no single rate" : "—"}
            hint="Year-1 revenue growth"
          />
        ))}
        <Stat label="Your year-1 forecast" value={percent(implied.forecast_growth)} hint={implied.faded ? "then fading" : "held flat"} />
        <Stat
          label="Realised CAGR"
          value={history.ok && history.cagr !== null ? percent(history.cagr) : "n/a"}
          hint={history.ok ? `${history.first_period}–${history.last_period}` : history.detail}
        />
        <Stat
          label="Market price"
          value={company.last_price !== null ? price(company.last_price, currency) : "n/a"}
          hint={company.price_date ? `Close ${isoDay(company.price_date)}` : undefined}
        />
      </SimpleGrid>

      <Card>
        <Stack gap="sm">
          <Caption>
            **How to read the two.** The Gordon solve is the fundamentals reading: the growth the business must deliver
            for the price to be right when growth after the forecast earns only its cost of capital. The exit-multiple
            solve holds the terminal multiple at the company's own recent median, so it mostly measures how far today's
            multiple sits from that history rather than what the business must do.
          </Caption>
          {implied.solves.map((solve) => (
            <Verdict key={solve.method} solve={solve} implied={implied} company={company} />
          ))}
          {history.ok && history.arithmetic_mean !== null && history.cagr !== null && (
            <Caption>
              {`**Two ways to state past growth, and they are not the same number.** The *arithmetic average* of yearly growth rates is **${percent(history.arithmetic_mean)}**; the *compound* rate is **${percent(history.cagr)}**, which is the one comparable to a forecast that compounds. They diverge whenever growth was uneven, and the average is always the higher.${
                history.minimum !== null && history.maximum !== null && history.median !== null
                  ? ` Yearly rates ranged ${percent(history.minimum)} to ${percent(history.maximum)}, median ${percent(history.median)}, over ${history.observations} year(s).`
                  : ""
              }${history.detail ? ` ${history.detail}` : ""}`}
            </Caption>
          )}
        </Stack>
      </Card>

      {implied.chart && (
        <Card>
          <Title order={4} mb="xs">
            Revenue paths
          </Title>
          <RevenueChart chart={implied.chart} currency={currency} />
          <Caption>
            {`Forecast years are labelled by offset: each is the twelve months after the base period, **${panel.base_label}**, which is where every path starts.${
              panel.base_is_ttm
                ? " The base period is a trailing twelve months, so the step from the last fiscal-year bar to the anchor is a real discontinuity, not a plotting artefact."
                : ""
            }${
              company.converted
                ? ` Reported bars are converted at today's spot rate, like the forecast, so they compare on one axis; the history tables stay in ${company.currency}.`
                : ""
            }`}
          </Caption>
        </Card>
      )}

      <Accordion variant="separated" multiple>
        {implied.scan.length > 0 && (
          <Accordion.Item value="scan">
            <Accordion.Control>
              <Text size="sm" fw={500}>
                Value per share across the growth range
              </Text>
            </Accordion.Control>
            <Accordion.Panel>
              <Scan implied={implied} company={company} />
            </Accordion.Panel>
          </Accordion.Item>
        )}
        <Accordion.Item value="what-if">
          <Accordion.Control>
            <Text size="sm" fw={500}>
              Solve for a different price
            </Text>
          </Accordion.Control>
          <Accordion.Panel>
            <WhatIf ticker={ticker} overrides={overrides} implied={implied} company={company} />
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>
    </Stack>
  );
}

function Verdict({ solve, implied, company }: { solve: Solve; implied: Implied; company: Company }) {
  const history = implied.history;
  const closing = company.last_price !== null ? price(company.last_price, company.valuation_currency) : "n/a";
  return (
    <Stack gap={4}>
      {solve.growth !== null && (
        <>
          <Text size="sm">
            <Md
              text={`**${solve.label}:** the price implies **${percent(solve.growth)}** in year 1${
                implied.faded ? ", then fading" : " a year"
              }, against your **${percent(implied.forecast_growth)}**.${
                history.ok && history.cagr !== null
                  ? ` The company compounded **${percent(history.cagr)}** over the ${history.intervals} years to ${history.last_period}.`
                  : ""
              }`}
            />
          </Text>
          <ProvenanceLine
            provenance={{
              origin: "derived",
              source: `Solved against the last close of ${closing}${company.price_date ? ` (${company.price_date})` : ""}, ${solve.label.toLowerCase()}`,
              formula: solve.formula,
              as_of: null,
              detail: "",
            }}
          />
        </>
      )}
      {solve.message && (
        <Alert variant="light" color={LEVEL_COLOR[solve.level || "info"]} p="xs">
          <Text size="xs">
            <Md text={`*${solve.label}* — ${solve.message}`} />
          </Text>
        </Alert>
      )}
    </Stack>
  );
}

function RevenueChart({ chart, currency }: { chart: NonNullable<Implied["chart"]>; currency: string }) {
  const theme = useMantineTheme();
  const dark = useComputedColorScheme("dark") === "dark";
  const p = chartPalette(theme, dark);
  const c = theme.colors;
  const shade = dark ? 4 : 6;

  const traces = useMemo<Data[]>(() => {
    const colour = (path: (typeof chart.paths)[number]) =>
      path.method === "gordon"
        ? c.orange[shade]
        : path.method === "exit_multiple"
          ? c.red[shade]
          : path.kind === "forecast"
            ? c.blue[shade]
            : c.teal[shade];
    const dash = { price: "dash", forecast: "dot", history: "longdash" } as const;
    const width = { price: 3, forecast: 2.5, history: 1.5 } as const;
    return [
      {
        type: "bar",
        name: "Reported revenue",
        x: chart.history_labels,
        y: chart.reported,
        marker: { color: dark ? c.dark[3] : c.gray[4] },
        hovertemplate: "Reported %{x}: %{y:,.4s}<extra></extra>",
      },
      ...chart.paths.map(
        (path): Data => ({
          type: "scatter",
          mode: "lines+markers",
          name: path.name,
          // Anchored at the base period so every path fans out from one point.
          x: [chart.anchor, ...chart.forward_labels],
          y: [chart.base_revenue, ...path.values],
          line: { color: colour(path), dash: dash[path.kind], width: width[path.kind] },
          marker: { size: 4 },
          hovertemplate: `${path.name}<br>%{x}: %{y:,.4s}<extra></extra>`,
        }),
      ),
    ];
  }, [chart, c, shade, dark]);

  const layout = useMemo<Partial<Layout>>(
    () => ({
      margin: { t: 8, r: 16, b: 40, l: 64 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: p.font, color: p.text, size: 11 },
      hovermode: "closest",
      hoverlabel: { font: { family: p.font, size: 11 } },
      legend: { orientation: "h", y: -0.18, bgcolor: "rgba(0,0,0,0)" },
      xaxis: {
        type: "category",
        categoryorder: "array",
        categoryarray: [...chart.history_labels, chart.anchor, ...chart.forward_labels],
        tickfont: { size: 10 },
        tickangle: -45,
      },
      yaxis: {
        gridcolor: p.grid,
        zerolinecolor: p.zero,
        tickformat: "~s",
        title: { text: `Revenue (${currency})`, font: { size: 10 } },
        tickfont: { size: 10 },
      },
    }),
    [p.font, p.text, p.grid, p.zero, chart, currency],
  );

  return <PlotlyChart data={traces} layout={layout} height={420} />;
}

function Scan({ implied, company }: { implied: Implied; company: Company }) {
  return (
    <Stack gap="xs">
      <Caption>
        The scans the solvers already ran, so this costs nothing extra. Every crossing of the market price in a column is
        a rate the price is consistent with under that method.
      </Caption>
      <Table.ScrollContainer minWidth={420}>
        <Table fz="sm" verticalSpacing={4} striped maw={560}>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Year-1 revenue growth</Table.Th>
              <Table.Th ta="right">Gordon growth</Table.Th>
              <Table.Th ta="right">Exit multiple</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {implied.scan.map((row) => (
              <Table.Tr key={row.growth}>
                <Table.Td className="num">{percent(row.growth)}</Table.Td>
                <Table.Td ta="right" className="num">
                  {row.gordon === null ? "" : fixed(row.gordon)}
                </Table.Td>
                <Table.Td ta="right" className="num">
                  {row.exit_multiple === null ? "" : fixed(row.exit_multiple)}
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
      {company.last_price !== null && (
        <Caption>{`Market price: **${price(company.last_price, company.valuation_currency)}**.`}</Caption>
      )}
    </Stack>
  );
}

function WhatIf({
  ticker,
  overrides,
  implied,
  company,
}: {
  ticker: string;
  overrides: Overrides;
  implied: Implied;
  company: Company;
}) {
  const [target, setTarget] = useState<number | null>(null);
  const answer = useDcfWhatIf(ticker, overrides, target);
  const currency = company.valuation_currency;
  return (
    <Stack gap="sm">
      <Caption>
        Answers *what growth would justify this price?* — using the same assumptions as the implied rates above, with only
        the target price changed.
      </Caption>
      <NumberField
        label={`Price per share (${currency})`}
        maw={240}
        min={0.01}
        step={1}
        value={target ?? company.last_price}
        fallback={company.last_price}
        overridden={target !== null}
        onCommit={setTarget}
        rightSection={answer.isFetching ? <Loader size={14} /> : undefined}
      />
      {answer.isError && (
        <Alert variant="light" color="red" p="xs">
          <Text size="xs">{answer.error.message}</Text>
        </Alert>
      )}
      {target !== null &&
        answer.data?.solves.map((solve) =>
          solve.growth !== null ? (
            <Stack key={solve.method} gap={4}>
              <Text size="sm">
                <Md
                  text={`**${solve.label}:** a price of **${price(answer.data.price, currency)}** implies year-1 revenue growth of **${percent(solve.growth)}**${
                    implied.faded ? ", fading from there." : " a year, held flat."
                  }`}
                />
              </Text>
              <ProvenanceLine
                provenance={{
                  origin: "derived",
                  source: `Solved against a hypothetical price of ${price(answer.data.price, currency)}, ${solve.label.toLowerCase()}`,
                  formula: solve.formula,
                  as_of: null,
                  detail: "",
                }}
              />
            </Stack>
          ) : (
            <Alert key={solve.method} variant="light" color={LEVEL_COLOR[solve.level || "info"]} p="xs">
              <Text size="xs">
                <Md text={`*${solve.label}* — ${solve.message || solve.reason}`} />
              </Text>
            </Alert>
          ),
        )}
      {target === null && (
        <Text size="xs" c="dimmed">
          Type a price other than the last close to solve for it.
        </Text>
      )}
    </Stack>
  );
}

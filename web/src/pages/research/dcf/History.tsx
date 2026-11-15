// The ratios the drivers are derived from, year by year, and the filing each
// fiscal year came from. The statements themselves are on the Financials tab.
import {
  Anchor,
  Badge,
  Card,
  SimpleGrid,
  Stack,
  Table,
  Text,
  Title,
  useComputedColorScheme,
  useMantineTheme,
} from "@mantine/core";
import { IconExternalLink } from "@tabler/icons-react";
import type { Data, Layout } from "plotly.js";
import { useMemo } from "react";

import type { Schemas } from "../../../api/client";
import { chartPalette } from "../../../components/chartPalette";
import { PlotlyChart } from "../../../components/PlotlyChart";
import { isoDay, MISSING, percent } from "../../../format";
import { Caption, type Company } from "./parts";

type History = Schemas["History"];

const CHARTED = ["operating_margin", "capex_pct_revenue", "dna_pct_revenue", "effective_tax_rate"];

export function HistoryPanel({ history, company }: { history: History; company: Company }) {
  // Newest first: thirty-odd fiscal years is a long table to scroll to today.
  const years = [...history.ratios].reverse();
  return (
    <Stack gap="lg">
      <Caption>
        {`${company.years_retrieved} fiscal years retrieved. Years used per driver: ${history.usable
          .map((u) => `${u.name}: ${u.value ?? 0}`)
          .join(" · ")}. Years missing a component are excluded, never zero-filled.${
          company.converted ? ` Figures here are in the reporting currency, ${company.currency}.` : ""
        }`}
      </Caption>

      <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="lg">
        <Card>
          <Title order={5} mb="xs">
            Revenue and EBIT
          </Title>
          <AmountsChart history={history} currency={company.currency} />
        </Card>
        <Card>
          <Title order={5} mb="xs">
            Driver ratios
          </Title>
          <RatiosChart history={history} />
        </Card>
      </SimpleGrid>

      <Card p={0}>
        <Table.ScrollContainer minWidth={180 + history.ratio_labels.length * 120} maxHeight={520}>
          <Table fz="xs" verticalSpacing={5} striped highlightOnHover stickyHeader>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Fiscal year</Table.Th>
                {history.ratio_labels.map((label) => (
                  <Table.Th key={label} ta="right">
                    {label}
                  </Table.Th>
                ))}
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {years.map((year) => (
                <Table.Tr key={year.period}>
                  <Table.Td>{year.period}</Table.Td>
                  {year.values.map((value, i) => (
                    <Table.Td key={i} ta="right" className="num">
                      {value === null ? (
                        <Text span c="dimmed" inherit>
                          {MISSING}
                        </Text>
                      ) : (
                        percent(value)
                      )}
                    </Table.Td>
                  ))}
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
      </Card>
      <Caption>
        These are the ratios the drivers are derived from, not the statements they were derived from. For the line items
        themselves — all four statements, sectioned, reconciled, annual or quarterly — see the **Financials** tab.
      </Caption>

      <Filings filings={history.filings} />
    </Stack>
  );
}

function AmountsChart({ history, currency }: { history: History; currency: string }) {
  const theme = useMantineTheme();
  const dark = useComputedColorScheme("dark") === "dark";
  const p = chartPalette(theme, dark);
  const c = theme.colors;
  const periods = history.ratios.map((r) => r.period);

  const traces = useMemo<Data[]>(() => {
    const out: Data[] = [];
    if (history.revenue.length)
      out.push({
        type: "bar",
        name: "Revenue",
        x: periods,
        y: history.revenue,
        marker: { color: c.indigo[dark ? 4 : 6] },
        hovertemplate: "Revenue %{x}: %{y:,.4s}<extra></extra>",
      });
    if (history.ebit.length)
      out.push({
        type: "bar",
        name: "EBIT",
        x: periods,
        y: history.ebit,
        marker: { color: c.teal[dark ? 4 : 6] },
        hovertemplate: "EBIT %{x}: %{y:,.4s}<extra></extra>",
      });
    return out;
  }, [history, periods, c, dark]);

  const layout = useMemo<Partial<Layout>>(
    () => ({
      margin: { t: 8, r: 8, b: 40, l: 56 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: p.font, color: p.text, size: 11 },
      hoverlabel: { font: { family: p.font, size: 11 } },
      barmode: "group",
      legend: { orientation: "h", y: -0.2, bgcolor: "rgba(0,0,0,0)" },
      xaxis: { type: "category", tickfont: { size: 10 }, tickangle: -45 },
      yaxis: {
        gridcolor: p.grid,
        zerolinecolor: p.zero,
        tickformat: "~s",
        title: { text: currency, font: { size: 10 } },
        tickfont: { size: 10 },
      },
    }),
    [p.font, p.text, p.grid, p.zero, currency],
  );
  return <PlotlyChart data={traces} layout={layout} height={340} />;
}

function RatiosChart({ history }: { history: History }) {
  const theme = useMantineTheme();
  const dark = useComputedColorScheme("dark") === "dark";
  const p = chartPalette(theme, dark);
  const c = theme.colors;
  const periods = history.ratios.map((r) => r.period);

  const traces = useMemo<Data[]>(() => {
    const colours = [c.indigo, c.orange, c.teal, c.grape].map((scale) => scale[dark ? 4 : 6]);
    return CHARTED.flatMap((key, n) => {
      const column = history.ratio_keys.indexOf(key);
      if (column < 0) return [];
      return [
        {
          type: "scatter",
          mode: "lines+markers",
          name: history.ratio_labels[column],
          x: periods,
          y: history.ratios.map((r) => r.values[column] ?? null),
          connectgaps: false,
          line: { color: colours[n], width: 2 },
          marker: { size: 4 },
          hovertemplate: `${history.ratio_labels[column]} %{x}: %{y:.2%}<extra></extra>`,
        } satisfies Data,
      ];
    });
  }, [history, periods, c, dark]);

  const layout = useMemo<Partial<Layout>>(
    () => ({
      margin: { t: 8, r: 8, b: 40, l: 48 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: p.font, color: p.text, size: 11 },
      hoverlabel: { font: { family: p.font, size: 11 } },
      legend: { orientation: "h", y: -0.2, bgcolor: "rgba(0,0,0,0)" },
      xaxis: { type: "category", tickfont: { size: 10 }, tickangle: -45 },
      yaxis: { gridcolor: p.grid, zerolinecolor: p.zero, tickformat: ".0%", tickfont: { size: 10 } },
    }),
    [p.font, p.text, p.grid, p.zero],
  );
  return <PlotlyChart data={traces} layout={layout} height={340} />;
}

function Filings({ filings }: { filings: History["filings"] }) {
  if (filings.length === 0) return null;
  const rows = [...filings].reverse();
  return (
    <Card p={0}>
      <Text size="sm" fw={600} px="md" pt="md">
        The filing behind each fiscal year
      </Text>
      <Table.ScrollContainer minWidth={560} maxHeight={420}>
        <Table fz="xs" verticalSpacing={5} striped stickyHeader>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Fiscal year</Table.Th>
              <Table.Th>Period</Table.Th>
              <Table.Th>Report date</Table.Th>
              <Table.Th>Published</Table.Th>
              <Table.Th>Restated</Table.Th>
              <Table.Th>SEC</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {rows.map((filing) => (
              <Table.Tr key={`${filing.fiscal_year}-${filing.period}`}>
                <Table.Td>{filing.fiscal_year}</Table.Td>
                <Table.Td>{filing.period}</Table.Td>
                <Table.Td>{filing.report_date ? isoDay(filing.report_date) : MISSING}</Table.Td>
                <Table.Td>{filing.publish_date ? isoDay(filing.publish_date) : MISSING}</Table.Td>
                <Table.Td>
                  {filing.restated && (
                    <Badge size="xs" variant="light" color="gray">
                      restated
                    </Badge>
                  )}
                </Table.Td>
                <Table.Td>
                  {filing.link ? (
                    <Anchor href={filing.link} target="_blank" rel="noreferrer" size="xs">
                      Filings <IconExternalLink size={11} stroke={1.6} />
                    </Anchor>
                  ) : (
                    MISSING
                  )}
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
      <Text size="xs" c="dimmed" px="md" pb="md" pt="xs">
        Report date is the period end; published is when the filing reached Sharadar. A restated year is one the company
        later revised — this model always uses the restated figures, so the series the drivers are averaged over is a
        consistent one.
      </Text>
    </Card>
  );
}

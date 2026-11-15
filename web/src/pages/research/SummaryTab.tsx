// 📊 Summary — the price window, the DCF line over it, and the key statistics.
//
// The candles come from the snapshot and draw at once. The DCF line is a second,
// slower request (a model run per filing, cached one by one), so it joins the
// chart when it is valued and never holds up the price.
import {
  Alert,
  Badge,
  Box,
  Card,
  Group,
  Loader,
  LoadingOverlay,
  SegmentedControl,
  SimpleGrid,
  Skeleton,
  Stack,
  Table,
  Text,
  Title,
  Tooltip,
  useComputedColorScheme,
  useMantineTheme,
} from "@mantine/core";
import { useMediaQuery } from "@mantine/hooks";
import { IconAlertTriangle } from "@tabler/icons-react";
import type { Data, Layout } from "plotly.js";
import { useMemo } from "react";

import type { Schemas } from "../../api/client";
import { chartPalette } from "../../components/chartPalette";
import { PlotlyChart } from "../../components/PlotlyChart";
import { isoDay, MISSING } from "../../format";
import { type Period, price, signedPercent, toneColor, useDcfLine, useResearchParams, useSummary } from "./common";
import { NotReported, OriginBadge, OriginLegend, plural, Segments, SourcesPanel } from "./parts";

type Summary = Schemas["Summary"];
type DcfLine = Schemas["DcfLine"];

const PERIODS = ["5D", "1M", "3M", "6M", "YTD", "1Y", "5Y", "ALL"] as const satisfies readonly Period[];
const DEFAULT_PERIOD: Period = "1Y";
const CHART_HEIGHT = 560;

/** The DCF line's colour, shared by the line and the swatch beside its headline. */
function useAmber() {
  const theme = useMantineTheme();
  const dark = useComputedColorScheme("dark") === "dark";
  return theme.colors.yellow[dark ? 5 : 7];
}

function asPeriod(value: string | null): Period {
  return (PERIODS as readonly string[]).includes(value ?? "") ? (value as Period) : DEFAULT_PERIOD;
}

export function SummaryTab({ ticker }: { ticker: string }) {
  const { params, set } = useResearchParams();
  const period = asPeriod(params.get("period"));
  const summary = useSummary(ticker, period);
  // Eight windows fit a phone's width only at the small size.
  const phone = useMediaQuery("(max-width: 30em)");
  const dcf = useDcfLine(ticker, period);

  return (
    <Stack gap="lg">
      {summary.isPending && <Skeleton h={CHART_HEIGHT + 90} radius="md" />}
      {summary.isError && (
        <Alert variant="light" color="red" icon={<IconAlertTriangle size={18} />} title="Couldn't load the summary">
          {summary.error.message}
        </Alert>
      )}
      {summary.data && (
        <>
          <Card pos="relative">
            <LoadingOverlay visible={summary.isPlaceholderData} zIndex={2} overlayProps={{ blur: 1 }} />
            <Group justify="space-between" align="flex-end" wrap="wrap" gap="md" mb="sm">
              <Headline summary={summary.data} dcf={dcf.data} dcfPending={dcf.isPending} />
              <Segments>
                <SegmentedControl
                  data={[...PERIODS]}
                  size={phone ? "xs" : "sm"}
                  value={period}
                  onChange={(value) => set({ period: value === DEFAULT_PERIOD ? null : value })}
                />
              </Segments>
            </Group>
            {summary.data.candles.sessions.length === 0 ? (
              <Alert variant="light" color="gray">
                No price history in the {period} window.
              </Alert>
            ) : (
              <>
                <PriceChart summary={summary.data} dcf={dcf.data} />
                <ChartNotes summary={summary.data} dcf={dcf} />
              </>
            )}
          </Card>
          <KeyStatistics summary={summary.data} />
          <Sources summary={summary.data} />
        </>
      )}
    </Stack>
  );
}

/** The latest close in the window, its change over the window, and today's DCF value beside it. */
function Headline({ summary, dcf, dcfPending }: { summary: Summary; dcf?: DcfLine; dcfPending: boolean }) {
  const closes = summary.candles.close;
  let last: number | null = null;
  for (let i = closes.length - 1; i >= 0 && last === null; i--) last = closes[i] ?? null;
  const valued = dcf?.steps.filter((step) => step.value !== null) ?? [];
  const latest = valued[valued.length - 1];
  const amber = useAmber();

  return (
    <Group gap="xl" align="flex-end" wrap="wrap">
      <div>
        <Text size="xs" fw={700} c="dimmed" tt="uppercase" lts={0.4}>
          Close · {summary.period}
        </Text>
        <Group gap="xs" align="baseline" wrap="nowrap">
          <Text fz={28} fw={700} lh={1.2} className="num">
            {last === null ? MISSING : price(last, summary.currency)}
          </Text>
          {summary.change !== null ? (
            <Tooltip label={summary.change_formula ?? "Change over the window"} withArrow disabled={!summary.change_formula}>
              <Badge size="lg" variant="light" color={toneColor(summary.change === 0 ? null : summary.change > 0)} className="num">
                {signedPercent(summary.change, 2)}
              </Badge>
            </Tooltip>
          ) : (
            <NotReported reason={summary.change_reason} />
          )}
        </Group>
      </div>
      <div>
        <Group gap={6} wrap="nowrap">
          <span style={{ width: 10, height: 3, borderRadius: 2, background: amber }} />
          <Text size="xs" fw={700} c="dimmed" tt="uppercase" lts={0.4}>
            DCF value{latest && latest.label !== "today" ? " · latest filing" : ""}
          </Text>
        </Group>
        {dcfPending ? (
          <Skeleton h={28} w={110} mt={4} />
        ) : latest ? (
          <Text fz={28} fw={700} lh={1.2} className="num">
            {price(latest.value!, summary.currency)}
          </Text>
        ) : (
          <Text size="sm" c="dimmed" fs="italic" lh={2.4}>
            not available
          </Text>
        )}
      </div>
    </Group>
  );
}

function PriceChart({ summary, dcf }: { summary: Summary; dcf?: DcfLine }) {
  const theme = useMantineTheme();
  const dark = useComputedColorScheme("dark") === "dark";
  const p = chartPalette(theme, dark);
  const c = theme.colors;
  const up = c.teal[dark ? 5 : 6];
  const down = c.red[dark ? 5 : 6];
  const amber = useAmber();

  const traces = useMemo<Data[]>(() => {
    const k = summary.candles;
    const out: Data[] = [
      {
        type: "candlestick",
        name: "Price",
        x: k.sessions,
        open: k.open,
        high: k.high,
        low: k.low,
        close: k.close,
        increasing: { line: { color: up, width: 1 }, fillcolor: up },
        decreasing: { line: { color: down, width: 1 }, fillcolor: down },
        showlegend: false,
      } as Data,
      {
        type: "bar",
        name: "Volume",
        x: k.sessions,
        y: k.volume,
        yaxis: "y2",
        showlegend: false,
        marker: {
          color: k.close.map((close, i) => ((close ?? 0) >= (k.open[i] ?? 0) ? up : down)),
          opacity: 0.45,
        },
        hovertemplate: "Volume %{y:,.0f}<extra></extra>",
      },
    ];
    const steps = dcf?.steps ?? [];
    if (steps.some((step) => step.value !== null)) {
      const valued = steps.filter((step) => step.value !== null);
      const latest = valued[valued.length - 1]!;
      out.push({
        type: "scatter",
        mode: "lines+markers",
        // The legend carries the value a reader looks for: today's, or the last filing's.
        name: `DCF value · ${price(latest.value!, summary.currency)} ${latest.label === "today" ? "today" : "latest"}`,
        x: steps.map((step) => step.session),
        y: steps.map((step) => step.value),
        customdata: steps.map((step) => step.label),
        line: { shape: "hv", color: amber, width: 2 },
        marker: { size: 5, color: amber },
        connectgaps: false,
        hovertemplate: "DCF value %{y:,.2f}<br>%{customdata}<extra></extra>",
      });
    }
    return out;
  }, [summary, dcf, up, down, amber]);

  const layout = useMemo<Partial<Layout>>(
    () => ({
      margin: { t: 8, r: 56, b: 28, l: 8 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: p.font, color: p.text, size: 11 },
      hovermode: "x unified",
      hoverlabel: { font: { family: p.font, size: 11 } },
      bargap: 0.2,
      showlegend: true,
      legend: {
        x: 0.01,
        y: 0.99,
        xanchor: "left",
        yanchor: "top",
        bgcolor: dark ? "rgba(26,27,30,0.8)" : "rgba(255,255,255,0.8)",
        font: { color: amber, size: 12 },
      },
      xaxis: {
        anchor: "y2",
        showgrid: false,
        rangeslider: { visible: false },
        // Without the weekend breaks every Saturday and Sunday draws a gap.
        rangebreaks: [{ bounds: ["sat", "mon"] }],
        showspikes: true,
        spikemode: "across",
        spikethickness: 1,
        spikedash: "dot",
        spikecolor: p.muted,
        tickfont: { size: 10 },
      },
      yaxis: {
        domain: [0.26, 1],
        side: "right",
        gridcolor: p.grid,
        zeroline: false,
        title: { text: summary.currency || "Price", font: { size: 10 } },
        tickfont: { size: 10 },
      },
      yaxis2: {
        domain: [0, 0.2],
        side: "right",
        gridcolor: p.grid,
        zeroline: false,
        tickformat: "~s",
        tickfont: { size: 10 },
        title: { text: "Volume", font: { size: 10 } },
      },
    }),
    [p.font, p.text, p.muted, p.grid, dark, amber, summary.currency],
  );

  return <PlotlyChart data={traces} layout={layout} height={CHART_HEIGHT} />;
}

function ChartNotes({ summary, dcf }: { summary: Summary; dcf: ReturnType<typeof useDcfLine> }) {
  const first = summary.candles.sessions[0];
  const line = dcf.data;
  const gaps = line?.steps.filter((step) => step.value === null) ?? [];
  return (
    <Stack gap={6} mt="sm">
      {dcf.isPending && (
        <Group gap="xs">
          <Loader size={14} />
          <Text size="xs" c="dimmed">
            Valuing the company at each filing in the window…
          </Text>
        </Group>
      )}
      {dcf.isError && (
        <Text size="xs" c="red">
          No DCF line: {dcf.error.message}
        </Text>
      )}
      {line && line.steps.length === 0 && (
        <Text size="xs" c="dimmed">
          No DCF line: {line.reason}
        </Text>
      )}
      {line && line.steps.length > 0 && (
        <Text size="xs" c="dimmed">
          <b>DCF value</b> (amber) is the DCF's default headline — Gordon growth, every assumption untouched — re-struck
          at each filing as the company stood that day: the statements filed by then, that day's price, 10-year yield and
          beta. Today's equity risk premium and credit-spread table stand in for their past values. The last step is
          today's.
          {line.start && first && line.start > first && ` The line reaches back ${line.max_years} years, to ${isoDay(line.start)}.`}
        </Text>
      )}
      {gaps.map((step) => (
        <Text key={step.session} size="xs" c="dimmed">
          No DCF value for {step.label}: {step.reason}
        </Text>
      ))}
      {summary.period === "5D" && (
        <Text size="xs" c="dimmed">
          Sharadar publishes end-of-day prices only — 5D is five trading sessions, not intraday.
        </Text>
      )}
    </Stack>
  );
}

/** The statistics laid out by the API's grid — columns of labels — so the grouping is declared once. */
function KeyStatistics({ summary }: { summary: Summary }) {
  const byLabel = new Map(summary.stats.map((stat) => [stat.label, stat]));
  return (
    <Card>
      <Title order={4} mb="xs">
        Key statistics
      </Title>
      <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="xl" verticalSpacing={0}>
        {summary.grid.map((column, i) => (
          <Box key={i}>
            {column.map((label) => {
              const stat = byLabel.get(label);
              return (
                <Group
                  key={label}
                  justify="space-between"
                  align="baseline"
                  wrap="nowrap"
                  gap="sm"
                  py={9}
                  style={{ borderTop: "1px solid var(--mantine-color-default-border)" }}
                >
                  <Text size="sm" c="dimmed" style={{ minWidth: 0 }}>
                    {label}
                  </Text>
                  {stat?.display ? (
                    <Text size="sm" fw={600} ta="right" className="num" style={{ overflowWrap: "anywhere" }}>
                      {stat.display}
                    </Text>
                  ) : (
                    <NotReported reason={stat?.provenance.detail} />
                  )}
                </Group>
              );
            })}
          </Box>
        ))}
      </SimpleGrid>
    </Card>
  );
}

function Sources({ summary }: { summary: Summary }) {
  const missing = summary.stats.filter((stat) => stat.display === null).length;
  return (
    <SourcesPanel
      title={`Sources & data status — ${plural(summary.stats.length, "figure")}${missing ? `, ${missing} unavailable` : ""}`}
    >
      <Table.ScrollContainer minWidth={860}>
        <Table fz="xs" verticalSpacing={6} highlightOnHover>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Statistic</Table.Th>
              <Table.Th>Value</Table.Th>
              <Table.Th>Status</Table.Th>
              <Table.Th>Source</Table.Th>
              <Table.Th>As of</Table.Th>
              <Table.Th>Notes</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {summary.stats.map((stat) => (
              <Table.Tr key={stat.label}>
                <Table.Td fw={600} style={{ whiteSpace: "nowrap" }}>
                  {stat.label}
                </Table.Td>
                <Table.Td className="num">{stat.display ?? <NotReported size="xs" />}</Table.Td>
                <Table.Td>
                  <OriginBadge origin={stat.provenance.origin} />
                </Table.Td>
                <Table.Td>{stat.provenance.source || MISSING}</Table.Td>
                <Table.Td style={{ whiteSpace: "nowrap" }}>
                  {stat.provenance.as_of ? isoDay(stat.provenance.as_of) : MISSING}
                </Table.Td>
                <Table.Td c="dimmed">
                  {[stat.provenance.formula, stat.provenance.detail].filter(Boolean).join(" · ") || MISSING}
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
      <OriginLegend estimate />
    </SourcesPanel>
  );
}

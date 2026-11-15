// 📈 History — one name's reward:risk over its trailing year, with ATM IV and skew.
//
// Every figure is read at a constant 30 days, so the line moves with the market
// rather than with the expiry calendar. The cards say where today sits in the
// year; under half a year of stored history they say how much there is instead.
import {
  Alert,
  Box,
  Card,
  Center,
  Loader,
  LoadingOverlay,
  SimpleGrid,
  Stack,
  Text,
  Title,
  useComputedColorScheme,
  useMantineTheme,
} from "@mantine/core";
import { IconAlertTriangle, IconInfoCircle } from "@tabler/icons-react";
import type { Data, Layout } from "plotly.js";
import { useMemo } from "react";

import type { Schemas } from "../../api/client";
import { PlotlyChart } from "../../components/PlotlyChart";
import { Stat } from "../../components/Stat";
import { fixed, isoDay, percent } from "../../format";
import { NameSelect } from "./NameSelect";
import { chartPalette, chosenName, type Overview, times, useHistory, useOptionsParams } from "./common";

const QUARTILE_COLOR: Record<string, string> = {
  "worst quartile": "red",
  "below median": "orange",
  "above median": "teal",
  "best quartile": "teal",
};

export function HistoryTab({ overview }: { overview: Overview }) {
  const { params, set } = useOptionsParams();
  const name = chosenName(params, overview.visible) ?? "GLD";
  const history = useHistory(name);

  return (
    <Stack gap="lg">
      <NameSelect overview={overview} value={name} onChange={(symbol) => set({ name: symbol })} />
      {history.isPending && (
        <Center mih={260}>
          <Loader />
        </Center>
      )}
      {history.isError && (
        <Alert variant="light" color="red" icon={<IconAlertTriangle size={18} />} title={`Couldn't load ${name}`}>
          {history.error.message}
        </Alert>
      )}
      {history.data && (
        <Box pos="relative">
          <LoadingOverlay visible={history.isPlaceholderData} zIndex={2} overlayProps={{ blur: 1 }} />
          <HistoryBody data={history.data} />
        </Box>
      )}
    </Stack>
  );
}

function HistoryBody({ data }: { data: Schemas["OptionsHistory"] }) {
  const stats = data.stats;
  return (
    <Stack gap="lg">
      {stats ? (
        <SimpleGrid cols={{ base: 2, sm: 3, lg: 5 }}>
          <Stat
            label={isoDay(stats.as_of)}
            value={times(stats.latest, 2)}
            hint={
              <Text size="xs" c={QUARTILE_COLOR[stats.quartile] ?? "dimmed"} fw={600}>
                {stats.quartile}
              </Text>
            }
          />
          <Stat label="Percentile of own year" value={percent(stats.percentile, 0)} />
          <Stat label="Median" value={times(stats.median, 2)} />
          <Stat label="Middle half" value={`${fixed(stats.q25)}–${fixed(stats.q75)}`} />
          <Stat label={`Range · ${stats.sessions} sessions`} value={`${fixed(stats.low)}–${fixed(stats.high)}`} />
        </SimpleGrid>
      ) : (
        <Alert variant="light" color="blue" icon={<IconInfoCircle size={18} />}>
          {data.stored} of the {data.needed} sessions needed for a percentile are stored. The collector adds one a
          night.
        </Alert>
      )}
      <Card>
        <Title order={4}>Reward:risk — {data.symbol}</Title>
        <Text size="sm" c="dimmed" mb="xs">
          The +1 SD call spread's payout ÷ the payout a flat smile at ATM IV gives, read off the smile at a constant 30
          days. Above 1.0× the spread costs less than flat vol says it should.
        </Text>
        <HistoryChart data={data} />
      </Card>
    </Stack>
  );
}

function HistoryChart({ data }: { data: Schemas["OptionsHistory"] }) {
  const theme = useMantineTheme();
  const dark = useComputedColorScheme("dark") === "dark";
  const p = chartPalette(theme, dark);
  const c = theme.colors;
  const stats = data.stats;

  const traces = useMemo<Data[]>(() => {
    const x = data.points.map((point) => point.session);
    return [
      {
        type: "scatter",
        mode: "lines",
        name: "Reward:risk",
        x,
        y: data.points.map((point) => point.reward_risk),
        line: { color: c.indigo[dark ? 4 : 6], width: 2 },
        hovertemplate: "%{y:.2f}×<extra>Reward:risk</extra>",
      },
      {
        type: "scatter",
        mode: "lines",
        name: "30d ATM IV (%)",
        x,
        y: data.points.map((point) => (point.atm_iv === null ? null : point.atm_iv * 100)),
        yaxis: "y2",
        line: { color: c.teal[dark ? 5 : 6], width: 1.2, dash: "dot" },
        hovertemplate: "%{y:.1f}%<extra>30d ATM IV</extra>",
      },
      {
        type: "scatter",
        mode: "lines",
        name: "25Δ call skew (vol pts)",
        x,
        y: data.points.map((point) => (point.skew === null ? null : point.skew * 100)),
        yaxis: "y2",
        line: { color: c.yellow[dark ? 5 : 7], width: 1.2, dash: "dot" },
        hovertemplate: "%{y:.1f} pts<extra>25Δ call skew</extra>",
      },
    ];
  }, [data.points, c, dark]);

  const layout = useMemo<Partial<Layout>>(
    () => ({
      margin: { t: 12, r: 56, b: 36, l: 56 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: p.font, color: p.text, size: 12 },
      hovermode: "x unified",
      xaxis: { gridcolor: p.grid, showspikes: false },
      yaxis: { title: { text: "Reward : risk" }, gridcolor: p.grid, zeroline: false },
      // Plotly syncs an overlaying axis's ticks to the primary's grid, which
      // lands them on values like 21.33; the right axis has no grid to match.
      yaxis2: {
        title: { text: "Surface (%)" },
        overlaying: "y",
        side: "right",
        showgrid: false,
        zeroline: false,
        tickmode: "auto",
        tickformat: ".0f",
      },
      legend: { orientation: "h", y: -0.12, bgcolor: "rgba(0,0,0,0)" },
      shapes: stats
        ? [
            {
              type: "rect",
              xref: "paper",
              x0: 0,
              x1: 1,
              yref: "y",
              y0: stats.q25,
              y1: stats.q75,
              fillcolor: c.orange[5],
              opacity: dark ? 0.12 : 0.1,
              line: { width: 0 },
              layer: "below",
            },
            {
              type: "line",
              xref: "paper",
              x0: 0,
              x1: 1,
              yref: "y",
              y0: stats.median,
              y1: stats.median,
              line: { color: p.zero, width: 1, dash: "dash" },
            },
          ]
        : [],
      annotations: stats
        ? [
            {
              xref: "paper",
              x: 0,
              xanchor: "left",
              yref: "y",
              y: stats.median,
              yanchor: "bottom",
              text: `median ${fixed(stats.median)}×`,
              showarrow: false,
              font: { size: 11, color: p.muted },
            },
          ]
        : [],
      hoverlabel: { font: { family: p.font } },
    }),
    [stats, p.font, p.text, p.grid, p.zero, p.muted, c, dark],
  );

  return <PlotlyChart data={traces} layout={layout} height={460} />;
}

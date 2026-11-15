// 🔭 Discover — where calls are cheap against each name's own year.
//
// The scatter puts the buy-vol score against a percentile of the reader's
// choosing; the table under it is the one-month 25Δ call on every visible name.
// A name with too short a history has no score and is listed under the chart
// rather than plotted at zero.
import {
  Alert,
  Box,
  Card,
  Center,
  Group,
  Loader,
  LoadingOverlay,
  SegmentedControl,
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
import { PlotlyChart } from "../../components/PlotlyChart";
import { fixed, isoDay, MISSING, percent, signed } from "../../format";
import { chartPalette, type Filters, type Overview, strike, themeColors, useDiscover, useOptionsParams } from "./common";
import { NameTd, NumTd, SortTh, useSorted } from "./table";

type Row = Schemas["DiscoverRow"];
type PercentileField = "skew_pct" | "iv_pct" | "rr_pct";

// `short` labels the control, which has to fit a phone; `label` titles the axis.
const Y_AXES: { value: string; short: string; label: string; field: PercentileField }[] = [
  { value: "skew", short: "25Δ skew", label: "25Δ call-skew percentile", field: "skew_pct" },
  { value: "iv", short: "30d IV", label: "30d IV percentile", field: "iv_pct" },
  { value: "rr", short: "Reward:risk", label: "Reward:risk percentile", field: "rr_pct" },
];

export function DiscoverTab({ filters, overview }: { filters: Filters; overview: Overview }) {
  const { params, set } = useOptionsParams();
  const discover = useDiscover(filters);
  const axis = Y_AXES.find((a) => a.value === params.get("y")) ?? Y_AXES[0]!;
  const theme = useMantineTheme();
  const colors = useMemo(() => themeColors(theme, overview.themes), [theme, overview.themes]);

  if (discover.isPending)
    return (
      <Center mih={300}>
        <Loader />
      </Center>
    );
  if (discover.isError)
    return (
      <Alert variant="light" color="red" icon={<IconAlertTriangle size={18} />} title="Couldn't load Discover">
        {discover.error.message}
      </Alert>
    );

  const rows = discover.data.rows;
  const unplotted = rows.filter((r) => r.score === null || r[axis.field] === null).map((r) => r.symbol);
  return (
    <Stack gap="lg" pos="relative">
      <LoadingOverlay visible={discover.isPlaceholderData} zIndex={2} overlayProps={{ blur: 1 }} />
      <Card>
        <Group justify="space-between" align="flex-start" wrap="wrap" gap="md" mb="sm">
          <div>
            <Title order={4}>Cheap against its own year</Title>
            <Text size="sm" c="dimmed">
              Right is a higher buy-vol score — calls cheaper than usual for that name. Up is where today's reading sits
              in the name's trailing year.
            </Text>
          </div>
          <SegmentedControl
            size="xs"
            data={Y_AXES.map(({ value, short }) => ({ value, label: `${short} %ile` }))}
            value={axis.value}
            onChange={(value) => set({ y: value === "skew" ? null : value })}
          />
        </Group>
        <Scatter rows={rows} field={axis.field} label={axis.label} colors={colors} />
        {unplotted.length > 0 && (
          <Text size="xs" c="dimmed" mt="xs">
            Not plotted — under {overview.min_history_sessions} stored sessions of history, so no score yet:{" "}
            {unplotted.join(", ")}
          </Text>
        )}
      </Card>
      <CallsCard rows={rows} colors={colors} session={discover.data.session} />
    </Stack>
  );
}

function Scatter({
  rows,
  field,
  label,
  colors,
}: {
  rows: Row[];
  field: PercentileField;
  label: string;
  colors: Map<string, string>;
}) {
  const theme = useMantineTheme();
  const dark = useComputedColorScheme("dark") === "dark";
  const p = chartPalette(theme, dark);
  // On a phone a legend of ~30 themes leaves the plot a sliver, so there is
  // none: the table under the chart marks each name's theme in its colour.
  const narrow = useMediaQuery(`(max-width: ${theme.breakpoints.sm})`) ?? false;

  const data = useMemo<Data[]>(() => {
    const plotted = rows.filter((r) => r.score !== null && r[field] !== null);
    const themes = [...new Set(plotted.map((r) => r.theme))].sort();
    return themes.map((name) => {
      const group = plotted.filter((r) => r.theme === name);
      return {
        type: "scatter",
        mode: "text+markers",
        name,
        x: group.map((r) => r.score),
        y: group.map((r) => (r[field] ?? 0) * 100),
        text: group.map((r) => r.symbol),
        textposition: "top center",
        textfont: { size: 10, color: p.text },
        marker: { size: 9, color: colors.get(name), line: { width: 0 } },
        hovertemplate: `%{text}<br>score %{x:.0f}<br>${label} %{y:.0f}<extra>${name}</extra>`,
      };
    });
  }, [rows, field, label, colors, p.text]);

  const layout = useMemo<Partial<Layout>>(
    () => ({
      margin: { t: 8, r: 8, b: 44, l: 48 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: p.font, color: p.text, size: 12 },
      xaxis: {
        title: { text: narrow ? "Buy-vol score" : "Buy-vol score (higher = calls cheaper than usual)" },
        gridcolor: p.grid,
        zeroline: false,
      },
      yaxis: { title: { text: label }, range: [-3, 103], gridcolor: p.grid, zeroline: false },
      showlegend: !narrow,
      legend: { font: { size: 11 }, bgcolor: "rgba(0,0,0,0)" },
      shapes: [
        { type: "line", x0: 0, x1: 0, yref: "paper", y0: 0, y1: 1, line: { color: p.zero, width: 1, dash: "dash" } },
      ],
      hoverlabel: { font: { family: p.font } },
    }),
    [label, narrow, p.font, p.text, p.grid, p.zero],
  );

  return <PlotlyChart data={data} layout={layout} height={narrow ? 460 : 540} />;
}

type CallKey =
  | "symbol"
  | "spot"
  | "score"
  | "z_iv"
  | "z_iv_rv"
  | "z_skew"
  | "expiration"
  | "strike"
  | "delta"
  | "cost"
  | "cost_pct"
  | "annualised"
  | "otm"
  | "iv";

function callValue(row: Row, key: CallKey): number | string | null | undefined {
  switch (key) {
    case "symbol":
    case "spot":
    case "score":
    case "z_iv":
    case "z_iv_rv":
    case "z_skew":
      return row[key];
    default:
      return row.call?.[key];
  }
}

function CallsCard({ rows, colors, session }: { rows: Row[]; colors: Map<string, string>; session: string }) {
  const { sorted, sort, toggle } = useSorted<Row, CallKey>(rows, { key: "score", descending: true }, callValue);
  const th = (label: string, column: CallKey, help?: string) => (
    <SortTh label={label} column={column} sort={sort} onSort={toggle} help={help} />
  );
  return (
    <Card>
      <Title order={4}>One-month 25Δ calls</Title>
      <Text size="sm" c="dimmed" mb="sm">
        The listed call nearest 25Δ on the standard monthly nearest 30 days, priced at its mid on the {isoDay(session)}{" "}
        close.
      </Text>
      <Table.ScrollContainer minWidth={1180}>
        <Table striped highlightOnHover verticalSpacing={6} horizontalSpacing="sm">
          <Table.Thead>
            <Table.Tr>
              <SortTh label="Name" column="symbol" sort={sort} onSort={toggle} align="left" />
              {th("Spot", "spot")}
              {th("Score", "score", "The buy-vol score: higher means calls are cheaper than usual for this name.")}
              {th("z IV", "z_iv", "Today's 30d ATM IV against its own year.")}
              {th("z IV−RV", "z_iv_rv", "30d IV less 21-session realized vol, against its own year.")}
              {th("z skew", "z_skew", "25Δ call IV less ATM IV, against its own year.")}
              {th("Expiry", "expiration")}
              {th("Strike", "strike")}
              {th("Δ", "delta")}
              {th("Cost", "cost", "Mid of the 17:15 ET NBBO. ° — the bid/ask is one-sided or wider than half its mid.")}
              {th("Cost %", "cost_pct")}
              {th("Ann.", "annualised", "Cost % × 252 ÷ weekdays to expiry.")}
              {th("OTM", "otm")}
              {th("IV", "iv")}
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {sorted.map((row) => (
              <Table.Tr key={row.symbol}>
                <NameTd symbol={row.symbol} theme={row.theme} color={colors.get(row.theme)} />
                <NumTd value={row.spot} format={(v) => fixed(v)} />
                <NumTd
                  value={row.score}
                  format={(v) => fixed(v, 1)}
                  color={row.score === null ? undefined : row.score > 0 ? "teal" : row.score < 0 ? "red" : undefined}
                />
                <NumTd value={row.z_iv} format={(v) => signed(v)} />
                <NumTd value={row.z_iv_rv} format={(v) => signed(v)} />
                <NumTd value={row.z_skew} format={(v) => signed(v)} />
                <Table.Td ta="right" style={{ whiteSpace: "nowrap" }}>
                  {row.call ? (
                    isoDay(row.call.expiration)
                  ) : (
                    <Text span c="dimmed" inherit>
                      {MISSING}
                    </Text>
                  )}
                </Table.Td>
                <NumTd value={row.call?.strike} format={strike} />
                <NumTd value={row.call?.delta} format={(v) => fixed(v)} />
                <NumTd
                  value={row.call?.cost}
                  format={(v) => (
                    <>
                      {fixed(v)}
                      {row.call?.wide && (
                        <Tooltip label="The bid/ask is one-sided or wider than half its mid">
                          <Box component="span" c="yellow" ml={2}>
                            °
                          </Box>
                        </Tooltip>
                      )}
                    </>
                  )}
                />
                <NumTd value={row.call?.cost_pct} format={(v) => percent(v)} />
                <NumTd value={row.call?.annualised} format={(v) => percent(v, 1)} />
                <NumTd value={row.call?.otm} format={(v) => percent(v, 1)} />
                <NumTd value={row.call?.iv} format={(v) => percent(v, 1)} />
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
    </Card>
  );
}

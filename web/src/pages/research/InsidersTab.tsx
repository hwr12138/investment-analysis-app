// 👤 Insiders — open-market form 4 activity, against the price.
//
// The legend is not decoration here. Open-market purchases are about one filed
// row in 120, and Apple went five years without one, so a tab that showed only
// what it counts would look broken at exactly the companies a reader opens
// first. Every excluded row is counted in the panel at the foot.
import {
  Alert,
  Box,
  Card,
  Group,
  SimpleGrid,
  Skeleton,
  Stack,
  Table,
  Text,
  Title,
  useComputedColorScheme,
  useMantineTheme,
} from "@mantine/core";
import { IconAlertTriangle, IconInfoCircle } from "@tabler/icons-react";
import type { Data, Layout } from "plotly.js";
import { useMemo } from "react";

import type { Schemas } from "../../api/client";
import { chartPalette } from "../../components/chartPalette";
import { PlotlyChart } from "../../components/PlotlyChart";
import { fixed, isoDay, MISSING } from "../../format";
import { compactCount, compactMoney, signedMoney, useInsiders } from "./common";
import { FrameTable, NotReported, OriginLegend, plural, SourcesPanel } from "./parts";

type Insiders = Schemas["Insiders"];

/** The largest trade's marker, in pixels; the rest are scaled by area against it. */
const MAX_MARKER_PX = 44;

export function InsidersTab({ ticker }: { ticker: string }) {
  const insiders = useInsiders(ticker);
  const data = insiders.data;
  return (
    <Stack gap="lg">
      {insiders.isPending && <Skeleton h={520} radius="md" />}
      {insiders.isError && (
        <Alert variant="light" color="red" icon={<IconAlertTriangle size={18} />} title="Couldn't load the insider filings">
          {insiders.error.message}
        </Alert>
      )}
      {data && !data.available && (
        <Alert variant="light" color="blue" icon={<IconInfoCircle size={18} />} title={`No insider filings for ${data.ticker}`}>
          Sharadar's <code>SHARADAR/SF2</code> covers SEC forms 3, 4 and 5 from 2008 onward. A company listed through that
          period with no rows has had nobody file — which is an answer, not a gap.
        </Alert>
      )}
      {data && data.available && <InsidersBody data={data} />}
    </Stack>
  );
}

function InsidersBody({ data }: { data: Insiders }) {
  return (
    <>
      <SimpleGrid cols={{ base: 1, sm: 3 }}>
        {data.windows.map((window) => (
          <WindowCard key={window.months} window={window} currency={data.currency} />
        ))}
      </SimpleGrid>
      <Group gap="xs" align="flex-start" wrap="nowrap">
        <Box c="dimmed" pt={2}>
          <IconInfoCircle size={16} stroke={1.6} />
        </Box>
        <Text size="xs" c="dimmed">
          <b>Open-market purchases (P) and sales (S) only.</b> Grants, option exercises and shares withheld for tax are
          excluded: they are compensation rather than a view on the price, and including them reverses the sign of the
          total at most large companies. Rows that restate a holding without reporting a trade are excluded too. Every
          excluded row is counted in <b>Sources & data status</b> at the foot of this tab. A 10% owner is labelled, never
          filtered — the code decides, not the role.
        </Text>
      </Group>
      <Card>
        <Title order={4}>Trades against the price</Title>
        <Text size="sm" c="dimmed" mb="xs">
          The last {data.lookback_years} years. Each marker is one trade at the price paid, its area proportional to the
          dollars; prices are split-adjusted so a trade sits on the line, and the hover keeps the price on the form.
        </Text>
        {data.trades.length === 0 && (
          <Text size="sm" c="dimmed" fs="italic" mb="xs">
            No open-market purchases or sales in the last {data.lookback_years} years.
          </Text>
        )}
        <EventsChart data={data} />
      </Card>
      <Card>
        <Title order={4}>Net open-market dollars per quarter</Title>
        <Text size="sm" c="dimmed" mb="xs">
          Bucketed on the <b>transaction</b> date, not the filing date. The faded bar is a quarter still open. Monthly
          buckets are not offered: they are 41–89% empty depending on the company, and an empty bucket reads as missing
          data rather than as a quiet month.
        </Text>
        <QuarterChart data={data} />
      </Card>
      <Clusters data={data} />
      <Roster data={data} />
      <Sources data={data} />
    </>
  );
}

function WindowCard({ window, currency }: { window: Schemas["InsiderWindow"]; currency: string }) {
  const net = window.net;
  return (
    <Card padding="md">
      <Text size="xs" fw={700} c="dimmed" tt="uppercase" lts={0.4}>
        Net open-market · {window.months}M
      </Text>
      {window.available && net !== null ? (
        <>
          <Text fz={26} fw={700} lh={1.3} className="num" c={net > 0 ? "teal" : net < 0 ? "red" : undefined}>
            {signedMoney(net, currency)}
          </Text>
          <Text size="xs" c="dimmed">
            {plural(window.buy_trades, "buy")} · {plural(window.sell_trades, "sell")} ·{" "}
            {plural(window.participants, "insider")}
          </Text>
          {window.concentrated && window.concentration !== null && (
            <Text size="xs" c="yellow" mt={4}>
              ⚠ {window.top_owner} is {fixed(window.concentration * 100, 0)}% of the gross
            </Text>
          )}
        </>
      ) : (
        <>
          <Text fz={20} lh={1.6} c="dimmed" fs="italic">
            not reported
          </Text>
          <Text size="xs" c="dimmed" lineClamp={3}>
            {window.reason}
          </Text>
        </>
      )}
    </Card>
  );
}

function EventsChart({ data }: { data: Insiders }) {
  const theme = useMantineTheme();
  const dark = useComputedColorScheme("dark") === "dark";
  const p = chartPalette(theme, dark);
  const c = theme.colors;
  const buy = c.teal[dark ? 5 : 6];
  const sell = c.red[dark ? 5 : 6];

  const traces = useMemo<Data[]>(() => {
    const out: Data[] = [
      {
        type: "scatter",
        mode: "lines",
        name: "Close",
        x: data.prices.sessions,
        y: data.prices.close,
        line: { color: dark ? c.gray[5] : c.gray[6], width: 1.2 },
        opacity: 0.8,
        hovertemplate: "%{x|%b %d, %Y}<br>Close %{y:,.2f}<extra></extra>",
      },
    ];
    const largest = Math.max(0, ...data.trades.map((trade) => Math.abs(trade.value)));
    // Scaled against the largest trade in view, so what is on screen sets the scale.
    const sizeref = largest ? (2 * largest) / MAX_MARKER_PX ** 2 : 1;
    for (const [code, name, colour] of [
      ["P", "Insider buy", buy],
      ["S", "Insider sell", sell],
    ] as const) {
      const side = data.trades.filter((trade) => trade.code === code);
      if (side.length === 0) continue;
      out.push({
        type: "scatter",
        mode: "markers",
        name,
        x: side.map((trade) => trade.session),
        y: side.map((trade) => trade.price),
        marker: {
          color: colour,
          // Area, not diameter: the default exaggerates a large trade by the square of its value.
          size: side.map((trade) => Math.abs(trade.value)),
          sizemode: "area",
          sizeref,
          sizemin: 5,
          opacity: 0.75,
          line: { width: 0.5, color: dark ? "rgba(0,0,0,0.5)" : "rgba(255,255,255,0.7)" },
        },
        customdata: side.map((trade) => [
          trade.owner,
          trade.role,
          trade.shares === null ? MISSING : compactCount(trade.shares),
          compactMoney(Math.abs(trade.value), data.currency),
          trade.filed ? isoDay(trade.filed) : MISSING,
          priceLabel(trade),
        ]) as unknown as number[],
        hovertemplate:
          "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{x|%b %d, %Y} · %{customdata[2]} shares @ %{customdata[5]}" +
          "<br>Value %{customdata[3]}<br>Filed %{customdata[4]}<extra></extra>",
      });
    }
    return out;
  }, [data, buy, sell, dark, c]);

  const layout = useMemo<Partial<Layout>>(
    () => ({
      margin: { t: 8, r: 16, b: 32, l: 56 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: p.font, color: p.text, size: 11 },
      // Point events, so "closest": a unified hover would group unrelated trades sharing an x.
      hovermode: "closest",
      hoverlabel: { font: { family: p.font, size: 11 } },
      legend: { orientation: "h", y: -0.1, bgcolor: "rgba(0,0,0,0)" },
      xaxis: { showgrid: false, rangebreaks: [{ bounds: ["sat", "mon"] }], tickfont: { size: 10 } },
      yaxis: {
        gridcolor: p.grid,
        zeroline: false,
        title: { text: data.currency ? `Price (${data.currency})` : "Price", font: { size: 10 } },
        tickfont: { size: 10 },
      },
    }),
    [p.font, p.text, p.grid, data.currency],
  );

  return <PlotlyChart data={traces} layout={layout} height={440} />;
}

/** The price paid; across a split, the split-adjusted figure with the form's beside it. */
function priceLabel(trade: Schemas["Trade"]): string {
  if (trade.price === null) return MISSING;
  const plotted = fixed(trade.price);
  if (trade.filed_price === null || Math.abs(trade.price - trade.filed_price) < 0.005) return plotted;
  return `${plotted} split-adjusted (${fixed(trade.filed_price)} as filed)`;
}

function QuarterChart({ data }: { data: Insiders }) {
  const theme = useMantineTheme();
  const dark = useComputedColorScheme("dark") === "dark";
  const p = chartPalette(theme, dark);
  const c = theme.colors;
  const buy = c.teal[dark ? 5 : 6];
  const sell = c.red[dark ? 5 : 6];

  const traces = useMemo<Data[]>(
    () => [
      {
        type: "bar",
        // Categorical labels, so an empty quarter keeps its own slot.
        x: data.quarters.map((q) => q.label),
        y: data.quarters.map((q) => q.net),
        marker: {
          color: data.quarters.map((q) => (q.net >= 0 ? buy : sell)),
          // The open quarter is drawn back, so a short final bar is not read as a collapse.
          opacity: data.quarters.map((q) => (q.partial ? 0.4 : 0.9)),
        },
        customdata: data.quarters.map((q) => [
          signedMoney(q.net, data.currency),
          compactMoney(q.bought, data.currency),
          compactMoney(q.sold, data.currency),
          plural(q.trades, "trade"),
          plural(q.owners, "insider"),
          q.partial ? " · still open" : "",
        ]) as unknown as number[],
        // Gross both ways rides in the hover: netting zero on $50m each way is not a quiet quarter.
        hovertemplate:
          "<b>%{x}</b>%{customdata[5]}<br>Net %{customdata[0]}<br>Bought %{customdata[1]}<br>Sold %{customdata[2]}" +
          "<br>%{customdata[3]}, %{customdata[4]}<extra></extra>",
      },
    ],
    [data, buy, sell],
  );

  const layout = useMemo<Partial<Layout>>(
    () => ({
      margin: { t: 8, r: 16, b: 40, l: 56 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: p.font, color: p.text, size: 11 },
      hoverlabel: { font: { family: p.font, size: 11 } },
      showlegend: false,
      bargap: 0.25,
      xaxis: { type: "category", tickfont: { size: 10 }, tickangle: -45 },
      yaxis: {
        gridcolor: p.grid,
        zerolinecolor: p.zero,
        tickformat: "~s",
        title: { text: data.currency ? `Net (${data.currency})` : "Net", font: { size: 10 } },
        tickfont: { size: 10 },
      },
    }),
    [p.font, p.text, p.grid, p.zero, data.currency],
  );

  return <PlotlyChart data={traces} layout={layout} height={280} />;
}

/** Several insiders buying at once. Absence is stated with the criteria: "none" and "did not look" differ. */
function Clusters({ data }: { data: Insiders }) {
  const criteria = (
    <Text size="xs" c="dimmed">
      {data.cluster_min_buyers} or more insiders each buying at least {compactMoney(data.cluster_min_value, data.currency)}{" "}
      on the open market within {data.cluster_window_days} days. Sales are deliberately not clustered: 10b5-1 plans and
      vesting schedules make executives sell together on dates chosen months earlier, so a sell cluster measures the plan
      rather than a view.
    </Text>
  );
  return (
    <Card>
      <Title order={4} mb="xs">
        Cluster buying
      </Title>
      {data.clusters.length === 0 ? (
        <Text size="sm" fw={600} mb={4}>
          None in the last {data.lookback_years} years.
        </Text>
      ) : (
        <Stack gap="sm" mb="sm">
          {data.clusters.map((cluster) => (
            <Box
              key={`${cluster.start}-${cluster.end}`}
              pl="sm"
              style={{ borderLeft: "3px solid var(--mantine-color-teal-6)" }}
            >
              <Text size="sm">
                <b>
                  {cluster.buyers} insiders bought {compactMoney(cluster.value, data.currency)}
                </b>{" "}
                across {plural(cluster.trades, "trade")}
                <Text span size="xs" c="dimmed">
                  {" "}
                  — {isoDay(cluster.start)} to {isoDay(cluster.end)}
                </Text>
              </Text>
              <Text size="xs" c="dimmed">
                {cluster.owners.join(", ")}
              </Text>
            </Box>
          ))}
        </Stack>
      )}
      {criteria}
    </Card>
  );
}

/** Balances as last reported: SF2 records no departure, so no row is a confirmed holding today. */
function Roster({ data }: { data: Insiders }) {
  return (
    <Card>
      <Title order={4} mb="xs">
        Shares held, as last reported
      </Title>
      {data.roster.length === 0 ? (
        <Text size="sm" c="dimmed">
          {data.roster_reason || "No share balances on file."}
        </Text>
      ) : (
        <>
          <Table.ScrollContainer minWidth={560} mah={460}>
            <Table fz="sm" verticalSpacing={6} highlightOnHover stickyHeader>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Insider</Table.Th>
                  <Table.Th>Role</Table.Th>
                  <Table.Th ta="right">Shares</Table.Th>
                  <Table.Th ta="right">As of</Table.Th>
                  <Table.Th ta="right">Last</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {data.roster.map((row) => (
                  <Table.Tr key={row.owner}>
                    <Table.Td fw={500}>{row.owner}</Table.Td>
                    <Table.Td>
                      <Text size="xs" c="dimmed">
                        {row.role}
                      </Text>
                    </Table.Td>
                    <Table.Td ta="right" className="num">
                      {row.shares === null ? <NotReported /> : Math.round(row.shares).toLocaleString("en-US")}
                    </Table.Td>
                    <Table.Td ta="right" className="num" style={{ whiteSpace: "nowrap" }}>
                      {row.as_of ? isoDay(row.as_of) : MISSING}
                    </Table.Td>
                    <Table.Td ta="right">{row.last_code}</Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
          <Text size="xs" c="dimmed" mt="xs">
            Each figure is the balance on that insider's most recent filing, not a confirmed holding today. Option and
            warrant positions are excluded — they are counted in contracts, not shares. {data.roster_reason}
          </Text>
        </>
      )}
    </Card>
  );
}

function Sources({ data }: { data: Insiders }) {
  const unavailable = data.windows.filter((window) => !window.available);
  return (
    <SourcesPanel
      title={`Sources & data status — ${data.shown_rows.toLocaleString("en-US")} of ${plural(data.filed_rows, "filed row")} shown`}
    >
      <Stack gap="sm">
        <Text size="sm">
          <b>Where the figures come from.</b> <code>SHARADAR/SF2</code>, one row per line item on an SEC form 3, 4 or 5.
          Dollar values are <code>transaction shares × price per share</code>, never the reported{" "}
          <code>transactionvalue</code> — Sharadar publishes that unsigned, so summing it counts every sale as a
          purchase, and it is absent on roughly half of all rows.
        </Text>
        <FrameTable frame={data.reconciliation} />
        {unavailable.length > 0 && (
          <div>
            <Text size="sm" fw={600}>
              Windows with no figure
            </Text>
            {unavailable.map((window) => (
              <Text key={window.months} size="xs" c="dimmed">
                <i>{window.months}M</i> — {window.reason}
              </Text>
            ))}
          </div>
        )}
        {data.roster_reason && (
          <Text size="xs" c="dimmed">
            <i>Roster</i> — {data.roster_reason}
          </Text>
        )}
        <div>
          <OriginLegend />
          <Text size="xs" c="dimmed">
            Every figure on this tab is <b>Derived</b>: no single SF2 column is the answer, and the column that looks
            like it is gets the sign wrong.
          </Text>
        </div>
      </Stack>
    </SourcesPanel>
  );
}

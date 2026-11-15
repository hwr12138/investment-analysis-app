// 🏛️ Holdings — who filed a 13F position in the company, as at the latest quarter on file.
//
// Two things the tab says rather than assumes. Nothing here is a position held
// today: 13F is quarterly with a 45-day deadline, so the banner states the
// arithmetic every time and never collapses. And the corrections are shown, not
// applied silently: filing entities are summed into managers and index managers
// are marked, from curated lists printed in the sources panel.
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
  Tooltip,
  useComputedColorScheme,
  useMantineTheme,
} from "@mantine/core";
import { IconAlertTriangle, IconBuildingBank, IconClockExclamation, IconInfoCircle } from "@tabler/icons-react";
import type { Data, Layout } from "plotly.js";
import { useMemo } from "react";

import type { Schemas } from "../../api/client";
import { chartPalette } from "../../components/chartPalette";
import { PlotlyChart } from "../../components/PlotlyChart";
import { fixed, isoDay, MISSING } from "../../format";
import { compactCount, compactMoney, signedPercent, useHoldings } from "./common";
import { FrameTable, plural, SourcesPanel } from "./parts";

type Holdings = Schemas["Holdings"];
type Holder = Schemas["Holder"];

const pct = (value: number, digits = 1) => `${fixed(value * 100, digits)}%`;

export function HoldingsTab({ ticker }: { ticker: string }) {
  const holdings = useHoldings(ticker);
  const data = holdings.data;
  return (
    <Stack gap="lg">
      {holdings.isPending && (
        <Stack gap="xs">
          <Skeleton h={520} radius="md" />
          <Text size="xs" c="dimmed">
            Loading the 13F register — five requests in a chain, and SF3 answers slowly. The slowest tab on the page.
          </Text>
        </Stack>
      )}
      {holdings.isError && (
        <Alert variant="light" color="red" icon={<IconAlertTriangle size={18} />} title="Couldn't load the 13F filings">
          {holdings.error.message}
        </Alert>
      )}
      {data && !data.available && (
        <Alert variant="light" color="blue" icon={<IconBuildingBank size={18} />} title={`No 13F positions on file for ${data.ticker}`}>
          {data.reason}
        </Alert>
      )}
      {data && data.available && <HoldingsBody data={data} />}
    </Stack>
  );
}

function HoldingsBody({ data }: { data: Holdings }) {
  return (
    <>
      <Staleness data={data} />
      <SimpleGrid cols={{ base: 1, xs: 2, lg: 4 }}>
        <Headline
          label="Institutional ownership"
          value={data.owned === null ? null : pct(data.owned)}
          sub={data.units === null ? "" : `${compactCount(data.units)} shares`}
        />
        <Headline
          label="Filers holding"
          value={data.filers === null ? null : Math.round(data.filers).toLocaleString("en-US")}
          sub={data.churn.available ? `${data.churn.opened} opened · ${data.churn.closed} closed` : ""}
        />
        <Headline
          label={`Largest ${data.top_n}`}
          value={data.top_share === null ? null : pct(data.top_share)}
          sub="of reported institutional value"
        />
        <Headline
          label="Reported value"
          value={compactMoney(data.total_value, data.currency)}
          sub={`across ${plural(data.filer_count, "filer")}`}
        />
      </SimpleGrid>
      <Group gap="xs" align="flex-start" wrap="nowrap">
        <Box c="dimmed" pt={2}>
          <IconInfoCircle size={16} stroke={1.6} />
        </Box>
        <Text size="xs" c="dimmed">
          <b>Common stock reported on SEC form 13F.</b> Managers with under $100m in US equities do not file at all, so
          this is who <i>filed</i>, not who owns the company. Long positions only — a reported put is a long put, not
          short interest. Readings above 100% are genuine and are not clamped: a lent share is reported by the lender and
          again by whoever bought it. Position changes are measured on <b>share counts</b>, never on value, which moves
          with the price whether or not anyone traded.
        </Text>
      </Group>
      <Card>
        <Title order={4}>Ownership and filers over time</Title>
        <Text size="sm" c="dimmed" mb="xs">
          Two panels rather than one axis, because the telling case is the two diverging: ownership flat while the filer
          count falls is consolidation.
        </Text>
        <OwnershipChart data={data} />
      </Card>
      <HoldersTable data={data} />
      <Churn data={data} />
      <Sources data={data} />
    </>
  );
}

/** Unconditional, because the fact is: a 13F register is always historical. */
function Staleness({ data }: { data: Holdings }) {
  if (!data.as_of || !data.deadline) return null;
  const age = data.days_old ?? 0;
  const tail =
    age < 0
      ? `and filers have until ${isoDay(data.deadline)} to report it, so the register is still filling in — holders are missing from it today`
      : `and the deadline to report it passed ${plural(age, "day")} ago, so nothing here describes a position held today`;
  return (
    <Alert
      variant="light"
      color="yellow"
      icon={<IconClockExclamation size={18} />}
      title={`Positions as at ${isoDay(data.as_of)}`}
    >
      13F is reported quarterly with a {data.filing_lag_days}-day filing deadline, {tail}.
    </Alert>
  );
}

function Headline({ label, value, sub }: { label: string; value: string | null; sub: string }) {
  return (
    <Card padding="md">
      <Text size="xs" fw={700} c="dimmed" tt="uppercase" lts={0.4}>
        {label}
      </Text>
      {value === null ? (
        <Text fz={20} lh={1.6} c="dimmed" fs="italic">
          not reported
        </Text>
      ) : (
        <Text fz={26} fw={700} lh={1.3} className="num">
          {value}
        </Text>
      )}
      {sub && (
        <Text size="xs" c="dimmed">
          {sub}
        </Text>
      )}
    </Card>
  );
}

function OwnershipChart({ data }: { data: Holdings }) {
  const theme = useMantineTheme();
  const dark = useComputedColorScheme("dark") === "dark";
  const p = chartPalette(theme, dark);
  const c = theme.colors;

  const traces = useMemo<Data[]>(() => {
    const x = data.ownership.map((point) => point.quarter);
    return [
      {
        type: "scatter",
        mode: "lines",
        name: "Ownership",
        x,
        y: data.ownership.map((point) => point.owned),
        line: { color: c.indigo[dark ? 4 : 6], width: 1.8 },
        connectgaps: false,
        hovertemplate: "%{x|%b %d, %Y}<br>%{y:.1%} owned<extra></extra>",
      },
      {
        type: "scatter",
        mode: "lines",
        name: "Filers",
        x,
        y: data.ownership.map((point) => point.holders),
        yaxis: "y2",
        line: { color: dark ? c.gray[5] : c.gray[6], width: 1.4 },
        connectgaps: false,
        hovertemplate: "%{x|%b %d, %Y}<br>%{y:,.0f} filers<extra></extra>",
      },
    ];
  }, [data.ownership, c, dark]);

  const layout = useMemo<Partial<Layout>>(
    () => {
      const label = (text: string, top: number) => ({
        text,
        xref: "paper" as const,
        yref: "paper" as const,
        x: 0,
        y: top,
        xanchor: "left" as const,
        yanchor: "bottom" as const,
        showarrow: false,
        font: { size: 11, color: p.muted },
      });
      return {
        margin: { t: 20, r: 16, b: 28, l: 56 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: p.font, color: p.text, size: 11 },
        hoverlabel: { font: { family: p.font, size: 11 } },
        hovermode: "x unified",
        showlegend: false,
        xaxis: { anchor: "y2", showgrid: false, tickfont: { size: 10 } },
        yaxis: { domain: [0.46, 1], gridcolor: p.grid, zeroline: false, tickformat: ".0%", tickfont: { size: 10 } },
        yaxis2: { domain: [0, 0.36], gridcolor: p.grid, zeroline: false, tickformat: ",d", tickfont: { size: 10 } },
        annotations: [label("Reported institutional ownership", 1), label("Filers holding", 0.36)],
      };
    },
    [p.font, p.text, p.grid, p.muted],
  );

  return <PlotlyChart data={traces} layout={layout} height={400} />;
}

/** One holder's shares across the window. Scaled to its own range, so a large
 *  holder's small move still shows; one stroke per unbroken run, so a quarter it
 *  filed nothing is a gap and never a line drawn across it. */
function Sparkline({ series, color }: { series: (number | null)[]; color: string }) {
  const W = 110;
  const H = 24;
  const PAD = 3;
  const finite = series.flatMap((value, i) => (value === null ? [] : [[i, value] as const]));
  if (finite.length === 0)
    return (
      <Text span size="xs" c="dimmed" fs="italic">
        no history
      </Text>
    );
  const span = Math.max(series.length - 1, 1);
  const lo = Math.min(...finite.map(([, v]) => v));
  const hi = Math.max(...finite.map(([, v]) => v));
  const point = (i: number, v: number): [number, number] => [
    PAD + (i * (W - 2 * PAD)) / span,
    hi === lo ? H / 2 : H - PAD - ((v - lo) / (hi - lo)) * (H - 2 * PAD),
  ];
  const runs: (readonly [number, number])[][] = [];
  for (const entry of finite) {
    const last = runs[runs.length - 1];
    if (last && entry[0] === last[last.length - 1]![0] + 1) last.push(entry);
    else runs.push([entry]);
  }
  const [endX, endY] = point(...finite[finite.length - 1]!);
  return (
    <Tooltip label={`${compactCount(lo)} to ${compactCount(hi)} shares over the window`} withArrow>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" style={{ display: "block", marginLeft: "auto" }}>
        {runs.map((run, i) => {
          const coords = run.map(([index, value]) => point(index, value));
          if (coords.length === 1) {
            const [x, y] = coords[0]!;
            return <circle key={i} cx={x} cy={y} r={1.8} fill={color} />;
          }
          return (
            <polyline
              key={i}
              points={coords.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ")}
              fill="none"
              stroke={color}
              strokeWidth={1.4}
              strokeLinejoin="round"
            />
          );
        })}
        <circle cx={endX} cy={endY} r={2.2} fill={color} />
      </svg>
    </Tooltip>
  );
}

function Change({ change }: { change: number | null }) {
  if (change === null)
    return (
      <Text span size="sm" c="dimmed" fs="italic">
        new
      </Text>
    );
  if (Math.abs(change) < 0.0005)
    return (
      <Text span size="sm" c="dimmed">
        {MISSING}
      </Text>
    );
  return (
    <Text span size="sm" c={change > 0 ? "teal" : "red"} fw={500}>
      {signedPercent(change, 1)}
    </Text>
  );
}

function HoldersTable({ data }: { data: Holdings }) {
  const theme = useMantineTheme();
  const dark = useComputedColorScheme("dark") === "dark";
  const up = theme.colors.teal[dark ? 5 : 6];
  const down = theme.colors.red[dark ? 5 : 6];
  const as_of = data.as_of ? isoDay(data.as_of) : "";
  return (
    <Card>
      <Title order={4} mb="xs">
        The {data.holders.length} largest holders
      </Title>
      <Table.ScrollContainer minWidth={820}>
        <Table verticalSpacing={8} highlightOnHover>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Holder</Table.Th>
              <Table.Th ta="right">Shares</Table.Th>
              <Table.Th ta="right">Δ quarter</Table.Th>
              <Table.Th ta="right">History</Table.Th>
              <Table.Th ta="right">% of shares</Table.Th>
              <Table.Th ta="right">Value</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {data.holders.map((holder: Holder) => (
              <Table.Tr key={holder.name}>
                <Table.Td>
                  <Group gap={8} wrap="nowrap">
                    <Text size="sm" fw={500}>
                      {holder.name}
                    </Text>
                    {holder.is_index && (
                      <Tooltip
                        label={`An index manager${holder.index_house ? ` (${holder.index_house})` : ""}: its changes are fund flows rather than decisions.`}
                        multiline
                        maw={280}
                        withArrow
                      >
                        <Text span size="xs" c="dimmed" style={{ cursor: "help", whiteSpace: "nowrap" }}>
                          ⬤ index
                        </Text>
                      </Tooltip>
                    )}
                  </Group>
                </Table.Td>
                <Table.Td ta="right" className="num">
                  {compactCount(holder.units)}
                </Table.Td>
                <Table.Td ta="right" className="num">
                  <Change change={holder.change} />
                </Table.Td>
                <Table.Td ta="right">
                  <Sparkline series={holder.series} color={(holder.change ?? 0) >= 0 ? up : down} />
                </Table.Td>
                <Table.Td ta="right" className="num">
                  {holder.owned === null ? MISSING : pct(holder.owned, 2)}
                </Table.Td>
                <Table.Td ta="right" className="num">
                  {compactMoney(holder.value, data.currency)}
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
          <Table.Tfoot>
            <Table.Tr>
              <Table.Th>All {data.filer_count.toLocaleString("en-US")} filers</Table.Th>
              <Table.Th ta="right" className="num">
                {compactCount(data.total_units)}
              </Table.Th>
              <Table.Th ta="right" className="num">
                <Change change={data.total_change} />
              </Table.Th>
              <Table.Th />
              <Table.Th />
              <Table.Th ta="right" className="num">
                {compactMoney(data.total_value, data.currency)}
              </Table.Th>
            </Table.Tr>
          </Table.Tfoot>
        </Table>
      </Table.ScrollContainer>
      <Text size="xs" c="dimmed" mt="xs">
        Positions as at {as_of}, over the last {data.periods.length} quarters.{" "}
        <b>Each line is scaled to its own range, so heights are not comparable between rows</b> — the change column
        carries the magnitude. A break in a line is a quarter the manager filed nothing, which is not the same as holding
        nothing. Managers filing under several entities are summed; ⬤ marks an index manager, whose changes are fund
        flows rather than decisions.
      </Text>
    </Card>
  );
}

function Churn({ data }: { data: Holdings }) {
  const churn = data.churn;
  const counts: [string, number][] = [
    ["opened", churn.opened],
    ["closed", churn.closed],
    ["added", churn.added],
    ["trimmed", churn.trimmed],
    ["unchanged", churn.unchanged],
  ];
  return (
    <Card>
      <Title order={4} mb="xs">
        What the register did last quarter
      </Title>
      {!churn.available ? (
        <Text size="sm" c="dimmed">
          {churn.reason}
        </Text>
      ) : (
        <>
          <SimpleGrid cols={{ base: 3, sm: 5 }} spacing="sm" mb="sm">
            {counts.map(([label, count]) => (
              <Box key={label}>
                <Text fz={22} fw={700} lh={1.2} className="num">
                  {count.toLocaleString("en-US")}
                </Text>
                <Text size="xs" c="dimmed">
                  {label}
                </Text>
              </Box>
            ))}
          </SimpleGrid>
          <Text size="xs" c="dimmed">
            A move of less than {pct(data.material_change)} counts as unchanged — 13F counts are rounded and a manager
            reinvesting a dividend has not decided anything. Concentration: the largest {data.top_n} hold{" "}
            {data.top_share === null ? MISSING : pct(data.top_share)} of reported value, Herfindahl index{" "}
            {data.hhi === null ? "not reported" : Math.round(data.hhi).toLocaleString("en-US")}.
          </Text>
        </>
      )}
    </Card>
  );
}

function Sources({ data }: { data: Holdings }) {
  const withCount = data.ownership.filter((point) => point.owned !== null).length;
  return (
    <SourcesPanel
      title={`Sources & data status — ${data.holders.length} of ${plural(data.filer_count, "filer")} shown`}
    >
      <Stack gap="sm">
        <Text size="sm">
          <b>Where the figures come from.</b> <code>SHARADAR/SF3</code> and <code>SF3A</code>, one row per manager per
          quarter from SEC form 13F, back to June 2013. Share counts arrive <b>as filed and are never restated for
          splits</b>, so every count here is put on today's basis before it is compared or divided —{" "}
          <code>SHARADAR/SF1</code>'s share count already is, and mixing the two would be wrong by the split ratio at
          every earlier quarter.
        </Text>
        <FrameTable frame={data.reconciliation} />
        <Text size="sm" fw={600}>
          Two curated lists, applied to the table above
        </Text>
        <Text size="xs" c="dimmed">
          <i>Summed before ranking:</i>{" "}
          {data.families.map((family) => `${family.label} (${plural(family.entities, "filing entity", "filing entities")})`).join(", ")}
          . A manager reorganising its filing entities otherwise shows as a large sale in one and several new positions
          beside it — Apple's Vanguard position read −33% on that reorganisation and −6% once summed. Deriving this from
          the names was tried and abandoned: it matched BANK OF MARIN to BANK OF AMERICA. Any manager not named here
          stands alone.
        </Text>
        <Text size="xs" c="dimmed">
          <i>Marked as index managers:</i> {data.index_managers.join(", ")}. Sharadar publishes no manager type, so this
          is a judgement and not a measurement. It is deliberately narrow — houses running both index and active money
          are left unmarked rather than guessed at, and an unmarked manager is not thereby a claim to be active.
        </Text>
        {data.ownership.length > 0 && (
          <Text size="xs" c="dimmed">
            <i>Ownership history:</i> {plural(data.ownership.length, "quarter")}, {withCount} with a share count to
            divide by.
          </Text>
        )}
        <Text size="xs" c="dimmed">
          <b>Derived</b> is arithmetic over reported figures · <b>Unavailable</b> could not be obtained and is never
          substituted. Ownership, concentration and every position change on this tab are Derived; the filer counts and
          reported values are as filed.
        </Text>
      </Stack>
    </SourcesPanel>
  );
}

// ⚖️ Relative valuation — what the market pays for the company, and how fast
// what it pays for is moving, each against the company's own five years and
// its industry's median.
//
// Valuation and Growth are one table read two ways. Their central rule is
// inverted — only positive multiples count, where a falling growth rate is the
// answer — and the gap is a ratio on one and percentage points on the other, so
// each view words its own verdict and legend; the grid is shared.
import {
  Alert,
  Box,
  Card,
  Group,
  LoadingOverlay,
  SegmentedControl,
  Skeleton,
  Stack,
  Table,
  Text,
  Tooltip,
} from "@mantine/core";
import { IconAlertTriangle, IconInfoCircle } from "@tabler/icons-react";
import type { ReactNode } from "react";

import type { Schemas } from "../../api/client";
import { fixed, isoDay, signed } from "../../format";
import { signedPercent, toneColor, useRelative, useResearchParams } from "./common";
import { NotReported, OriginBadge, OriginLegend, plural, Prose, SourcesPanel } from "./parts";

type Relative = Schemas["Relative"];
type Row = Schemas["ComparisonRow"];
type Kind = Relative["kind"];

export function ValuationTab({ ticker }: { ticker: string }) {
  const { params, set } = useResearchParams();
  const kind: Kind = params.get("view") === "growth" ? "growth" : "valuation";
  const relative = useRelative(ticker, kind);

  return (
    <Stack gap="lg">
      <Group justify="space-between" align="center" wrap="wrap" gap="md">
        <SegmentedControl
          data={[
            { value: "valuation", label: "Valuation" },
            { value: "growth", label: "Growth" },
          ]}
          value={kind}
          onChange={(value) => set({ view: value === "valuation" ? null : value })}
        />
        {relative.data && <Heading relative={relative.data} />}
      </Group>
      {relative.isPending && (
        <Stack gap="xs">
          <Skeleton h={420} radius="md" />
          <Text size="xs" c="dimmed">
            Loading the peer universe — every listed company, priced today. The first company of the day pays for it;
            every one after reads it from the cache.
          </Text>
        </Stack>
      )}
      {relative.isError && (
        <Alert variant="light" color="red" icon={<IconAlertTriangle size={18} />} title="Couldn't load the comparison">
          {relative.error.message}
        </Alert>
      )}
      {relative.data && (
        <>
          <Warnings relative={relative.data} />
          <Card pos="relative" p={0}>
            <LoadingOverlay visible={relative.isPlaceholderData} zIndex={2} overlayProps={{ blur: 1 }} />
            <ComparisonTable relative={relative.data} />
          </Card>
          <Legend relative={relative.data} />
          <Sources relative={relative.data} />
        </>
      )}
    </Stack>
  );
}

/** What the medians are taken over, said before the numbers are read. */
function Heading({ relative }: { relative: Relative }) {
  if (!relative.universe_loaded || !relative.industry) return null;
  return (
    <Text size="sm" c="dimmed">
      <Text span fw={600} c="var(--mantine-color-text)">
        {relative.name}
      </Text>{" "}
      against{" "}
      <Text span fw={600} c="var(--mantine-color-text)">
        {relative.industry}
      </Text>{" "}
      — {plural(relative.peers, "listed company", "listed companies")}
      {relative.priced && `, priced ${isoDay(relative.priced)}`}
    </Text>
  );
}

function Warnings({ relative }: { relative: Relative }) {
  if (!relative.universe_loaded) {
    return (
      <Alert variant="light" color="yellow" icon={<IconAlertTriangle size={18} />}>
        The peer universe could not be loaded, so the industry medians are unavailable. The company's own current and
        five-year figures below are unaffected.
      </Alert>
    );
  }
  if (!relative.industry) {
    return (
      <Alert variant="light" color="yellow" icon={<IconAlertTriangle size={18} />}>
        Sharadar publishes no industry classification for {relative.ticker}, so there is no peer group to compare
        against. The company's own history below is unaffected.
      </Alert>
    );
  }
  return null;
}

const value = (row: Row, v: number) => (row.percent ? `${fixed(v * 100, 2)}%` : `${fixed(v, 2)}×`);

/** A multiple's gap is a ratio (+25% dearer); a rate's is a difference in points. */
const gapText = (kind: Kind, gap: number) => (kind === "valuation" ? signedPercent(gap, 0) : `${signed(gap * 100, 1)} pp`);

function Sub({ children }: { children: ReactNode }) {
  return (
    <Text size="xs" c="dimmed" lh={1.3}>
      {children}
    </Text>
  );
}

function ComparisonTable({ relative }: { relative: Relative }) {
  return (
    <Table.ScrollContainer minWidth={720}>
      <Table verticalSpacing="sm" horizontalSpacing="md" highlightOnHover layout="fixed">
        <colgroup>
          <col style={{ width: "24%" }} />
          <col style={{ width: "14%" }} />
          <col style={{ width: "15%" }} />
          <col style={{ width: "15%" }} />
          <col style={{ width: "16%" }} />
          <col style={{ width: "16%" }} />
        </colgroup>
        <Table.Thead>
          <Table.Tr>
            {["Metric", "Current", "5Y median", "5Y percentile", "Industry median", "vs industry"].map((label, i) => (
              <Table.Th key={label} ta={i === 0 ? "left" : "right"}>
                <Text size="xs" fw={700} c="dimmed" tt="uppercase" lts={0.4}>
                  {label}
                </Text>
              </Table.Th>
            ))}
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {relative.rows.map((row) => (
            <Table.Tr key={row.label}>
              <Table.Td>
                <Tooltip label={<Prose text={row.definition} />} multiline maw={340} withArrow position="top-start">
                  <Text size="sm" fw={600} style={{ cursor: "help" }}>
                    {row.label}
                  </Text>
                </Tooltip>
              </Table.Td>
              <Table.Td ta="right" className="num">
                {row.current === null ? (
                  <NotReported reason={row.current_reason} />
                ) : (
                  <Text size="sm" fw={700} inherit>
                    {value(row, row.current)}
                  </Text>
                )}
              </Table.Td>
              <Table.Td ta="right" className="num">
                {row.history_median === null ? (
                  <NotReported reason={row.history_reason} />
                ) : (
                  <>
                    <Text size="sm">{value(row, row.history_median)}</Text>
                    {/* A multiple is sampled every session and a rate once a quarter; the unit says which. */}
                    <Sub>
                      {row.history_observations.toLocaleString("en-US")} {row.history_unit}
                    </Sub>
                  </>
                )}
              </Table.Td>
              <Table.Td ta="right" className="num">
                {row.percentile === null ? (
                  <NotReported reason={row.history_reason || row.current_reason} />
                ) : (
                  <>
                    <Text size="sm">{fixed(row.percentile * 100, 0)}%</Text>
                    <Sub>of its 5Y was lower</Sub>
                  </>
                )}
              </Table.Td>
              <Table.Td ta="right" className="num">
                {row.industry_median === null ? (
                  <NotReported reason={row.industry_reason} />
                ) : (
                  <>
                    <Text size="sm">{value(row, row.industry_median)}</Text>
                    <Sub>n = {row.peer_count.toLocaleString("en-US")}</Sub>
                  </>
                )}
              </Table.Td>
              <Table.Td ta="right" className="num">
                {row.gap === null ? (
                  <NotReported reason={row.industry_reason || row.current_reason} />
                ) : (
                  <>
                    <Text size="sm" fw={600} c={toneColor(row.favourable)}>
                      {gapText(relative.kind, row.gap)}
                    </Text>
                    <Sub>{row.verdict}</Sub>
                  </>
                )}
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Table.ScrollContainer>
  );
}

function Legend({ relative }: { relative: Relative }) {
  const unavailable = relative.rows.filter((row) => row.current === null).length;
  return (
    <Group gap="xs" align="flex-start" wrap="nowrap">
      <Box c="dimmed" pt={2}>
        <IconInfoCircle size={16} stroke={1.6} />
      </Box>
      <Text size="xs" c="dimmed">
        {relative.kind === "valuation" ? (
          <>
            Medians, not means — a multiple whose denominator approaches zero prints in the thousands, and one such
            quarter moves an average but not a median. Non-positive readings are excluded everywhere: a negative P/E is
            a loss, not a cheap stock, and a zero yield is a non-payer. <b>5Y percentile</b> is the share of the last
            five years the company traded below where it does today. Green is cheaper than the industry, red dearer.
          </>
        ) : (
          <>
            Trailing twelve months against the same period a year earlier. <b>A falling figure is shown</b> — it is the
            answer, not a gap. What is not shown is a rate measured from a <b>negative or zero base</b>: a loss that
            narrowed from −4.75 to −2.12 would print −55% and read as a collapse. Capex growth is taken on the
            magnitude, so a rise means the company spent more — for capex and working capital, faster is a fact rather
            than a verdict. <b>vs industry</b> is a difference in percentage points, not a ratio.
          </>
        )}
        {unavailable > 0 &&
          ` ${unavailable} of ${relative.rows.length} metrics have no current reading; hover one for the reason, or open the panel below.`}
      </Text>
    </Group>
  );
}

function Sources({ relative }: { relative: Relative }) {
  const unavailable = relative.rows.filter(
    (row) => row.current === null || row.history_median === null || row.industry_median === null,
  ).length;
  return (
    <SourcesPanel
      title={`Sources & data status — ${plural(relative.rows.length, "metric")}${unavailable ? `, ${unavailable} with a figure unavailable` : ""}`}
    >
      <Table.ScrollContainer minWidth={820}>
        <Table fz="xs" verticalSpacing={6} highlightOnHover>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Metric</Table.Th>
              <Table.Th>Status</Table.Th>
              <Table.Th>Definition</Table.Th>
              <Table.Th>Notes</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {relative.rows.map((row) => (
              <Table.Tr key={row.label}>
                <Table.Td fw={600} style={{ whiteSpace: "nowrap" }}>
                  {row.label}
                </Table.Td>
                <Table.Td>
                  <OriginBadge origin={row.origin} />
                </Table.Td>
                <Table.Td>
                  <Prose text={row.definition} />
                </Table.Td>
                <Table.Td c="dimmed">
                  {[row.current_reason, row.history_reason, row.industry_reason].filter(Boolean).join(" · ") || "—"}
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
      <Text size="xs" c="dimmed" mt="sm">
        Peer group: Sharadar's <code>industry</code> classification{relative.industry && ` — ${relative.industry}`},
        priced {relative.priced ? isoDay(relative.priced) : "not loaded"}. The classification is the vendor's own, not
        SIC; the two disagree, and the DCF's financial-sector gate deliberately uses SIC instead.
      </Text>
      <OriginLegend />
    </SourcesPanel>
  );
}

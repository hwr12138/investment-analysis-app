// 🎯 Options — buying upside through calls, across ~120 names, from a stored night.
//
// The shell: the close the page is from (and a banner once that is more than a
// session behind), the theme and earnings filters, the names that are not on
// the page and why, and the four tabs. Only the open tab fetches.
import {
  Accordion,
  Alert,
  Anchor,
  Badge,
  Card,
  Center,
  Collapse,
  Group,
  Loader,
  MultiSelect,
  SegmentedControl,
  Stack,
  Tabs,
  Text,
  Title,
} from "@mantine/core";
import {
  IconAlertTriangle,
  IconCalculator,
  IconChartLine,
  IconClockExclamation,
  IconListDetails,
  IconTelescope,
} from "@tabler/icons-react";
import { useState } from "react";

import { isoDay } from "../../format";
import { DiscoverTab } from "./DiscoverTab";
import { EvaluateTab } from "./EvaluateTab";
import { HistoryTab } from "./HistoryTab";
import { ScreenTab } from "./ScreenTab";
import { EARNINGS_WINDOWS, type Overview, type Tab, useOptionsParams, useOverview } from "./common";

const TAB_LABELS: { value: Tab; label: string; icon: typeof IconTelescope }[] = [
  { value: "discover", label: "Discover", icon: IconTelescope },
  { value: "screen", label: "Screen", icon: IconListDetails },
  { value: "evaluate", label: "Evaluate", icon: IconCalculator },
  { value: "history", label: "History", icon: IconChartLine },
];

export function OptionsPage() {
  const { tab, filters, set } = useOptionsParams();
  const overview = useOverview(filters);
  const data = overview.data;

  return (
    <Stack gap="lg" maw={1440} mx="auto">
      <Group justify="space-between" align="flex-end" wrap="wrap" gap="md">
        <div>
          <Text size="xs" fw={700} c="dimmed" tt="uppercase" lts={0.6}>
            Analysis
          </Text>
          <Title order={1} fz={34} lh={1.15}>
            Options
          </Title>
          <Text size="sm" c="dimmed" maw={680}>
            Buying upside with calls: what it costs, how that compares with each name's own year, and which call
            spreads pay most if a move arrives. End-of-day ThetaData chains, collected nightly.
          </Text>
        </div>
        {data && <Freshness overview={data} />}
      </Group>

      {overview.isPending && (
        <Center mih="40vh">
          <Loader />
        </Center>
      )}
      {overview.isError && (
        <Alert variant="light" color="red" icon={<IconAlertTriangle size={18} />} title="Couldn't read the options store">
          {overview.error.message}
        </Alert>
      )}

      {data && (
        <>
          {data.sessions_behind > 1 && (
            <Alert
              variant="light"
              color="yellow"
              icon={<IconClockExclamation size={18} />}
              title={`The options store is ${data.sessions_behind} sessions behind`}
            >
              The newest stored close is {isoDay(data.session)}, and every figure on this page is from that day. The
              nightly collector has not completed since — check the Cloud Run job's logs.
            </Alert>
          )}

          <Filters
            overview={data}
            themes={filters.themes}
            earnings={filters.earnings}
            onThemes={(themes) => set({ themes: themes.length ? themes.join(",") : null })}
            onEarnings={(days) => set({ earnings: days === 21 ? null : String(days) })}
          />

          {data.visible.length === 0 ? (
            <Alert variant="light" color="gray">
              The filters leave no names on screen.
            </Alert>
          ) : (
            <Tabs value={tab} onChange={(value) => set({ tab: value === "discover" ? null : value })} keepMounted={false}>
              <Tabs.List mb="md">
                {TAB_LABELS.map(({ value, label, icon: Icon }) => (
                  <Tabs.Tab key={value} value={value} leftSection={<Icon size={16} stroke={1.6} />}>
                    {label}
                  </Tabs.Tab>
                ))}
              </Tabs.List>
              <Tabs.Panel value="discover">
                <DiscoverTab filters={filters} overview={data} />
              </Tabs.Panel>
              <Tabs.Panel value="screen">
                <ScreenTab filters={filters} themes={data.themes} />
              </Tabs.Panel>
              <Tabs.Panel value="evaluate">
                <EvaluateTab overview={data} />
              </Tabs.Panel>
              <Tabs.Panel value="history">
                <HistoryTab overview={data} />
              </Tabs.Panel>
            </Tabs>
          )}

          <Definitions overview={data} />
        </>
      )}
    </Stack>
  );
}

function Freshness({ overview }: { overview: Overview }) {
  const written = overview.written_at?.slice(0, 16).replace("T", " ");
  return (
    <Stack gap={4} align="flex-end">
      <Badge size="lg" variant="light" color={overview.sessions_behind > 1 ? "yellow" : "indigo"}>
        {isoDay(overview.session)} close
      </Badge>
      <Text size="xs" c="dimmed">
        {written ? `Collected ${written} UTC · ` : ""}
        {overview.universe.reduce((n, group) => n + group.symbols.length, 0)} names in the universe
      </Text>
    </Stack>
  );
}

function Filters({
  overview,
  themes,
  earnings,
  onThemes,
  onEarnings,
}: {
  overview: Overview;
  themes: string[] | null;
  earnings: number;
  onThemes: (themes: string[]) => void;
  onEarnings: (days: number) => void;
}) {
  const [showMissing, setShowMissing] = useState(false);
  const missing = overview.unavailable.length + overview.not_collected.length;
  return (
    <Card>
      <Group align="flex-end" gap="lg" wrap="wrap">
        <MultiSelect
          label="Themes"
          placeholder={themes ? undefined : "All themes"}
          data={overview.themes}
          value={themes ?? []}
          onChange={onThemes}
          searchable
          clearable
          hidePickedOptions
          maxDropdownHeight={320}
          style={{ flex: "1 1 360px" }}
        />
        <div>
          <Text size="sm" fw={500} mb={4}>
            Hide earnings within
          </Text>
          <SegmentedControl data={EARNINGS_WINDOWS} value={String(earnings)} onChange={(v) => onEarnings(Number(v))} />
        </div>
      </Group>
      <Group gap="lg" mt="sm" wrap="wrap">
        <Text size="sm" c="dimmed">
          <Text span fw={600} c="var(--mantine-color-text)">
            {overview.visible.length}
          </Text>{" "}
          of {overview.names.length} priced names on screen
        </Text>
        {overview.hidden_for_earnings.length > 0 && (
          <Text size="sm" c="dimmed">
            Hidden for earnings within {earnings} days: {overview.hidden_for_earnings.join(", ")}
          </Text>
        )}
        {!overview.earnings_known && earnings > 0 && (
          <Text size="sm" c="yellow">
            Earnings dates could not be read, so no name is hidden for them.
          </Text>
        )}
        {missing > 0 && (
          <Anchor component="button" size="sm" onClick={() => setShowMissing((open) => !open)}>
            {missing} {missing === 1 ? "name" : "names"} not on this page
          </Anchor>
        )}
      </Group>
      <Collapse expanded={showMissing}>
        <Stack gap={4} mt="sm">
          {overview.unavailable.map(({ symbol, reason }) => (
            <Text key={symbol} size="sm">
              <Text span fw={600}>
                {symbol}
              </Text>{" "}
              <Text span c="dimmed">
                — {reason}
              </Text>
            </Text>
          ))}
          {overview.not_collected.length > 0 && (
            <Text size="sm" c="dimmed">
              Not collected yet: {overview.not_collected.join(", ")}
            </Text>
          )}
        </Stack>
      </Collapse>
    </Card>
  );
}

function Definitions({ overview }: { overview: Overview }) {
  return (
    <Accordion variant="separated">
      <Accordion.Item value="definitions">
        <Accordion.Control>Universe and definitions</Accordion.Control>
        <Accordion.Panel>
          <Stack gap="sm">
            <Text size="sm">
              <b>Universe.</b> A curated list, grouped into themes as reading aids rather than as a classification.{" "}
              {overview.universe.map((group, i) => (
                <span key={group.theme}>
                  {i > 0 && " · "}
                  <i>{group.theme}</i>: {group.symbols.join(", ")}
                </span>
              ))}
            </Text>
            <Text size="sm">
              <b>Prices</b> are mids of ThetaData's 17:15 ET end-of-day NBBO; a zero bid is no market. <b>IVs</b> are
              Black-76 on each expiry's put–call-parity forward, so dividends need no estimate. <b>30-day</b> figures
              interpolate total variance between the expiries either side of 30 days; <b>25Δ</b> is read at constant
              delta on the call wing.
            </Text>
            <Text size="sm">
              <b>Buy-vol score</b> = {overview.score_scale} × the mean of −z(30d ATM IV), −z(30d IV − 21d realized) and
              −z(25Δ call skew), each against the name's own trailing year. Higher means calls are cheaper than usual
              for that name. A name needs {overview.min_history_sessions} stored sessions for a score. Our definition —
              the page this was modelled on did not publish one.
            </Text>
            <Text size="sm">
              <b>+k SD target</b> = spot × e<sup>k·σ·√t</sup> at the expiry's ATM IV. <b>Reward:risk</b> = the +1 SD
              spread's payout ÷ the payout a flat smile at ATM IV gives; 1.0× is fairly priced. On Screen it is the two
              quotes you would trade; on History it is read off the smile at a constant 30 days, so one stale quote or
              an expiry roll does not move it.
            </Text>
          </Stack>
        </Accordion.Panel>
      </Accordion.Item>
    </Accordion>
  );
}

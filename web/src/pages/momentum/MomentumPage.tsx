import {
  Alert,
  Badge,
  Box,
  Card,
  Divider,
  Grid,
  Group,
  LoadingOverlay,
  Progress,
  SegmentedControl,
  SimpleGrid,
  Skeleton,
  Stack,
  Text,
  TextInput,
  ThemeIcon,
  Title,
  useComputedColorScheme,
  useMantineTheme,
} from "@mantine/core";
import { DatePickerInput } from "@mantine/dates";
import {
  IconAlertTriangle,
  IconArrowDownRight,
  IconArrowUpRight,
  IconCalendar,
  IconMinus,
  IconSearch,
} from "@tabler/icons-react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useMemo, useState, type ReactNode } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router";

import { api, unwrap, type Schemas } from "../../api/client";
import { PlotlyChart } from "../../components/PlotlyChart";
import { fixed, isoDay, money, signed } from "../../format";
import { TONE_COLOR, type Tone } from "../../theme";
import { momentumLayout, momentumTraces, palette } from "./chart";
import { read } from "./signals";

type Period = Schemas["MomentumWindow"]["period"];
type AnchorKind = "Lows" | "Highs";
type Range = [string | null, string | null];

const PERIODS = ["5D", "1M", "3M", "6M", "1Y", "5Y"] as const satisfies readonly Period[];
const DEFAULT_PERIOD: Period = "1Y";
const CHART_HEIGHT = 620;

function asPeriod(value: string | null): Period {
  return (PERIODS as readonly string[]).includes(value ?? "") ? (value as Period) : DEFAULT_PERIOD;
}

export function MomentumPage() {
  // The ticker and period live in the URL, so a view is a link you can share.
  const { ticker = "AAPL" } = useParams();
  const [params, setParams] = useSearchParams();
  const period = asPeriod(params.get("period"));
  const navigate = useNavigate();

  const window = useQuery({
    queryKey: ["momentum", ticker, period],
    queryFn: () =>
      unwrap(api.GET("/api/momentum/{ticker}", { params: { path: { ticker }, query: { period } } })),
  });
  const sessions = window.data?.bars.sessions;

  return (
    <Stack gap="lg" maw={1440} mx="auto">
      <Group justify="space-between" align="flex-end" wrap="wrap" gap="md">
        <div>
          <Text size="xs" fw={700} c="dimmed" tt="uppercase" lts={0.6}>
            Momentum analysis
          </Text>
          <Title order={1} fz={34} lh={1.15}>
            {ticker}
          </Title>
          {sessions ? (
            <Text size="sm" c="dimmed">
              {sessions.length} sessions · {isoDay(sessions[0]!)} – {isoDay(sessions[sessions.length - 1]!)}
            </Text>
          ) : (
            window.isPending && <Skeleton h={14} w={220} mt={6} />
          )}
        </div>
        <Group gap="sm" wrap="wrap">
          <TickerSearch
            key={ticker}
            initial={ticker}
            onSubmit={(next) => navigate({ pathname: `/momentum/${next}`, search: params.toString() })}
          />
          <SegmentedControl
            data={[...PERIODS]}
            value={period}
            onChange={(value) => setParams({ period: value })}
          />
        </Group>
      </Group>

      {window.isPending && <LoadingBody />}
      {window.isError && (
        <Alert variant="light" color="red" icon={<IconAlertTriangle size={18} />} title={`Couldn't load ${ticker}`}>
          {window.error.message}
        </Alert>
      )}
      {window.data && <MomentumBody key={`${ticker}/${period}`} window={window.data} />}
    </Stack>
  );
}

function TickerSearch({ initial, onSubmit }: { initial: string; onSubmit: (ticker: string) => void }) {
  const [draft, setDraft] = useState(initial);
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        const next = draft.trim().toUpperCase();
        if (next) onSubmit(next);
      }}
    >
      <TextInput
        aria-label="Ticker"
        placeholder="Ticker"
        value={draft}
        onChange={(event) => setDraft(event.currentTarget.value)}
        leftSection={<IconSearch size={16} />}
        w={150}
        styles={{ input: { textTransform: "uppercase" } }}
      />
    </form>
  );
}

/** Everything drawn from one window. Keyed on ticker/period, so the anchors
 *  reset to the new window's defaults when either changes. */
function MomentumBody({ window }: { window: Schemas["MomentumWindow"] }) {
  const { ticker, period, bars } = window;
  const sessions = bars.sessions;
  const [range, setRange] = useState<Range>([window.default_anchors.first, window.default_anchors.second]);
  const [kind, setKind] = useState<AnchorKind>("Lows");
  const [first, second] = range;
  const complete = Boolean(first && second);

  const divergence = useQuery({
    queryKey: ["divergence", ticker, period, first, second, kind],
    queryFn: () =>
      unwrap(
        api.GET("/api/momentum/{ticker}/divergence", {
          params: { path: { ticker }, query: { period, first: first!, second: second!, anchor: kind } },
        }),
      ),
    enabled: complete,
    placeholderData: keepPreviousData,
  });
  // A placeholder is the previous anchors' answer. It stays visible under an
  // overlay, but is never drawn on the chart as if it matched the inputs.
  const current = divergence.isPlaceholderData ? null : (divergence.data ?? null);

  const theme = useMantineTheme();
  const dark = useComputedColorScheme("dark") === "dark";
  const colors = useMemo(() => palette(theme, dark), [theme, dark]);
  const data = useMemo(() => momentumTraces(window, current, colors), [window, current, colors]);
  const layout = useMemo(() => momentumLayout(colors), [colors]);

  return (
    <Stack gap="lg">
      <LatestReadings bars={bars} />
      <Grid gap="lg">
        <Grid.Col span={{ base: 12, lg: 8 }}>
          <Card padding="md">
            <PlotlyChart data={data} layout={layout} height={CHART_HEIGHT} />
          </Card>
        </Grid.Col>
        <Grid.Col span={{ base: 12, lg: 4 }}>
          <Stack gap="lg">
            <SignalsCard scanners={window.scanners} />
            <Card>
              <Text fw={600}>Divergence</Text>
              <Text size="xs" c="dimmed" mb="md">
                Join two swing points and compare the line with MACD and RSI over the same span.
              </Text>
              <Stack gap="sm">
                <DatePickerInput
                  type="range"
                  label="Anchors"
                  value={range}
                  onChange={setRange}
                  minDate={sessions[0]}
                  maxDate={sessions[sessions.length - 1]}
                  valueFormat="MMM D, YYYY"
                  leftSection={<IconCalendar size={16} />}
                />
                <SegmentedControl
                  fullWidth
                  value={kind}
                  onChange={(value) => setKind(value as AnchorKind)}
                  data={[
                    { label: "Swing lows", value: "Lows" },
                    { label: "Swing highs", value: "Highs" },
                  ]}
                />
              </Stack>
              <Divider my="md" />
              {!complete ? (
                <Text size="sm" c="dimmed">
                  Pick the second anchor to compare.
                </Text>
              ) : divergence.isError ? (
                <Alert variant="light" color="yellow" icon={<IconAlertTriangle size={18} />}>
                  {divergence.error.message}
                </Alert>
              ) : (
                <Box pos="relative" mih={88}>
                  <LoadingOverlay
                    visible={divergence.isPlaceholderData || !divergence.data}
                    overlayProps={{ blur: 1 }}
                    loaderProps={{ size: "sm" }}
                  />
                  {divergence.data && (
                    <Stack gap="md">
                      <DivergenceReading label="MACD" reading={divergence.data.macd} />
                      <DivergenceReading label="RSI" reading={divergence.data.rsi} />
                    </Stack>
                  )}
                </Box>
              )}
            </Card>
          </Stack>
        </Grid.Col>
      </Grid>
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// The latest session, at a glance
// ---------------------------------------------------------------------------

function LatestReadings({ bars }: { bars: Schemas["Bars"] }) {
  const last = bars.sessions.length - 1;
  const close = bars.close[last]!;
  const rsi = bars.rsi[last]!;
  const macd = bars.macd[last]!;
  const signal = bars.signal[last]!;
  const histogram = bars.histogram[last]!;
  return (
    <SimpleGrid cols={{ base: 2, md: 4 }} spacing="lg">
      <Stat label="Last close" value={money(close)} hint={isoDay(bars.sessions[last]!)} />
      <Stat
        label="RSI 14"
        value={fixed(rsi, 1)}
        hint={<Progress value={rsi} size="sm" mt={6} color={rsi >= 70 ? "red" : rsi <= 30 ? "teal" : "violet"} />}
      />
      <Stat label="MACD" value={fixed(macd)} hint={`Signal ${fixed(signal)}`} />
      <Stat
        label="Histogram"
        value={signed(histogram)}
        color={histogram >= 0 ? "teal" : "red"}
        hint="MACD less signal"
      />
    </SimpleGrid>
  );
}

function Stat({ label, value, hint, color }: { label: string; value: string; hint: ReactNode; color?: string }) {
  return (
    <Card padding="md">
      <Text size="xs" fw={700} c="dimmed" tt="uppercase" lts={0.4}>
        {label}
      </Text>
      <Text fz={26} fw={700} lh={1.3} className="num" c={color}>
        {value}
      </Text>
      {typeof hint === "string" ? (
        <Text size="xs" c="dimmed">
          {hint}
        </Text>
      ) : (
        hint
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Signals and divergence readings
// ---------------------------------------------------------------------------

const TONE_ICON: Record<Tone, typeof IconMinus> = {
  bullish: IconArrowUpRight,
  bearish: IconArrowDownRight,
  neutral: IconMinus,
};

function SignalsCard({ scanners }: { scanners: Schemas["Scanners"] }) {
  const rows: [string, string][] = [
    ["MACD · momentum", scanners.macd_momentum],
    ["MACD · zero line", scanners.macd_trend],
    ["RSI · trend", scanners.rsi_trend],
    ["RSI · failure swing", scanners.rsi_failure_swings],
  ];
  return (
    <Card>
      <Group justify="space-between" mb="md">
        <Text fw={600}>Signals</Text>
        <Text size="xs" c="dimmed">
          at the last close
        </Text>
      </Group>
      <Stack gap="md">
        {rows.map(([label, text]) => {
          const signal = read(text);
          const Icon = TONE_ICON[signal.tone];
          return (
            <Group key={label} wrap="nowrap" align="flex-start" gap="sm">
              <ThemeIcon
                size={34}
                radius="md"
                color={TONE_COLOR[signal.tone]}
                variant={signal.fresh ? "filled" : "light"}
              >
                <Icon size={18} stroke={1.8} />
              </ThemeIcon>
              <Box miw={0}>
                <Text size="xs" c="dimmed" fw={600}>
                  {label}
                </Text>
                <Text size="sm" fw={500}>
                  {signal.headline}
                </Text>
                {signal.detail && (
                  <Text size="xs" c="dimmed">
                    {signal.detail}
                  </Text>
                )}
              </Box>
            </Group>
          );
        })}
      </Stack>
    </Card>
  );
}

function DivergenceReading({ label, reading }: { label: string; reading: Schemas["Reading"] }) {
  const tone = read(reading.state).tone;
  return (
    <div>
      <Group justify="space-between" wrap="nowrap" gap="xs">
        <Text size="sm" fw={600}>
          {label}
        </Text>
        <Badge variant="light" color={TONE_COLOR[tone]} radius="sm">
          {reading.state}
        </Badge>
      </Group>
      <Text size="xs" c="dimmed" mt={2}>
        {reading.bias}
      </Text>
    </div>
  );
}

// ---------------------------------------------------------------------------
// While the window loads
// ---------------------------------------------------------------------------

function LoadingBody() {
  return (
    <Stack gap="lg">
      <SimpleGrid cols={{ base: 2, md: 4 }} spacing="lg">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} h={96} radius="md" />
        ))}
      </SimpleGrid>
      <Grid gap="lg">
        <Grid.Col span={{ base: 12, lg: 8 }}>
          <Skeleton h={CHART_HEIGHT + 32} radius="md" />
        </Grid.Col>
        <Grid.Col span={{ base: 12, lg: 4 }}>
          <Stack gap="lg">
            <Skeleton h={300} radius="md" />
            <Skeleton h={280} radius="md" />
          </Stack>
        </Grid.Col>
      </Grid>
    </Stack>
  );
}

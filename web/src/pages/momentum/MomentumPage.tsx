import {
  Alert,
  Button,
  Group,
  Loader,
  SegmentedControl,
  SimpleGrid,
  Stack,
  Text,
  TextInput,
  Title,
  useComputedColorScheme,
} from "@mantine/core";
import { DatePickerInput } from "@mantine/dates";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router";

import { api, unwrap, type Schemas } from "../../api/client";
import { PlotlyChart } from "../../components/PlotlyChart";
import { momentumLayout, momentumTraces } from "./chart";

type Period = Schemas["MomentumWindow"]["period"];
type AnchorKind = "Lows" | "Highs";

const PERIODS = ["5D", "1M", "3M", "6M", "1Y", "5Y"] as const satisfies readonly Period[];
const DEFAULT_PERIOD: Period = "1Y";

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

  return (
    <Stack gap="lg" maw={1400}>
      <Title order={2}>Momentum Analysis</Title>
      <Group align="flex-end" wrap="wrap">
        <TickerForm
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

      {window.isPending && <Loader />}
      {window.isError && (
        <Alert color="red" title="Could not load this ticker">
          {window.error.message}
        </Alert>
      )}
      {window.data && <MomentumBody key={`${ticker}/${period}`} window={window.data} />}
    </Stack>
  );
}

function TickerForm({ initial, onSubmit }: { initial: string; onSubmit: (ticker: string) => void }) {
  const [draft, setDraft] = useState(initial);
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        const next = draft.trim().toUpperCase();
        if (next) onSubmit(next);
      }}
    >
      <Group align="flex-end" gap="xs">
        <TextInput
          label="Stock ticker"
          value={draft}
          onChange={(event) => setDraft(event.currentTarget.value)}
          w={160}
        />
        <Button type="submit" variant="default">
          Load
        </Button>
      </Group>
    </form>
  );
}

/** Everything drawn from one window. Keyed on ticker/period, so the anchors
 *  reset to the new window's defaults the way the Streamlit widgets did. */
function MomentumBody({ window }: { window: Schemas["MomentumWindow"] }) {
  const { ticker, period, bars, scanners } = window;
  const sessions = bars.sessions;
  const [first, setFirst] = useState<string | null>(window.default_anchors.first);
  const [second, setSecond] = useState<string | null>(window.default_anchors.second);
  const [kind, setKind] = useState<AnchorKind>("Lows");

  const divergence = useQuery({
    queryKey: ["divergence", ticker, period, first, second, kind],
    queryFn: () =>
      unwrap(
        api.GET("/api/momentum/{ticker}/divergence", {
          params: { path: { ticker }, query: { period, first: first!, second: second!, anchor: kind } },
        }),
      ),
    enabled: Boolean(first && second),
    placeholderData: keepPreviousData,
  });
  // A placeholder is the previous anchors' answer. It may be shown — dimmed and
  // labelled — but never drawn on the chart as if it matched the inputs.
  const current = divergence.isPlaceholderData ? null : (divergence.data ?? null);

  const dark = useComputedColorScheme("dark") === "dark";
  const data = useMemo(() => momentumTraces(window, current), [window, current]);
  const layout = useMemo(() => momentumLayout(dark), [dark]);

  return (
    <Stack gap="lg">
      <Text c="dimmed" size="sm">
        {sessions.length} sessions, {sessions[0]} to {sessions[sessions.length - 1]} (last close shown)
      </Text>

      <Title order={3}>Live indicator scanners</Title>
      <SimpleGrid cols={{ base: 1, sm: 2 }}>
        <Stack gap="xs">
          <Title order={4}>MACD</Title>
          <Alert title="Immediate momentum">{scanners.macd_momentum}</Alert>
          <Alert title="Macro trend (zero line)">{scanners.macd_trend}</Alert>
        </Stack>
        <Stack gap="xs">
          <Title order={4}>RSI</Title>
          <Alert title="Macro trend">{scanners.rsi_trend}</Alert>
          <Alert title="Failure swings (last 25 periods)">{scanners.rsi_failure_swings}</Alert>
        </Stack>
      </SimpleGrid>

      <Title order={3}>Manual divergence analysis</Title>
      <Group align="flex-end" wrap="wrap">
        <DatePickerInput
          label="Anchor date 1"
          value={first}
          onChange={setFirst}
          minDate={sessions[0]}
          maxDate={sessions[sessions.length - 1]}
          w={180}
        />
        <DatePickerInput
          label="Anchor date 2"
          value={second}
          onChange={setSecond}
          minDate={sessions[0]}
          maxDate={sessions[sessions.length - 1]}
          w={180}
        />
        <Stack gap={4}>
          <Text size="sm" fw={500}>
            What are these points?
          </Text>
          <SegmentedControl
            data={["Lows", "Highs"]}
            value={kind}
            onChange={(value) => setKind(value as AnchorKind)}
          />
        </Stack>
      </Group>
      <Text c="dimmed" size="xs">
        Connecting swing lows looks for bullish setups; connecting highs looks for bearish ones.
      </Text>

      {divergence.isError && <Alert color="yellow">{divergence.error.message}</Alert>}
      {divergence.data && (
        <SimpleGrid
          cols={{ base: 1, sm: 2 }}
          style={{ opacity: divergence.isPlaceholderData ? 0.45 : 1, transition: "opacity 120ms" }}
          aria-busy={divergence.isPlaceholderData}
        >
          <Alert color="green" title={`MACD: ${divergence.data.macd.state}`}>
            Bias: {divergence.data.macd.bias}
          </Alert>
          <Alert color="green" title={`RSI: ${divergence.data.rsi.state}`}>
            Bias: {divergence.data.rsi.bias}
          </Alert>
        </SimpleGrid>
      )}
      {divergence.isPlaceholderData && (
        <Text c="dimmed" size="xs">
          Updating for the new anchors…
        </Text>
      )}

      <PlotlyChart data={data} layout={layout} height={800} />
    </Stack>
  );
}

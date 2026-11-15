// 📋 Screen — the call spreads that pay in full at a +k SD move.
//
// Ranked by the payout at the ask, with the mid beside it. On a wide market the
// two differ by a multiple, and a screen ranked on mids rewards exactly the
// names a buyer cannot trade at the mid.
import {
  Alert,
  Card,
  Center,
  Group,
  Loader,
  LoadingOverlay,
  SegmentedControl,
  Select,
  Stack,
  Table,
  Text,
  Title,
  useMantineTheme,
} from "@mantine/core";
import { IconAlertTriangle } from "@tabler/icons-react";
import { useMemo } from "react";

import type { Schemas } from "../../api/client";
import { fixed, isoDay, percent } from "../../format";
import { daysTo, type Filters, strike, themeColors, times, useOptionsParams, useScreen } from "./common";
import { NameTd, NumTd, SortTh, useSorted } from "./table";

type Row = Schemas["Spread"];
type SpreadKey = Exclude<keyof Row, "theme" | "short_strike">;

const spreadValue = (row: Row, key: SpreadKey) => row[key];

export function ScreenTab({ filters, themes }: { filters: Filters; themes: string[] }) {
  const { params, set } = useOptionsParams();
  const sd = Number(params.get("sd") ?? 1);
  const screen = useScreen(filters, sd, params.get("expiry"));
  const theme = useMantineTheme();
  const colors = useMemo(() => themeColors(theme, themes), [theme, themes]);

  if (screen.isPending)
    return (
      <Center mih={300}>
        <Loader />
      </Center>
    );
  if (screen.isError)
    return (
      <Alert variant="light" color="red" icon={<IconAlertTriangle size={18} />} title="Couldn't load Screen">
        {screen.error.message}
      </Alert>
    );

  const data = screen.data;
  const moveLabel = `+${data.sd} SD`;
  return (
    <Stack gap="lg" pos="relative">
      <LoadingOverlay visible={screen.isPlaceholderData} zIndex={2} overlayProps={{ blur: 1 }} />
      <Group gap="lg" align="flex-end" wrap="wrap">
        <div>
          <Text size="sm" fw={500} mb={4}>
            Move
          </Text>
          <SegmentedControl
            data={data.sds.map((value) => ({ value: String(value), label: `+${value} SD` }))}
            value={String(data.sd)}
            onChange={(value) => set({ sd: value === "1" ? null : value })}
          />
        </div>
        {data.expirations.length > 0 && (
          <Select
            label="Expiry"
            data={data.expirations.map((e) => ({ value: e, label: `${isoDay(e)} · ${daysTo(data.session, e)}d` }))}
            value={data.expiration}
            onChange={(value) => set({ expiry: value })}
            allowDeselect={false}
            w={240}
          />
        )}
      </Group>

      {data.expiration === null ? (
        <Alert variant="light" color="gray">
          No expiry is listed for enough of the names on screen.
        </Alert>
      ) : (
        <Card>
          <Title order={4}>
            Call spreads paying in full at {moveLabel} · {isoDay(data.expiration)}
          </Title>
          <Text size="sm" c="dimmed" mb="sm">
            The one-strike spread whose short strike is the highest tightly quoted one at or below the target, so the
            whole width is earned if the move arrives.
          </Text>
          {data.rows.length === 0 ? (
            <Text size="sm" c="dimmed">
              No name has a tightly quoted spread at that target.
            </Text>
          ) : (
            <SpreadTable rows={data.rows} colors={colors} />
          )}
          <Text size="xs" c="dimmed" mt="sm">
            {data.rows.length} of {data.visible_count} names list a tightly quoted spread at this target. Legs with a
            bid/ask wider than half their mid are skipped, not priced.
          </Text>
        </Card>
      )}
    </Stack>
  );
}

function SpreadTable({ rows, colors }: { rows: Row[]; colors: Map<string, string> }) {
  const { sorted, sort, toggle } = useSorted<Row, SpreadKey>(rows, { key: "payout_ask", descending: true }, spreadValue);
  const th = (label: string, column: SpreadKey, help?: string) => (
    <SortTh label={label} column={column} sort={sort} onSort={toggle} help={help} />
  );
  return (
    <Table.ScrollContainer minWidth={980}>
      <Table striped highlightOnHover verticalSpacing={6} horizontalSpacing="sm">
        <Table.Thead>
          <Table.Tr>
            <SortTh label="Name" column="symbol" sort={sort} onSort={toggle} align="left" />
            {th("Move to target", "move", "spot × e^(k·σ·√t) at the expiry's ATM IV")}
            {th("Spread", "long_strike")}
            {th("Price (mid)", "price")}
            {th("Width", "width")}
            {th("Bid", "bid", "Floored at zero")}
            {th("Ask", "ask")}
            {th("Payout at mid", "payout_mid")}
            {th("Payout at ask", "payout_ask", "What a buyer paying the ask earns if the target is reached")}
            {th("Reward:risk", "reward_risk", "Mid payout ÷ the payout at a flat smile; 1.0× is fair")}
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {sorted.map((row) => (
            <Table.Tr key={row.symbol}>
              <NameTd symbol={row.symbol} theme={row.theme} color={colors.get(row.theme)} />
              <NumTd value={row.move} format={(v) => `+${percent(v, 1)}`} />
              <Table.Td ta="right" className="num" style={{ whiteSpace: "nowrap" }}>
                {strike(row.long_strike)}/{strike(row.short_strike)}C
              </Table.Td>
              <NumTd value={row.price} format={(v) => fixed(v)} />
              <NumTd value={row.width} format={(v) => fixed(v)} />
              <NumTd value={row.bid} format={(v) => fixed(v)} />
              <NumTd value={row.ask} format={(v) => fixed(v)} />
              <NumTd value={row.payout_mid} format={(v) => times(v)} />
              <NumTd value={row.payout_ask} format={(v) => times(v)} />
              <NumTd
                value={row.reward_risk}
                format={(v) => times(v, 2)}
                color={row.reward_risk !== null && row.reward_risk > 1 ? "teal" : undefined}
              />
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Table.ScrollContainer>
  );
}

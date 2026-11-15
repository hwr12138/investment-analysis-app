// 🧮 Evaluate — what replacing stock with calls costs, by strike and expiry, for one name.
//
// Each cell is the call's mid as a share of spot (in Spread mode, the call
// spread up to ten points of spot higher), shaded by the annualised cost so a
// two-year LEAP and a one-month call read on one scale. A strike outside the
// quoted range is a dash, never an extrapolation.
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
} from "@mantine/core";
import { IconAlertTriangle } from "@tabler/icons-react";

import type { Schemas } from "../../api/client";
import { fixed, isoDay, MISSING, percent } from "../../format";
import { NameSelect } from "./NameSelect";
import { chosenName, type Overview, useEvaluate, useOptionsParams } from "./common";

type Mode = "single" | "spread";

export function EvaluateTab({ overview }: { overview: Overview }) {
  const { params, set } = useOptionsParams();
  const name = chosenName(params, overview.visible) ?? "GLD";
  const mode: Mode = params.get("structure") === "spread" ? "spread" : "single";
  const evaluate = useEvaluate(name);

  return (
    <Stack gap="lg" pos="relative">
      <Group gap="lg" align="flex-end" wrap="wrap">
        <NameSelect overview={overview} value={name} onChange={(symbol) => set({ name: symbol })} />
        <div>
          <Text size="sm" fw={500} mb={4}>
            Structure
          </Text>
          <SegmentedControl
            data={[
              { value: "single", label: "Single call" },
              { value: "spread", label: "Call spread" },
            ]}
            value={mode}
            onChange={(value) => set({ structure: value === "single" ? null : value })}
          />
        </div>
      </Group>

      {evaluate.isPending && (
        <Center mih={260}>
          <Loader />
        </Center>
      )}
      {evaluate.isError && (
        <Alert variant="light" color="red" icon={<IconAlertTriangle size={18} />} title={`Couldn't evaluate ${name}`}>
          {evaluate.error.message}
        </Alert>
      )}
      {evaluate.data && (
        <Box pos="relative">
          <LoadingOverlay visible={evaluate.isPlaceholderData} zIndex={2} overlayProps={{ blur: 1 }} />
          <GridCard data={evaluate.data} mode={mode} />
        </Box>
      )}
    </Stack>
  );
}

function GridCard({ data, mode }: { data: Schemas["OptionsEvaluate"]; mode: Mode }) {
  const rows = data[mode];
  // Shade by annualised cost across the whole grid: cheapest per year darkest.
  const annualised = rows.flatMap((r) => r.cells.map((c) => c.annualised)).filter((v): v is number => v !== null);
  const low = Math.min(...annualised);
  const high = Math.max(...annualised);
  const strength = (value: number) => (high > low ? 12 + Math.round(((high - value) / (high - low)) * 58) : 40);

  return (
    <Card>
      <Title order={4}>
        Replacement cost — {data.symbol} at {fixed(data.spot)}
      </Title>
      <Text size="sm" c="dimmed" mb="sm">
        {mode === "spread"
          ? `The call spread from the row's strike to ${percent(data.spread_width, 0)} of spot higher`
          : "The call at spot × (1 + row)"}
        , at its mid, as a share of spot, on the {isoDay(data.session)} close. Darker is cheaper per year of time bought.
      </Text>
      <Table.ScrollContainer minWidth={140 + data.columns.length * 130}>
        <Table verticalSpacing={8} horizontalSpacing="xs" style={{ borderCollapse: "separate", borderSpacing: 4 }}>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Strike vs spot</Table.Th>
              {data.columns.map((column) => (
                <Table.Th key={column.expiration} ta="center">
                  <Text size="sm" fw={700}>
                    {column.label}
                  </Text>
                  <Text size="xs" c="dimmed" fw={400}>
                    {isoDay(column.expiration)}
                  </Text>
                </Table.Th>
              ))}
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {rows.map((row) => (
              <Table.Tr key={row.upside}>
                <Table.Th className="num">+{percent(row.upside, 0)}</Table.Th>
                {row.cells.map((cell, j) => (
                  <Table.Td
                    key={data.columns[j]!.expiration}
                    ta="center"
                    className="num"
                    style={{
                      borderRadius: 6,
                      background:
                        cell.annualised === null
                          ? "var(--mantine-color-default-hover)"
                          : `color-mix(in srgb, var(--mantine-color-teal-6) ${strength(cell.annualised)}%, transparent)`,
                    }}
                  >
                    {cell.cost_pct === null ? (
                      <Text span c="dimmed" inherit>
                        {MISSING}
                      </Text>
                    ) : (
                      <Tooltip
                        label={`${percent(cell.annualised ?? 0, 1)} a year${cell.wide ? " · a bracketing quote is wide" : ""}`}
                        withArrow
                      >
                        <span>
                          {percent(cell.cost_pct)}
                          {cell.wide && (
                            <Text span c="yellow" inherit ml={2}>
                              °
                            </Text>
                          )}
                        </span>
                      </Tooltip>
                    )}
                  </Table.Td>
                ))}
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
      <Text size="xs" c="dimmed" mt="sm">
        ° marks a bracketing quote that is one-sided or wider than half its mid; a dash, a strike outside the quoted
        range. Hover a cell for its annualised cost.
      </Text>
    </Card>
  );
}

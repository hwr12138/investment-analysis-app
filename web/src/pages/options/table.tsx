// Click-to-sort column headers for the Options tables. A missing figure sorts
// last whichever way the column runs, so a gap never tops a ranking.
import { Group, Table, Text, Tooltip, UnstyledButton } from "@mantine/core";
import { IconChevronDown, IconChevronUp, IconSelector } from "@tabler/icons-react";
import { useMemo, useState, type ReactNode } from "react";

import { MISSING } from "../../format";

export type Sort<K extends string> = { key: K; descending: boolean };

export function useSorted<T, K extends string>(
  rows: T[],
  initial: Sort<K>,
  value: (row: T, key: K) => number | string | null | undefined,
) {
  const [sort, setSort] = useState(initial);
  const sorted = useMemo(() => {
    const out = [...rows];
    out.sort((a, b) => {
      const x = value(a, sort.key);
      const y = value(b, sort.key);
      if (x == null || y == null) return x == null ? (y == null ? 0 : 1) : -1;
      const order = x < y ? -1 : x > y ? 1 : 0;
      return sort.descending ? -order : order;
    });
    return out;
  }, [rows, sort, value]);
  const toggle = (key: K) =>
    setSort((current) => (current.key === key ? { key, descending: !current.descending } : { key, descending: true }));
  return { sorted, sort, toggle };
}

export function SortTh<K extends string>({
  label,
  column,
  sort,
  onSort,
  help,
  align = "right",
}: {
  label: string;
  column: K;
  sort: Sort<K>;
  onSort: (key: K) => void;
  help?: string;
  align?: "left" | "right";
}) {
  const active = sort.key === column;
  const Icon = active ? (sort.descending ? IconChevronDown : IconChevronUp) : IconSelector;
  const button = (
    <UnstyledButton onClick={() => onSort(column)} style={{ width: "100%" }}>
      <Group gap={4} wrap="nowrap" justify={align === "right" ? "flex-end" : "flex-start"}>
        <Text size="sm" fw={600} c={active ? undefined : "dimmed"} style={{ whiteSpace: "nowrap" }}>
          {label}
        </Text>
        <Icon size={14} stroke={1.6} opacity={active ? 1 : 0.5} />
      </Group>
    </UnstyledButton>
  );
  return (
    <Table.Th ta={align}>
      {help ? (
        <Tooltip label={help} multiline maw={260} withArrow>
          {button}
        </Tooltip>
      ) : (
        button
      )}
    </Table.Th>
  );
}

/** A number cell: the formatted figure, or a dimmed dash where it is missing. */
export function NumTd({ value, format, color }: { value: number | null | undefined; format: (v: number) => ReactNode; color?: string }) {
  return (
    <Table.Td ta="right" className="num" c={value == null ? undefined : color}>
      {value == null ? (
        <Text span c="dimmed" inherit>
          {MISSING}
        </Text>
      ) : (
        format(value)
      )}
    </Table.Td>
  );
}

/** The name, with its theme under it in the theme's colour. */
export function NameTd({ symbol, theme, color }: { symbol: string; theme: string; color?: string }) {
  return (
    <Table.Td>
      <Text size="sm" fw={600} lh={1.2}>
        {symbol}
      </Text>
      <Group gap={5} wrap="nowrap">
        <span style={{ width: 7, height: 7, borderRadius: 2, background: color ?? "var(--mantine-color-gray-5)" }} />
        <Text size="xs" c="dimmed" style={{ whiteSpace: "nowrap" }}>
          {theme}
        </Text>
      </Group>
    </Table.Td>
  );
}

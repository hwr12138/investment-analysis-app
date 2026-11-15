// The one-name picker Evaluate and History share. The choice is one URL
// parameter, so a name picked on one tab is the name the other opens on.
import { Select } from "@mantine/core";
import { useMemo } from "react";

import type { Overview } from "./common";

export function NameSelect({
  overview,
  value,
  onChange,
}: {
  overview: Overview;
  value: string;
  onChange: (symbol: string) => void;
}) {
  // Grouped by theme, in the order the universe lists them.
  const data = useMemo(() => {
    const groups = new Map<string, string[]>();
    for (const { symbol, theme } of overview.names) {
      if (!overview.visible.includes(symbol)) continue;
      groups.set(theme, [...(groups.get(theme) ?? []), symbol]);
    }
    return [...groups].map(([group, items]) => ({ group, items }));
  }, [overview.names, overview.visible]);
  return (
    <Select
      label="Name"
      data={data}
      value={value}
      onChange={(next) => next && onChange(next)}
      searchable
      allowDeselect={false}
      maxDropdownHeight={360}
      w={180}
    />
  );
}

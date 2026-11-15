// A headline figure in a card: a small label, the number, and a line under it.
import { Card, Text } from "@mantine/core";
import type { ReactNode } from "react";

type Props = { label: string; value: string; hint?: ReactNode; color?: string };

export function Stat({ label, value, hint, color }: Props) {
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

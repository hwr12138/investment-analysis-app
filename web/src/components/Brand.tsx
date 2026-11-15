import { Group, Text, ThemeIcon } from "@mantine/core";
import { IconChartCandle } from "@tabler/icons-react";

export function Brand({ size = "md" }: { size?: "md" | "lg" }) {
  const large = size === "lg";
  return (
    <Group gap="xs" wrap="nowrap">
      <ThemeIcon
        size={large ? 44 : 32}
        radius="md"
        variant="gradient"
        gradient={{ from: "indigo", to: "teal", deg: 135 }}
      >
        <IconChartCandle size={large ? 26 : 20} stroke={1.8} />
      </ThemeIcon>
      <Text fw={800} fz={large ? 24 : 19} lts={-0.3}>
        Equiti
      </Text>
    </Group>
  );
}

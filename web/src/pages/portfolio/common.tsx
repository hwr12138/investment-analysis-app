// What the two portfolio pages share: the risk-free field, the date defaults,
// the colour each asset is drawn in, and the page's opening lines.
import { Loader, NumberInput, Stack, Text, Title } from "@mantine/core";
import { useQuery } from "@tanstack/react-query";
import dayjs from "dayjs";
import type { ReactNode } from "react";

import { api, unwrap, type Schemas } from "../../api/client";
import { isoDay } from "../../format";

export const DEFAULT_START = "2020-01-01";
export const today = () => dayjs().format("YYYY-MM-DD");

/** Comma-joined lists keep the page's URL short and readable. */
export const joined = (values: (string | number)[]) => values.join(",");
export const split = (value: string | null) => (value ? value.split(",").filter(Boolean) : []);

/** A number from the URL, or null when absent or not a number. */
export function numberParam(value: string | null): number | null {
  if (value === null || value.trim() === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

// Assets keep their colour across every chart and bar on a page, by position.
const ASSET_COLORS = ["indigo", "teal", "orange", "grape", "cyan", "yellow", "pink", "lime", "blue", "red"];

/** The asset's hue, for components that pick their own shade (a light Badge). */
export function assetHue(index: number, ticker?: string): string {
  return ticker === "CASH" ? "gray" : ASSET_COLORS[index % ASSET_COLORS.length]!;
}

/** The asset's solid colour, for charts, bars and swatches. */
export function assetColor(index: number, ticker?: string): string {
  return ticker === "CASH" ? "gray.5" : `${assetHue(index)}.6`;
}

/** "indigo.6" as a CSS colour, for elements outside Mantine's colour props. */
export const cssColor = (color: string) => `var(--mantine-color-${color.replace(".", "-")})`;

export function PageIntro({ title, description }: { title: string; description: ReactNode }) {
  return (
    <div>
      <Text size="xs" fw={700} c="dimmed" tt="uppercase" lts={0.6}>
        Portfolio
      </Text>
      <Title order={1} fz={34} lh={1.15}>
        {title}
      </Title>
      <Text size="sm" c="dimmed" maw={680}>
        {description}
      </Text>
    </div>
  );
}

// ---------------------------------------------------------------------------
// The risk-free rate
// ---------------------------------------------------------------------------

export function useRiskFreeRate() {
  return useQuery({
    queryKey: ["risk-free"],
    queryFn: () => unwrap(api.GET("/api/rates/risk-free")),
    staleTime: 60 * 60_000,
  });
}

/** The fetched rate in percent to two places, or null when there is none. */
export function defaultRiskFreePct(rate: Schemas["RiskFreeRate"] | undefined): number | null {
  return rate?.value == null ? null : Math.round(rate.value * 10_000) / 100;
}

/** The rate in percent. Pre-filled from the T-bill yield; when that could not
 *  be fetched the field starts empty and says why — nothing stands in for it. */
export function RiskFreeInput({ value, onChange }: { value: number | ""; onChange: (value: number | "") => void }) {
  const rate = useRiskFreeRate();
  const failed = rate.isError || (rate.data !== undefined && rate.data.value === null);
  const reason = rate.isError ? rate.error.message : rate.data?.detail;
  return (
    <NumberInput
      label="Risk-free rate"
      suffix="%"
      decimalScale={2}
      step={0.1}
      min={-5}
      max={50}
      value={value}
      onChange={(next) => onChange(typeof next === "number" ? next : "")}
      rightSection={rate.isPending ? <Loader size="xs" /> : undefined}
      description={
        rate.data?.value != null
          ? `13-week T-bill yield (^IRX)${rate.data.as_of ? `, ${isoDay(rate.data.as_of)}` : ""}`
          : undefined
      }
      error={failed && value === "" ? `Couldn't fetch the T-bill yield (${reason}). Enter a rate.` : undefined}
    />
  );
}

export function Notices({ children }: { children: ReactNode }) {
  return <Stack gap="xs">{children}</Stack>;
}

// The neutral colours every Plotly chart draws its frame in — grid, text, the
// zero line — from the Mantine theme, so a chart follows light and dark mode.
import type { MantineTheme } from "@mantine/core";

export type ChartPalette = { grid: string; text: string; muted: string; zero: string; font: string };

export function chartPalette(theme: MantineTheme, dark: boolean): ChartPalette {
  const c = theme.colors;
  return {
    grid: dark ? c.dark[5] : c.gray[2],
    text: dark ? c.dark[1] : c.gray[7],
    muted: dark ? c.dark[2] : c.gray[6],
    zero: dark ? c.dark[3] : c.gray[5],
    font: theme.fontFamily ?? "sans-serif",
  };
}

// What every Options tab shares: the filters as they sit in the URL, the
// queries they key, the colour each theme is drawn in, and the chart palette.
import type { MantineTheme } from "@mantine/core";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import dayjs from "dayjs";
import { useSearchParams } from "react-router";

import { api, unwrap, type Schemas } from "../../api/client";
import { fixed } from "../../format";

export type Overview = Schemas["OptionsOverview"];

/** The earnings filter's windows, in calendar days; 0 hides nothing. The API's
 *  default is 21, and the URL leaves it out when it is the default. */
export const EARNINGS_WINDOWS = [
  { value: "0", label: "Show all" },
  { value: "7", label: "1 week" },
  { value: "14", label: "2 weeks" },
  { value: "21", label: "3 weeks" },
];
const DEFAULT_EARNINGS = 21;

export const TABS = ["discover", "screen", "evaluate", "history"] as const;
export type Tab = (typeof TABS)[number];

/** The night only changes once a day; a tab revisited within ten minutes is not refetched. */
const STALE_MS = 10 * 60_000;

export type Filters = { themes: string[] | null; earnings: number };

/** The page's state in its URL, so a view is a link: tab, filters, and each tab's choice. */
export function useOptionsParams() {
  const [params, setParams] = useSearchParams();
  const themes = params.get("themes");
  const earnings = Number(params.get("earnings") ?? DEFAULT_EARNINGS);
  const tab = (TABS as readonly string[]).includes(params.get("tab") ?? "") ? (params.get("tab") as Tab) : "discover";
  const filters: Filters = {
    themes: themes ? themes.split(",").filter(Boolean) : null,
    earnings: [0, 7, 14, 21].includes(earnings) ? earnings : DEFAULT_EARNINGS,
  };
  const set = (changes: Record<string, string | null>) =>
    setParams(
      (current) => {
        const next = new URLSearchParams(current);
        for (const [key, value] of Object.entries(changes)) {
          if (value === null || value === "") next.delete(key);
          else next.set(key, value);
        }
        return next;
      },
      { replace: true },
    );
  return { params, tab, filters, set };
}

/** The filters as the API takes them: themes omitted for every theme. */
export function filterQuery(filters: Filters) {
  return {
    ...(filters.themes ? { themes: filters.themes } : {}),
    earnings_window: filters.earnings,
  };
}

export function useOverview(filters: Filters) {
  return useQuery({
    queryKey: ["options", "overview", filters],
    queryFn: () => unwrap(api.GET("/api/options/overview", { params: { query: filterQuery(filters) } })),
    placeholderData: keepPreviousData,
    staleTime: STALE_MS,
  });
}

export function useDiscover(filters: Filters) {
  return useQuery({
    queryKey: ["options", "discover", filters],
    queryFn: () => unwrap(api.GET("/api/options/discover", { params: { query: filterQuery(filters) } })),
    placeholderData: keepPreviousData,
    staleTime: STALE_MS,
  });
}

export function useScreen(filters: Filters, sd: number, expiration: string | null) {
  return useQuery({
    queryKey: ["options", "screen", filters, sd, expiration],
    queryFn: () =>
      unwrap(
        api.GET("/api/options/screen", {
          params: { query: { ...filterQuery(filters), sd, ...(expiration ? { expiration } : {}) } },
        }),
      ),
    placeholderData: keepPreviousData,
    staleTime: STALE_MS,
  });
}

export function useEvaluate(symbol: string) {
  return useQuery({
    queryKey: ["options", "evaluate", symbol],
    queryFn: () => unwrap(api.GET("/api/options/evaluate/{symbol}", { params: { path: { symbol } } })),
    placeholderData: keepPreviousData,
    staleTime: STALE_MS,
  });
}

export function useHistory(symbol: string) {
  return useQuery({
    queryKey: ["options", "history", symbol],
    queryFn: () => unwrap(api.GET("/api/options/history/{symbol}", { params: { path: { symbol } } })),
    placeholderData: keepPreviousData,
    staleTime: STALE_MS,
  });
}

/** The one name the Evaluate and History tabs draw: the URL's when it is on
 *  screen, else GLD, else the first visible name. */
export function chosenName(params: URLSearchParams, visible: string[]): string | null {
  const asked = params.get("name");
  if (asked && visible.includes(asked)) return asked;
  return visible.includes("GLD") ? "GLD" : (visible[0] ?? null);
}

// ---------------------------------------------------------------------------
// Colours
// ---------------------------------------------------------------------------

const HUES = ["indigo", "teal", "orange", "grape", "cyan", "yellow", "pink", "lime", "blue", "red", "green", "violet"];
const SHADES = [6, 9, 4];

/** Each theme's colour, fixed by its place among every priced theme so it
 *  keeps its colour when the filter hides others. */
export function themeColors(theme: MantineTheme, themes: string[]): Map<string, string> {
  return new Map(
    [...themes].sort().map((name, i) => {
      const hue = HUES[i % HUES.length]!;
      const shade = SHADES[Math.floor(i / HUES.length) % SHADES.length]!;
      return [name, theme.colors[hue]![shade]!];
    }),
  );
}

export { chartPalette, type ChartPalette } from "../../components/chartPalette";

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/** A multiple: 1.94 → "1.9×". */
export const times = (value: number, digits = 1) => `${fixed(value, digits)}×`;

/** Calendar days from the session to an expiry, for its label. */
export const daysTo = (session: string, expiration: string) => dayjs(expiration).diff(dayjs(session), "day");

/** A strike as quoted: 398 → "398", 7952.5 → "7,952.5". */
export const strike = (value: number) => value.toLocaleString("en-US", { maximumFractionDigits: 2 });

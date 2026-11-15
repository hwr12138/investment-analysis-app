// What every Stock Research tab shares: the URL state, one hook per route, and
// money in the company's own currency.
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useParams, useSearchParams } from "react-router";

import { api, unwrap, type Schemas } from "../../api/client";
import { fixed } from "../../format";

export type Profile = Schemas["Profile"];
export type Period = Schemas["Summary"]["period"];
export type StatementKey = Schemas["Financials"]["statement"];
export type UnitName = Schemas["Financials"]["unit"];

export const TABS = ["summary", "valuation", "insiders", "holdings", "financials"] as const;
export type Tab = (typeof TABS)[number];

// A research page is a snapshot: the API caches the fetches for the day, so a
// tab revisited within the session is served from here without asking again.
const STALE = 30 * 60 * 1000;

/** The ticker from the path and every choice from the query string, so a view is a link. */
export function useResearchParams() {
  const { ticker = "AAPL" } = useParams();
  const [params, setParams] = useSearchParams();
  const tab = (TABS as readonly string[]).includes(params.get("tab") ?? "") ? (params.get("tab") as Tab) : "summary";
  /** Sets or clears query parameters, leaving the rest alone. */
  const set = (changes: Record<string, string | null>) =>
    setParams(
      (current) => {
        const next = new URLSearchParams(current);
        for (const [key, value] of Object.entries(changes)) {
          if (value === null) next.delete(key);
          else next.set(key, value);
        }
        return next;
      },
      { replace: true },
    );
  return { ticker: ticker.toUpperCase(), params, tab, set };
}

export function useProfile(ticker: string) {
  return useQuery({
    queryKey: ["research", ticker, "profile"],
    queryFn: () => unwrap(api.GET("/api/research/{ticker}", { params: { path: { ticker } } })),
    staleTime: STALE,
    // A ticker Sharadar does not carry will not appear on a retry.
    retry: (count, error) => !("status" in error && error.status === 404) && count < 2,
  });
}

// The shell mounts a tab only once the profile is in, so a cold search
// assembles the snapshot once rather than once for the header and again for
// the open tab.

export function useSummary(ticker: string, period: Period) {
  return useQuery({
    queryKey: ["research", ticker, "summary", period],
    queryFn: () =>
      unwrap(api.GET("/api/research/{ticker}/summary", { params: { path: { ticker }, query: { period } } })),
    staleTime: STALE,
    placeholderData: keepPreviousData,
  });
}

export function useDcfLine(ticker: string, period: Period) {
  return useQuery({
    queryKey: ["research", ticker, "dcf-line", period],
    queryFn: () =>
      unwrap(api.GET("/api/research/{ticker}/dcf-line", { params: { path: { ticker }, query: { period } } })),
    staleTime: STALE,
  });
}

export function useFinancials(
  ticker: string,
  query: { statement: StatementKey; period: "annual" | "quarterly"; unit: UnitName; show_empty: boolean },
) {
  return useQuery({
    queryKey: ["research", ticker, "financials", query],
    queryFn: () => unwrap(api.GET("/api/research/{ticker}/financials", { params: { path: { ticker }, query } })),
    staleTime: STALE,
    placeholderData: keepPreviousData,
  });
}

export function useRelative(ticker: string, kind: "valuation" | "growth") {
  return useQuery({
    queryKey: ["research", ticker, kind],
    queryFn: () =>
      kind === "valuation"
        ? unwrap(api.GET("/api/research/{ticker}/valuation", { params: { path: { ticker } } }))
        : unwrap(api.GET("/api/research/{ticker}/growth", { params: { path: { ticker } } })),
    staleTime: STALE,
    placeholderData: keepPreviousData,
  });
}

export function useInsiders(ticker: string) {
  return useQuery({
    queryKey: ["research", ticker, "insiders"],
    queryFn: () => unwrap(api.GET("/api/research/{ticker}/insiders", { params: { path: { ticker } } })),
    staleTime: STALE,
  });
}

export function useHoldings(ticker: string) {
  return useQuery({
    queryKey: ["research", ticker, "holdings"],
    queryFn: () => unwrap(api.GET("/api/research/{ticker}/holdings", { params: { path: { ticker } } })),
    staleTime: STALE,
  });
}

// ---------------------------------------------------------------------------
// Formatting in the company's currency
// ---------------------------------------------------------------------------

function currencyFormat(currency: string, options: Intl.NumberFormatOptions) {
  try {
    return new Intl.NumberFormat("en-US", { style: "currency", currency: currency || "USD", ...options });
  } catch {
    // A code the browser does not know: plain digits, with the code beside them.
    return null;
  }
}

/** Intl's compact suffix in the house style: K, M, B, T → k, m, bn, tn. Only the
 *  suffix part is touched, so a currency code such as MXN is left alone. */
function compact(format: Intl.NumberFormat, value: number): string {
  return format
    .formatToParts(value)
    .map((part) => (part.type === "compact" ? ({ B: "bn", T: "tn" }[part.value] ?? part.value.toLowerCase()) : part.value))
    .join("");
}

/** $1.23bn, $45.6m, ¥820k — compact, in the reporting currency. */
export function compactMoney(value: number, currency: string): string {
  const format = currencyFormat(currency, { notation: "compact", maximumFractionDigits: 2 });
  if (format) return compact(format, value);
  return `${compactCount(value)} ${currency}`;
}

/** A price: 182.41 → "$182.41". */
export function price(value: number, currency: string): string {
  const format = currencyFormat(currency, { maximumFractionDigits: 2, minimumFractionDigits: 2 });
  return format ? format.format(value) : `${fixed(value)} ${currency}`;
}

/** 1,234,567 → "1.23m"; small counts stay whole. */
export function compactCount(value: number): string {
  if (Math.abs(value) < 10_000) return Math.round(value).toLocaleString("en-US");
  return compact(new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 }), value);
}

/** A signed change: 0.052 → "+5.2%". */
export const signedPercent = (value: number, digits = 1) =>
  `${value > 0 ? "+" : value < 0 ? "−" : ""}${fixed(Math.abs(value) * 100, digits)}%`;

/** Mantine colour for a reading that may be good, bad or neither. */
export const toneColor = (favourable: boolean | null | undefined) =>
  favourable == null ? "dimmed" : favourable ? "teal" : "red";

/** A signed amount: +$880m, −$5.99bn. */
export const signedMoney = (value: number, currency: string) =>
  `${value > 0 ? "+" : value < 0 ? "−" : ""}${compactMoney(Math.abs(value), currency)}`;

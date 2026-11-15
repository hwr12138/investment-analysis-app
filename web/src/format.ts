// Display formatting only — no figure is derived here.
const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
const day = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
  timeZone: "UTC",
});

export const money = (value: number) => usd.format(value);
/** Typographic minus (−), which lines up with "+" in tabular figures; "-0.00" reads as "0.00". */
export const fixed = (value: number, digits = 2) => {
  const text = Math.abs(value).toFixed(digits);
  return value < 0 && Number(text) !== 0 ? `−${text}` : text;
};
export const signed = (value: number, digits = 2) => {
  const text = fixed(value, digits);
  return value > 0 && Number(text) !== 0 ? `+${text}` : text;
};
/** "2026-11-13" → "Nov 13, 2026", read as a calendar date with no time zone shift. */
export const isoDay = (iso: string) => day.format(new Date(`${iso}T00:00:00Z`));
/** A decimal as a percentage: 0.1234 → "12.34%". */
export const percent = (value: number, digits = 2) => `${fixed(value * 100, digits)}%`;
/** Shown where a figure is missing on purpose, never a zero. */
export const MISSING = "—";

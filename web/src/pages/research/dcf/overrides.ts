// The reader's overrides: only what they changed. An input left alone is no
// override at all, so it follows its default when the default moves — an
// untyped cost of debt follows the risk-free rate, a driver follows its basis.
import type { DcfOverrides } from "../common";

export type Overrides = DcfOverrides;

type MapKey = "bases" | "values" | "target_on" | "target_values";
type ScalarKey = Exclude<keyof Overrides, MapKey>;

export const NO_OVERRIDES: Overrides = { bases: {}, values: {}, target_on: {}, target_values: {} };

/** Sets one input, or drops it back to its default with null. */
export function setScalar<K extends ScalarKey>(o: Overrides, key: K, value: Overrides[K] | null): Overrides {
  const next = { ...o };
  if (value === null || value === undefined) delete next[key];
  else next[key] = value;
  return next;
}

/** Sets one entry of a per-driver map, or drops it with null. */
export function setEntry<M extends MapKey>(o: Overrides, map: M, key: string, value: Overrides[M][string] | null): Overrides {
  const entries: Record<string, unknown> = { ...o[map] };
  if (value === null) delete entries[key];
  else entries[key] = value;
  return { ...o, [map]: entries };
}

export const isUntouched = (o: Overrides) =>
  Object.keys(o).every((key) => ["bases", "values", "target_on", "target_values"].includes(key)) &&
  [o.bases, o.values, o.target_on, o.target_values].every((map) => Object.keys(map).length === 0);

export type Update = (change: (o: Overrides) => Overrides) => void;

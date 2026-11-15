// What more than one part of the 💰 DCF tab draws: the logic layer's prose,
// a figure's provenance, and the inputs that send an override up.
import { ActionIcon, Alert, Code, Group, NumberInput, Text, Tooltip, type NumberInputProps } from "@mantine/core";
import { useDebouncedCallback } from "@mantine/hooks";
import { IconAlertTriangle, IconArrowBackUp, IconInfoCircle } from "@tabler/icons-react";
import { Fragment, useEffect, useRef, useState, type ReactNode } from "react";

import type { Schemas } from "../../../api/client";
import { isoDay } from "../../../format";
import { compactMoney } from "../common";
import { OriginBadge } from "../parts";

export type Dcf = Schemas["Dcf"];
export type Panel = Schemas["Panel"];
export type Valuation = Schemas["Valuation"];
export type Company = Schemas["Company"];
export type Note = Schemas["Note"];
export type Field = Schemas["Input"];
export type Provenance = Schemas["Provenance"];

/** Compact money with a typographic minus; a negative zero reads as zero. */
export function money(value: number, currency: string): string {
  const text = compactMoney(Math.abs(value), currency);
  return value < 0 && /[1-9]/.test(text) ? `−${text}` : text;
}

const INLINE = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)/g;

/** The logic layer's notes are written in Streamlit's inline markdown: bold,
 *  italics and code, nothing else. */
export function Md({ text }: { text: string }) {
  return (
    <>
      {text.split(INLINE).map((part, i) => {
        if (part.startsWith("**") && part.endsWith("**") && part.length > 4) return <b key={i}>{part.slice(2, -2)}</b>;
        if (part.startsWith("`") && part.endsWith("`") && part.length > 2)
          return (
            <Code key={i} fz="inherit">
              {part.slice(1, -1)}
            </Code>
          );
        if (part.startsWith("*") && part.endsWith("*") && part.length > 2) return <i key={i}>{part.slice(1, -1)}</i>;
        return <Fragment key={i}>{part}</Fragment>;
      })}
    </>
  );
}

/** A caption reads as small print; an info or a warning is a callout. */
export function NoteView({ note }: { note: Note }) {
  if (note.level === "caption") return <Caption>{note.text}</Caption>;
  const warning = note.level === "warning";
  return (
    <Alert
      variant="light"
      color={warning ? "yellow" : "blue"}
      p="xs"
      icon={warning ? <IconAlertTriangle size={16} /> : <IconInfoCircle size={16} />}
    >
      <Text size="xs">
        <Md text={note.text} />
      </Text>
    </Alert>
  );
}

export function Notes({ notes }: { notes: Note[] }) {
  return (
    <>
      {notes.map((note, i) => (
        <NoteView key={i} note={note} />
      ))}
    </>
  );
}

/** Small print under a block; a string is read as the logic layer's markdown. */
export function Caption({ children }: { children: ReactNode }) {
  return (
    <Text size="xs" c="dimmed">
      {typeof children === "string" ? <Md text={children} /> : children}
    </Text>
  );
}

export function describe(p: Provenance): string {
  return [p.source, p.formula, p.as_of && `as of ${isoDay(p.as_of)}`, p.detail].filter(Boolean).join(" · ");
}

/** Where a figure came from: its origin badge, then the source and formula. */
export function ProvenanceLine({ provenance }: { provenance: Provenance }) {
  return (
    <Group gap={6} wrap="nowrap" align="flex-start">
      <OriginBadge origin={provenance.origin} />
      <Text size="xs" c="dimmed" style={{ minWidth: 0 }}>
        {describe(provenance)}
      </Text>
    </Group>
  );
}

/** A rate typed as a percentage, within the range the API accepts (−100% to 500%). */
export const RATE = { suffix: "%", scale: 100, min: -100, max: 500, clampBehavior: "strict" } as const;

const round = (value: number, places: number) => Number(value.toFixed(places));

type NumberFieldProps = Omit<NumberInputProps, "value" | "onChange" | "defaultValue"> & {
  /** The figure in model units — a fraction for a rate, raw currency for money. */
  value: number | null;
  /** What the field returns to when cleared or reset; null where there is none. */
  fallback?: number | null;
  /** Shown units per model unit: 100 for a percentage, 1e-6 for millions. */
  scale?: number;
  places?: number;
  overridden?: boolean;
  /** A number in model units, or null to drop the override. */
  onCommit: (value: number | null) => void;
};

/**
 * A number the reader types, sent up once they pause or leave the field.
 *
 * The field follows the server's figure while it is not being edited — an
 * input nobody typed follows its default, and the default can move — and a
 * value typed back to the default, at the precision shown, drops the override
 * rather than pinning it.
 */
export function NumberField({
  value,
  fallback = null,
  scale = 1,
  places = 2,
  overridden = false,
  onCommit,
  rightSection,
  ...rest
}: NumberFieldProps) {
  const shown = value === null ? "" : round(value * scale, places);
  const [draft, setDraft] = useState<number | string>(shown);
  const editing = useRef(false);

  // Follow the server's figure, and catch up when the override is dropped —
  // a reset, or an undo after a refused input, where that figure never moved.
  const wasOverridden = useRef(overridden);
  useEffect(() => {
    if (!editing.current) setDraft(shown);
  }, [shown]);
  useEffect(() => {
    if (wasOverridden.current && !overridden && !editing.current) setDraft(shown);
    wasOverridden.current = overridden;
  }, [overridden, shown]);

  const commit = useDebouncedCallback((next: number | string) => {
    if (next === "" || typeof next !== "number" || !Number.isFinite(next)) {
      if (next === "") onCommit(null);
      return;
    }
    const isDefault = fallback !== null && Math.abs(next - fallback * scale) < 0.5 * 10 ** -places;
    onCommit(isDefault ? null : next / scale);
  }, 700);

  const reset = overridden && (
    <Tooltip label="Back to the default" withArrow>
      <ActionIcon
        variant="subtle"
        color="gray"
        size="sm"
        aria-label="Back to the default"
        onClick={() => {
          commit.cancel();
          onCommit(null);
        }}
      >
        <IconArrowBackUp size={15} />
      </ActionIcon>
    </Tooltip>
  );

  return (
    <NumberInput
      {...rest}
      value={draft}
      decimalScale={places}
      rightSection={reset || rightSection}
      onChange={(next) => {
        setDraft(next);
        commit(next);
      }}
      onFocus={() => {
        editing.current = true;
      }}
      onBlur={() => {
        editing.current = false;
        commit.flush();
        if (draft === "") setDraft(shown);
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter") commit.flush();
      }}
    />
  );
}

/** A labelled figure in a block of the panel: small caps label, the number, a line under it. */
export function Figure({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div>
      <Text size="xs" fw={700} c="dimmed" tt="uppercase" lts={0.4}>
        {label}
      </Text>
      <Text fz="lg" fw={700} className="num">
        {value}
      </Text>
      {hint && (
        <Text size="xs" c="dimmed">
          {hint}
        </Text>
      )}
    </div>
  );
}

// Pieces more than one research tab draws the same way: where a figure came
// from, what a missing one looks like, and the collapsed panel that holds both.
import { Accordion, Badge, Box, Code, Table, Text, Tooltip } from "@mantine/core";
import type { ReactNode } from "react";

import type { Schemas } from "../../api/client";
import { isoDay } from "../../format";

type Origin = Schemas["Provenance"]["origin"];

const ORIGIN: Record<Origin, { label: string; color: string }> = {
  reported: { label: "Reported", color: "blue" },
  derived: { label: "Derived", color: "violet" },
  estimate: { label: "Estimate", color: "yellow" },
  user: { label: "User", color: "grape" },
  unavailable: { label: "Unavailable", color: "red" },
};

export function OriginBadge({ origin }: { origin: Origin }) {
  const { label, color } = ORIGIN[origin];
  return (
    // A narrow table column would otherwise clip the label to "Repo…".
    <Badge size="xs" variant="light" color={color} radius="sm" style={{ minWidth: "max-content" }}>
      {label}
    </Badge>
  );
}

/** A figure that could not be obtained. Never a blank, never a zero; the reason
 *  is a hover away where there is one. */
export function NotReported({ reason, size = "sm" }: { reason?: string; size?: "xs" | "sm" }) {
  const text = (
    <Text span size={size} c="dimmed" fs="italic" style={reason ? { cursor: "help" } : undefined}>
      not reported
    </Text>
  );
  return reason ? (
    <Tooltip label={reason} multiline maw={320} withArrow>
      {text}
    </Tooltip>
  ) : (
    text
  );
}

/** Closes every sources panel: what each origin means. */
export function OriginLegend({ estimate = false }: { estimate?: boolean }) {
  return (
    <Text size="xs" c="dimmed" mt="sm">
      <b>Reported</b> is read straight off a Sharadar line item · <b>Derived</b> is arithmetic over reported figures
      {estimate && (
        <>
          {" "}
          · <b>Estimate</b> is an approximation and always labelled
        </>
      )}{" "}
      · <b>Unavailable</b> could not be obtained and is never substituted.
    </Text>
  );
}

/** A collapsed panel for provenance: reachable, never in the way. */
export function SourcesPanel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <Accordion variant="separated">
      <Accordion.Item value="sources">
        <Accordion.Control>
          <Text size="sm" fw={500}>
            {title}
          </Text>
        </Accordion.Control>
        <Accordion.Panel>{children}</Accordion.Panel>
      </Accordion.Item>
    </Accordion>
  );
}

const ISO_DAY = /^\d{4}-\d{2}-\d{2}$/;

function frameCell(value: string | number | null): ReactNode {
  if (value === null) return <Text span c="dimmed" inherit>—</Text>;
  if (typeof value === "number") {
    // Large figures are dollars or share counts, where cents are noise; small ones keep two places.
    const digits = Number.isInteger(value) || Math.abs(value) >= 10_000 ? 0 : 2;
    const text = Math.abs(value).toLocaleString("en-US", { maximumFractionDigits: digits });
    return value < 0 && text !== "0" ? `−${text}` : text;
  }
  if (ISO_DAY.test(value)) return isoDay(value);
  return value;
}

/** A small table printed as the logic layer laid it out. */
export function FrameTable({ frame, minWidth = 640 }: { frame: Schemas["Frame"]; minWidth?: number }) {
  if (frame.rows.length === 0) return null;
  return (
    <Table.ScrollContainer minWidth={minWidth}>
      <Table striped highlightOnHover fz="xs" verticalSpacing={6}>
        <Table.Thead>
          <Table.Tr>
            {frame.columns.map((column) => (
              <Table.Th key={column} style={{ whiteSpace: "nowrap" }}>
                {column}
              </Table.Th>
            ))}
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {frame.rows.map((row, i) => (
            <Table.Tr key={i}>
              {row.map((value, j) => (
                <Table.Td key={j} ta={typeof value === "number" ? "right" : undefined} className={typeof value === "number" ? "num" : undefined}>
                  {frameCell(value)}
                </Table.Td>
              ))}
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Table.ScrollContainer>
  );
}

/** The logic layer's prose, with its `backticked` column names set as code. */
export function Prose({ text }: { text: string }) {
  return (
    <>
      {text.split("`").map((part, i) =>
        i % 2 ? (
          <Code key={i} fz="inherit">
            {part}
          </Code>
        ) : (
          part
        ),
      )}
    </>
  );
}

/** "3 of 16 figures unavailable" style counts in a panel title. */
export const plural = (count: number, one: string, many = `${one}s`) => `${count.toLocaleString("en-US")} ${count === 1 ? one : many}`;

/** A segmented control that scrolls sideways when its row is wider than the screen,
 *  rather than spilling out of its card. */
export function Segments({ children }: { children: ReactNode }) {
  return <Box style={{ maxWidth: "100%", overflowX: "auto", scrollbarWidth: "none" }}>{children}</Box>;
}

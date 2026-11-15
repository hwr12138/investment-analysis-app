// 📄 Financials — the four statements, sectioned and reconciled.
//
// The only statement viewer in the app. The layout, the formatting of every
// cell and the reconciliation are all `logic.statements`; this draws the rows
// it is handed. Subtotals are shown as the company reported them, never as the
// sum of the rows above, and a disagreement marks the row rather than fixing it.
import "./financials.css";

import {
  Alert,
  Badge,
  Box,
  Card,
  Checkbox,
  Group,
  LoadingOverlay,
  SegmentedControl,
  Skeleton,
  Stack,
  Table,
  Text,
  Title,
  Tooltip,
} from "@mantine/core";
import { IconAlertTriangle, IconInfoCircle } from "@tabler/icons-react";
import type { ReactNode } from "react";

import type { Schemas } from "../../api/client";
import { fixed, MISSING } from "../../format";
import { type StatementKey, type UnitName, useFinancials, useResearchParams } from "./common";
import { FrameTable, plural, Segments, SourcesPanel } from "./parts";

type Financials = Schemas["Financials"];
type Row = Schemas["StatementRow"];

const STATEMENT_KEYS = ["PL", "BS", "CF", "DERIVED"] as const satisfies readonly StatementKey[];
const UNIT_NAMES = ["As reported", "Millions", "Billions"] as const satisfies readonly UnitName[];

const ZERO_FILLED =
  "Sharadar writes 0 into this indicator both for a genuine nil and for a line item absent from the filing. The two cannot be told apart.";

export function FinancialsTab({ ticker }: { ticker: string }) {
  const { params, set } = useResearchParams();
  const statement = pick(params.get("statement"), STATEMENT_KEYS, "PL");
  const period = params.get("fperiod") === "quarterly" ? "quarterly" : "annual";
  const unit = pick(params.get("unit"), UNIT_NAMES, "Millions");
  const showEmpty = params.get("empty") === "1";
  const financials = useFinancials(ticker, { statement, period, unit, show_empty: showEmpty });
  const data = financials.data;

  return (
    <Stack gap="lg">
      {financials.isPending && <Skeleton h={560} radius="md" />}
      {financials.isError && (
        <Alert variant="light" color="red" icon={<IconAlertTriangle size={18} />} title="Couldn't load the statements">
          {financials.error.message}
        </Alert>
      )}
      {data && !data.available && (
        <Alert variant="light" color="blue" icon={<IconInfoCircle size={18} />}>
          <b>Sharadar returned no fundamentals for {data.ticker}.</b> Nothing has been substituted. Prices and the
          summary statistics are unaffected.
        </Alert>
      )}
      {data && data.available && (
        <>
          <Card>
            <Group gap="lg" align="flex-end" wrap="wrap" mb="md">
              <Control label="Statement">
                <Segments>
                  <SegmentedControl
                    data={data.statements.map((choice) => ({ value: choice.key, label: choice.label }))}
                    value={statement}
                    onChange={(value) => set({ statement: value === "PL" ? null : value })}
                  />
                </Segments>
              </Control>
              <Control label="Period">
                <SegmentedControl
                  data={[
                    { value: "annual", label: "Annual" },
                    { value: "quarterly", label: "Quarterly" },
                  ]}
                  value={period}
                  onChange={(value) => set({ fperiod: value === "annual" ? null : value })}
                />
              </Control>
              <Control
                label="Units"
                help="Display only. Per-share figures, share counts, margins and multiples are never scaled — only money is."
              >
                <Segments>
                  <SegmentedControl
                    data={data.units}
                    value={unit}
                    onChange={(value) => set({ unit: value === "Millions" ? null : value })}
                  />
                </Segments>
              </Control>
              <Checkbox
                label={`Show the line items Sharadar left empty for ${data.ticker}`}
                checked={showEmpty}
                onChange={(event) => set({ empty: event.currentTarget.checked ? "1" : null })}
                mb={8}
              />
            </Group>
            {data.sections.length === 0 ? (
              <Alert variant="light" color="blue" icon={<IconInfoCircle size={18} />}>
                Sharadar returned no {data.label.toLowerCase()} for {data.ticker} at the {data.period} dimension.
                Nothing has been substituted — try the other period.
              </Alert>
            ) : (
              <Box pos="relative">
                <LoadingOverlay visible={financials.isPlaceholderData} zIndex={3} overlayProps={{ blur: 1 }} />
                <StatementTable data={data} />
              </Box>
            )}
          </Card>
          {data.sections.length > 0 && <Caption data={data} />}
          <Sources data={data} />
        </>
      )}
    </Stack>
  );
}

function pick<T extends string>(value: string | null, options: readonly T[], fallback: T): T {
  return (options as readonly string[]).includes(value ?? "") ? (value as T) : fallback;
}

function Control({ label, help, children }: { label: string; help?: string; children: ReactNode }) {
  const heading = (
    <Text size="sm" fw={500} mb={4} style={help ? { cursor: "help" } : undefined}>
      {label}
    </Text>
  );
  return (
    <div style={{ minWidth: 0, maxWidth: "100%" }}>
      {help ? (
        <Tooltip label={help} multiline maw={300} withArrow>
          {heading}
        </Tooltip>
      ) : (
        heading
      )}
      {children}
    </div>
  );
}

function StatementTable({ data }: { data: Financials }) {
  const lead = data.periods[0]?.is_latest ?? false;
  const width = data.periods.length;
  return (
    <div className="fin-scroll">
      <table className="fin">
        <thead>
          <tr>
            <th className="fin-label">
              {data.label}
              {data.unit_suffix}
            </th>
            {data.periods.map((column, i) => (
              <th key={column.key} className={lead && i === 0 ? "fin-lead" : undefined}>
                {column.label}
                <span className="fin-sub">{column.sublabel}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.sections.map((section) => [
            <tr key={`section-${section.title}`} className="fin-section">
              <th className="fin-label">
                {section.note ? (
                  <Tooltip label={section.note} multiline maw={320} withArrow position="top-start">
                    <span style={{ cursor: "help" }}>{section.title}</span>
                  </Tooltip>
                ) : (
                  section.title
                )}
              </th>
              <td colSpan={width} />
            </tr>,
            ...section.rows.map((row) => <StatementLine key={`${section.title}-${row.key}`} row={row} lead={lead} />),
          ])}
        </tbody>
      </table>
    </div>
  );
}

function StatementLine({ row, lead }: { row: Row; lead: boolean }) {
  const classes = ["fin-row", `fin-${row.emphasis}`, `fin-i${row.indent}`, row.flagged ? "fin-flagged" : ""];
  return (
    <tr className={classes.join(" ")}>
      <th className="fin-label">
        <Group gap={6} wrap="nowrap">
          <span>{row.label}</span>
          {row.origin === "derived" ? (
            <Tag tip={row.formula || row.note || "Computed here, not a filed line item."}>derived</Tag>
          ) : (
            row.note && <Tag tip={row.note}>note</Tag>
          )}
          {row.flagged && (
            <Tooltip label={row.flag_detail} multiline maw={340} withArrow>
              <Text span c="orange" fw={700} style={{ cursor: "help" }}>
                ≠
              </Text>
            </Tooltip>
          )}
        </Group>
      </th>
      {row.cells.map((cell, i) => {
        const className = [lead && i === 0 ? "fin-lead" : "", cell.missing ? "fin-missing" : ""].join(" ").trim();
        // Native titles, not Tooltips: a wide statement is thousands of cells.
        if (cell.missing)
          return (
            <td key={i} className={className} title={cell.reason || undefined}>
              {cell.text}
            </td>
          );
        return (
          <td key={i} className={className || undefined}>
            {cell.zero_filled ? (
              <span className="fin-zerofill" title={ZERO_FILLED}>
                {cell.text}
              </span>
            ) : (
              cell.text
            )}
          </td>
        );
      })}
    </tr>
  );
}

function Tag({ tip, children }: { tip: string; children: ReactNode }) {
  return (
    <Tooltip label={tip} multiline maw={340} withArrow>
      <Badge size="xs" variant="outline" color="gray" radius="sm" style={{ cursor: "help", minWidth: "max-content" }}>
        {children}
      </Badge>
    </Tooltip>
  );
}

function Caption({ data }: { data: Financials }) {
  const total = data.reported_rows + data.hidden_rows;
  return (
    <Text size="xs" c="dimmed">
      <b>{data.label}</b> · {data.dimension} · {data.reported_rows} of {total} line items reported
      {data.hidden_rows > 0 && ` · ${data.hidden_rows} left empty by Sharadar and hidden`} · figures in {data.unit_label}{" "}
      of {data.currency || "the reporting currency"}. A <b>{MISSING}</b> is a figure Sharadar carries no value for — hover
      it for why. A <b>0</b> with a dotted underline is an indicator Sharadar zero-fills, where 0 may mean
      absent-from-the-filing rather than nil. Rows marked <b>derived</b> are computed here — hover for the formula — and
      a <b>≠</b> means the reported subtotal and the lines above it disagree.
    </Text>
  );
}

const STATUS_COLOR: Record<string, string> = { ok: "teal", gap: "red", "not evaluated": "gray" };

function Sources({ data }: { data: Financials }) {
  const failing = data.checks.filter((check) => check.status === "gap").length;
  return (
    <SourcesPanel
      title={`Reconciliation and sources — ${plural(data.checks.length, "check")}${failing ? `, ${failing} failing` : ""}`}
    >
      <Stack gap="md">
        {data.checks.length > 0 && (
          <div>
            <Table.ScrollContainer minWidth={900}>
              <Table fz="xs" verticalSpacing={6} highlightOnHover>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Line item</Table.Th>
                    <Table.Th>Should equal</Table.Th>
                    <Table.Th>Status</Table.Th>
                    <Table.Th ta="right">Periods</Table.Th>
                    <Table.Th ta="right">Largest gap</Table.Th>
                    <Table.Th>In</Table.Th>
                    <Table.Th ta="right">Tolerance</Table.Th>
                    <Table.Th>Notes</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {data.checks.map((check) => (
                    <Table.Tr key={`${check.row_label}-${check.formula}`}>
                      <Table.Td fw={600} style={{ whiteSpace: "nowrap" }}>
                        {check.row_label}
                      </Table.Td>
                      <Table.Td>{check.formula}</Table.Td>
                      <Table.Td>
                        <Badge
                          size="xs"
                          variant="light"
                          color={STATUS_COLOR[check.status] ?? "gray"}
                          radius="sm"
                          style={{ minWidth: "max-content" }}
                        >
                          {check.status}
                        </Badge>
                      </Table.Td>
                      <Table.Td ta="right" className="num">
                        {check.periods_checked}
                      </Table.Td>
                      <Table.Td ta="right" className="num">
                        {check.worst_gap === null ? MISSING : Math.round(check.worst_gap).toLocaleString("en-US")}
                      </Table.Td>
                      <Table.Td style={{ whiteSpace: "nowrap" }}>{check.worst_period || MISSING}</Table.Td>
                      <Table.Td ta="right" className="num">
                        {fixed(check.tolerance * 100, 1)}%
                      </Table.Td>
                      <Table.Td c="dimmed">{check.detail || MISSING}</Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </Table.ScrollContainer>
            <Text size="xs" c="dimmed" mt="xs">
              Subtotals are shown <b>as the company reported them</b>, never as the sum of the rows above. Where both
              exist the two are compared here, and a disagreement marks the row rather than silently correcting it. A
              check whose inputs are missing reports <i>not evaluated</i> rather than passing.
            </Text>
          </div>
        )}
        {data.filings.rows.length > 0 && (
          <div>
            <Title order={6} mb={6}>
              The filing behind each column
            </Title>
            <FrameTable frame={data.filings} />
          </div>
        )}
      </Stack>
    </SourcesPanel>
  );
}

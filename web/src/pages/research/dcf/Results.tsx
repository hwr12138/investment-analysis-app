// The answer: both terminal methods side by side, the grid around the
// headline, the bridge from enterprise value to a share, and the forecast
// year by year.
import { Alert, Badge, Card, Stack, Table, Text, Title } from "@mantine/core";
import { IconAlertTriangle } from "@tabler/icons-react";
import type { ReactNode } from "react";

import { fixed, percent } from "../../../format";
import { price } from "../common";
import { OriginBadge } from "../parts";
import { Caption, type Company, money as formatMoney, type Valuation } from "./parts";

type Props = { valuation: Valuation; company: Company; unacknowledged: number; ebitaProxy: boolean };

export function Results({ valuation, company, unacknowledged, ebitaProxy }: Props) {
  return (
    <Stack gap="lg">
      {unacknowledged > 0 && (
        <Alert variant="light" color="red" icon={<IconAlertTriangle size={18} />}>
          <Text size="sm">
            <b>
              The figures below carry {unacknowledged} unacknowledged critical data problem
              {unacknowledged === 1 ? "" : "s"}.
            </b>{" "}
            Read the red banner above before using this number.
          </Text>
        </Alert>
      )}
      <Methods valuation={valuation} company={company} />
      <SensitivityGrid valuation={valuation} company={company} />
      <EquityBridge valuation={valuation} company={company} />
      <Projection valuation={valuation} company={company} />
      {valuation.terminal_year && <TerminalYear valuation={valuation} company={company} ebitaProxy={ebitaProxy} />}
    </Stack>
  );
}

function Block({ title, children }: { title: string; children: ReactNode }) {
  return (
    <Card>
      <Title order={4} mb="sm">
        {title}
      </Title>
      <Stack gap="sm">{children}</Stack>
    </Card>
  );
}

type Part = { valuation: Valuation; company: Company };

function Methods({ valuation, company }: Part) {
  const money = (value: number) => formatMoney(value, company.valuation_currency);
  const gordon = valuation.methods.find((m) => m.method === "gordon");
  const exit = valuation.methods.find((m) => m.method === "exit_multiple");
  return (
    <Block title="Both terminal value methods, side by side">
      <Table.ScrollContainer minWidth={1040}>
        <Table fz="sm" verticalSpacing={8} highlightOnHover>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Method</Table.Th>
              <Table.Th ta="right">Terminal value</Table.Th>
              <Table.Th ta="right">PV of terminal value</Table.Th>
              <Table.Th ta="right">Enterprise value</Table.Th>
              <Table.Th ta="right">Equity value</Table.Th>
              <Table.Th ta="right">Value per share</Table.Th>
              <Table.Th ta="right">TV share of EV</Table.Th>
              <Table.Th miw={280}>This method implies</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {valuation.methods.map((m) => (
              <Table.Tr key={m.method} fw={m.headline ? 600 : undefined}>
                <Table.Td style={{ whiteSpace: "nowrap" }}>
                  {m.label}{" "}
                  {m.headline && (
                    <Badge size="xs" variant="light" ml={4}>
                      headline
                    </Badge>
                  )}
                </Table.Td>
                <Table.Td ta="right" className="num">
                  {money(m.terminal_value)}
                </Table.Td>
                <Table.Td ta="right" className="num">
                  {money(m.present_value)}
                </Table.Td>
                <Table.Td ta="right" className="num">
                  {money(m.enterprise_value)}
                </Table.Td>
                <Table.Td ta="right" className="num">
                  {money(m.equity_value)}
                </Table.Td>
                <Table.Td ta="right" className="num">
                  {price(m.value_per_share, company.valuation_currency)}
                </Table.Td>
                <Table.Td ta="right" className="num">
                  {m.share_of_ev === null ? "—" : percent(m.share_of_ev, 1)}
                </Table.Td>
                <Table.Td fw={400}>{m.implies}</Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
      <Caption>
        **Reading the cross-check.** Each method is restated in the other's terms, so the last column is a different
        quantity on each row — that is the point of it. The Gordon row answers *what exit multiple would my perpetuity
        assumption have to be worth?*, and the exit-multiple row *what is my chosen multiple secretly assuming?* — the
        perpetual growth it needs even if growth cost nothing, and the return on new capital it needs at your terminal
        rate. If either is something you would never underwrite, the multiple is too high.
      </Caption>
      {gordon && exit && (
        <Caption>
          {`Gordon value discounted at exponent ${fixed(gordon.discount_exponent, 4)}, the final forecast year's own (${
            valuation.mid_year ? "mid-year" : "end-of-year"
          }) convention, because it is a perpetuity of flows arriving the same way. Exit value discounted at ${fixed(
            exit.discount_exponent,
            4,
          )} — the end of year N — because a multiple is a sale price struck on that date.`}
        </Caption>
      )}
    </Block>
  );
}

function SensitivityGrid({ valuation, company }: Part) {
  const grid = valuation.sensitivity;
  const middle = Math.floor(grid.waccs.length / 2);
  const centre = Math.floor(grid.columns.length / 2);
  const method = grid.method === "exit_multiple" ? "exit multiple" : "Gordon growth";
  const across =
    grid.method === "exit_multiple"
      ? "the exit multiple at −20%, −10%, as set, +10% and +20% across the top — terminal growth barely moves an exit-multiple value, so it is not what the grid varies"
      : "terminal growth across the top";
  return (
    <Block title={`Sensitivity — value per share, ${method} headline`}>
      <Caption>
        {`The ${method} headline rerun with the WACC down the side and ${across}; everything else as above. The centre cell is the headline. A blank cell is one where the model cannot be built — a WACC at or below terminal growth, where no perpetuity exists.`}
      </Caption>
      <Table.ScrollContainer minWidth={560}>
        <Table fz="sm" verticalSpacing={6} withColumnBorders maw={760}>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>WACC \ {grid.column_label}</Table.Th>
              {grid.headings.map((heading) => (
                <Table.Th key={heading} ta="right" style={{ whiteSpace: "nowrap" }}>
                  {heading}
                </Table.Th>
              ))}
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {grid.waccs.map((wacc, i) => (
              <Table.Tr key={wacc}>
                <Table.Th className="num">{percent(wacc)}</Table.Th>
                {(grid.values[i] ?? []).map((value, j) => {
                  const headline = i === middle && j === centre;
                  return (
                    <Table.Td
                      key={j}
                      ta="right"
                      className="num"
                      fw={headline ? 700 : undefined}
                      bg={headline ? "var(--mantine-primary-color-light)" : undefined}
                    >
                      {value === null ? "" : fixed(value)}
                    </Table.Td>
                  );
                })}
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
      {company.last_price !== null && (
        <Caption>{`Last close: **${price(company.last_price, company.valuation_currency)}**.`}</Caption>
      )}
    </Block>
  );
}

function EquityBridge({ valuation, company }: Part) {
  const bridge = valuation.bridge;
  const currency = company.valuation_currency;
  return (
    <Block title="Equity bridge">
      <Caption>
        Every bridge item is taken from the balance sheet the base period closes on. Sharadar provides no
        treasury-stock-method calculation for outstanding options and RSUs, so the diluted count is not fully loaded.
      </Caption>
      <Table.ScrollContainer minWidth={560}>
        <Table fz="sm" verticalSpacing={6} maw={900}>
          <Table.Tbody>
            {bridge.lines.map((line) => {
              const total = line.kind !== "adjustment";
              return (
                <Table.Tr key={line.label} fw={total ? 600 : undefined}>
                  <Table.Td>{line.label}</Table.Td>
                  <Table.Td ta="right" className="num" style={{ whiteSpace: "nowrap" }}>
                    {formatMoney(line.amount, currency)}
                  </Table.Td>
                  <Table.Td>
                    <OriginBadge origin={line.reported ? "reported" : "unavailable"} />
                  </Table.Td>
                  <Table.Td fw={400}>
                    <Text size="xs" c="dimmed">
                      {line.note}
                    </Text>
                  </Table.Td>
                </Table.Tr>
              );
            })}
            <Table.Tr>
              <Table.Td>÷ shares — {bridge.share_basis}</Table.Td>
              <Table.Td ta="right" className="num">
                {Math.round(bridge.shares).toLocaleString("en-US")}
              </Table.Td>
              <Table.Td>
                <OriginBadge origin="reported" />
              </Table.Td>
              <Table.Td />
            </Table.Tr>
            <Table.Tr fw={700}>
              <Table.Td>= Value per share</Table.Td>
              <Table.Td ta="right" className="num">
                {price(bridge.value_per_share, currency)}
              </Table.Td>
              <Table.Td>
                <OriginBadge origin="derived" />
              </Table.Td>
              <Table.Td />
            </Table.Tr>
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
    </Block>
  );
}

type Row = Valuation["projection"][number];

/** The projection reads like a model: line items down the side, years across. */
function Projection({ valuation, company }: Part) {
  const currency = company.valuation_currency;
  const money = (value: number) => formatMoney(value, currency);
  const lines: { label: string; cell: (row: Row) => string; strong?: boolean }[] = [
    { label: "Revenue", cell: (r) => money(r.revenue), strong: true },
    { label: "Revenue growth", cell: (r) => percent(r.revenue_growth) },
    { label: "Operating margin", cell: (r) => percent(r.operating_margin) },
    { label: "EBIT", cell: (r) => money(r.ebit) },
    { label: "Tax rate", cell: (r) => percent(r.tax_rate) },
    { label: "NOPAT", cell: (r) => money(r.nopat) },
    { label: "D&A", cell: (r) => money(r.dna) },
    { label: "CapEx", cell: (r) => money(r.capex) },
    { label: "ΔNWC", cell: (r) => money(r.delta_nwc) },
    { label: "FCFF", cell: (r) => money(r.fcff), strong: true },
    { label: "Years after base period", cell: (r) => fixed(r.years_after, 4) },
    { label: "Discount exponent", cell: (r) => fixed(r.discount_exponent, 4) },
    { label: "Discount factor", cell: (r) => fixed(r.discount_factor, 4) },
    { label: "PV of FCFF", cell: (r) => money(r.present_value), strong: true },
  ];
  const headline = valuation.methods.find((m) => m.headline);
  return (
    <Block title="Projection">
      <Caption>
        FCFF = EBIT × (1 − tax) + D&A − CapEx − ΔNWC. PV = FCFF × discount factor. EBIT is after stock-based
        compensation, which is not added back.
      </Caption>
      <Table.ScrollContainer minWidth={160 + valuation.projection.length * 92}>
        <Table fz="xs" verticalSpacing={5}>
          <Table.Thead>
            <Table.Tr>
              <Table.Th className="sticky-col" />
              {valuation.projection.map((row) => (
                <Table.Th key={row.label} ta="right" style={{ whiteSpace: "nowrap" }}>
                  {row.label}
                </Table.Th>
              ))}
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {lines.map((line) => (
              <Table.Tr key={line.label} fw={line.strong ? 600 : undefined}>
                <Table.Td className="sticky-col" style={{ whiteSpace: "nowrap" }}>
                  {line.label}
                </Table.Td>
                {valuation.projection.map((row) => (
                  <Table.Td key={row.label} ta="right" className="num" style={{ whiteSpace: "nowrap" }}>
                    {line.cell(row)}
                  </Table.Td>
                ))}
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
      <Caption>
        {`Sum of discounted explicit cash flows: **${money(valuation.pv_explicit)}**${
          headline ? ` · PV of terminal value: **${money(headline.present_value)}**` : ""
        }`}
      </Caption>
    </Block>
  );
}

function TerminalYear({ valuation, company, ebitaProxy }: Part & { ebitaProxy: boolean }) {
  const steady = valuation.terminal_year!;
  const money = (value: number) => formatMoney(value, company.valuation_currency);
  const rows: [string, string][] = [
    ["Revenue", money(steady.revenue)],
    ["Operating margin", percent(steady.operating_margin)],
    ["EBIT", money(steady.ebit)],
    ["Amortisation add-back (EBITA proxy)", ebitaProxy ? money(steady.amortisation) : "off"],
    ["NOPAT", money(steady.nopat)],
    ["Tax rate", percent(steady.tax_rate)],
    ["Return on new invested capital", percent(steady.ronic)],
    ["Reinvestment rate (g ÷ RONIC)", percent(steady.reinvestment_rate)],
    ["Reinvestment", money(steady.reinvestment)],
    ["FCFF", money(steady.fcff)],
  ];
  return (
    <Block title="Terminal year (N + 1)">
      <Caption>
        The Gordon value capitalises a normalised year rather than growing the final forecast year: reinvestment is set to
        what growing at g costs at the return on new capital, so reinvestment sized for the forecast's growth is not
        carried into perpetuity. With the EBITA proxy on, NOPAT adds back the final year's D&A in excess of capex, grown
        at g — amortisation that capex never replaces.
      </Caption>
      <Table fz="sm" verticalSpacing={6} maw={520}>
        <Table.Tbody>
          {rows.map(([label, value]) => (
            <Table.Tr key={label}>
              <Table.Td>{label}</Table.Td>
              <Table.Td ta="right" className="num">
                {value}
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Block>
  );
}

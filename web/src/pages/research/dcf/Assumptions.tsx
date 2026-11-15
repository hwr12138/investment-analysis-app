// The assumption panel: every input the run takes, defaulted from the
// company's own history, with where each default came from beside it.
//
// The panel draws what the API resolved, and an edit sends only the change —
// the server re-resolves everything else, because one input can move another's
// default (the cost of debt follows the risk-free rate, a driver its basis).
import {
  Accordion,
  Alert,
  Anchor,
  Button,
  Card,
  Checkbox,
  Code,
  Group,
  Radio,
  SegmentedControl,
  Select,
  SimpleGrid,
  Slider,
  Stack,
  Switch,
  Table,
  Text,
  Title,
} from "@mantine/core";
import { IconExternalLink, IconInfoCircle, IconRestore } from "@tabler/icons-react";
import type { ReactNode } from "react";

import { fixed, isoDay, percent } from "../../../format";
import { NotReported } from "../parts";
import { isUntouched, NO_OVERRIDES, type Overrides, setEntry, setScalar, type Update } from "./overrides";
import {
  Caption,
  type Company,
  Figure,
  Md,
  money as formatMoney,
  Notes,
  NumberField,
  type Panel,
  ProvenanceLine,
  RATE,
} from "./parts";

type Props = { panel: Panel; company: Company; overrides: Overrides; update: Update };

const FINRA = "https://www.finra.org/finra-data/fixed-income";
const WARNING_MARK = "⚠️";

export function Assumptions({ panel, company, overrides, update }: Props) {
  return (
    <Stack gap="lg">
      <Group justify="space-between" align="center" wrap="wrap" gap="sm">
        <Caption>
          Every default is derived from reported history and shows where it came from. Nothing here is invented —
          where a basis does not exist, the field says so.
        </Caption>
        {!isUntouched(overrides) && (
          <Button
            variant="light"
            size="xs"
            leftSection={<IconRestore size={15} />}
            onClick={() => update(() => NO_OVERRIDES)}
          >
            Reset every assumption
          </Button>
        )}
      </Group>
      <ForecastDrivers panel={panel} overrides={overrides} update={update} />
      <DiscountRate panel={panel} company={company} overrides={overrides} update={update} />
      <TerminalValue panel={panel} overrides={overrides} update={update} />
      <CashAndBridge panel={panel} company={company} overrides={overrides} update={update} />
      <BaseFcff panel={panel} company={company} />
    </Stack>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <Card>
      <Title order={4} mb="sm">
        {title}
      </Title>
      <Stack gap="md">{children}</Stack>
    </Card>
  );
}

function Label({ children }: { children: ReactNode }) {
  return (
    <Text size="sm" fw={600}>
      {children}
    </Text>
  );
}

// ---------------------------------------------------------------------------
// Forecast drivers
// ---------------------------------------------------------------------------

type Part = Omit<Props, "company">;

function ForecastDrivers({ panel, overrides, update }: Part) {
  const low = panel.years_range[0] ?? 5;
  const high = panel.years_range[1] ?? 15;
  return (
    <Section title="Forecast drivers">
      <Caption>{`Base period: **${panel.base_label}** — ${panel.base_detail} Balance sheet as of **${panel.stocks_as_of ?? "not reported"}**.`}</Caption>
      <div>
        <Label>Forecast years: {panel.years}</Label>
        <Slider
          key={panel.years}
          defaultValue={panel.years}
          min={low}
          max={high}
          step={1}
          marks={[low, 10, high].map((value) => ({ value, label: String(value) }))}
          mb="lg"
          mt={4}
          maw={480}
          onChangeEnd={(years) =>
            update((o) => {
              const next = setScalar(o, "years", years);
              // The hold has to leave at least one year to fade over.
              return o.hold != null && o.hold > years - 1 ? setScalar(next, "hold", years - 1) : next;
            })
          }
        />
        <Caption>
          Ten years by default, so a company still growing well above the terminal rate has room to compound before it
          fades. The excess return lives in the explicit forecast; the terminal year earns only its RONIC.
        </Caption>
      </div>

      <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="lg" verticalSpacing="lg">
        {panel.drivers.map((driver) => (
          <DriverBlock key={driver.key} driver={driver} overrides={overrides} update={update} />
        ))}
      </SimpleGrid>

      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="lg">
        <Stack gap="xs">
          <Label>Growth path</Label>
          <Switch
            label="Fade revenue growth toward terminal growth"
            checked={panel.fade}
            onChange={(event) => update((o) => setScalar(o, "fade", event.currentTarget.checked))}
          />
          <Caption>
            Year-1 growth is held for the years below, then steps linearly toward the perpetuity rate, landing on it in
            the terminal year — no cliff between year N and the terminal value. Off holds the rate flat for every
            forecast year.
          </Caption>
          <NumberField
            label="Years to hold year-1 growth"
            maw={240}
            value={panel.hold}
            places={0}
            min={1}
            max={panel.years - 1}
            clampBehavior="strict"
            allowDecimal={false}
            disabled={!panel.fade}
            overridden={overrides.hold != null}
            onCommit={(hold) => update((o) => setScalar(o, "hold", hold === null ? null : Math.round(hold)))}
          />
        </Stack>
        <Stack gap="md">
          {panel.targets.map((target) => (
            <TargetBlock key={target.key} target={target} years={panel.years} overrides={overrides} update={update} />
          ))}
        </Stack>
      </SimpleGrid>

      {panel.acquisitions && (
        <Stack gap={4}>
          <Switch
            label="Include acquisitions in reinvestment"
            checked={panel.acquisitions.on}
            onChange={(event) => update((o) => setScalar(o, "acquisitions", event.currentTarget.checked))}
          />
          <Caption>
            {`Acquisitions averaged **${percent(panel.acquisitions.share)}** of revenue over ${panel.acquisitions.years_used} year(s)${
              panel.acquisitions.on ? " — added to capex." : ", not in capex."
            } Reported growth includes what acquisitions bought; capex excludes what they cost.`}
          </Caption>
        </Stack>
      )}
      {panel.capex_estimate && (
        <Alert variant="light" color="yellow" p="xs">
          <Text size="xs">
            <Md text={panel.capex_estimate} />
          </Text>
        </Alert>
      )}
    </Section>
  );
}

const DRIVER_NOTES: Record<string, string> = {
  capex_pct_revenue:
    "Cash capex only. Assets obtained under finance leases — material at some hyperscalers — are excluded, and their principal repayments sit in financing cash flow, which FCFF does not see.",
  nwc_pct_revenue:
    "Working capital is held at this share of revenue, so it moves with each year's change in revenue. A negative share — customers funding the business — releases cash as revenue grows.",
};

function DriverBlock({
  driver,
  overrides,
  update,
}: {
  driver: Panel["drivers"][number];
  overrides: Overrides;
  update: Update;
}) {
  const after = driver.input.notes.filter((note) => note.text.startsWith(WARNING_MARK));
  const before = driver.input.notes.filter((note) => !note.text.startsWith(WARNING_MARK));
  return (
    <Stack gap={6}>
      <Select
        label={`${driver.label} — basis`}
        data={driver.bases}
        value={driver.basis}
        allowDeselect={false}
        // A new basis is a new default: a value typed against the old one is dropped.
        onChange={(basis) =>
          basis &&
          update((o) =>
            setEntry(setEntry(o, "bases", driver.key, basis === driver.default_basis ? null : basis), "values", driver.key, null),
          )
        }
      />
      <Notes notes={before} />
      <NumberField
        label={driver.label}
        aria-label={`${driver.label} (%)`}
        {...RATE}
        step={0.25}
        value={driver.input.value}
        fallback={driver.input.default}
        overridden={overrides.values[driver.key] != null}
        onCommit={(value) => update((o) => setEntry(o, "values", driver.key, value))}
      />
      <ProvenanceLine provenance={driver.input.provenance} />
      <Notes notes={after} />
      {DRIVER_NOTES[driver.key] && <Caption>{DRIVER_NOTES[driver.key]}</Caption>}
    </Stack>
  );
}

function TargetBlock({
  target,
  years,
  overrides,
  update,
}: {
  target: Panel["targets"][number];
  years: number;
  overrides: Overrides;
  update: Update;
}) {
  const after = target.notes.filter((note) => note.text.startsWith(WARNING_MARK));
  const before = target.notes.filter((note) => !note.text.startsWith(WARNING_MARK));
  return (
    <Stack gap={6}>
      <Switch
        label={`Move the ${target.label} to a target`}
        checked={target.on}
        onChange={(event) => {
          const on = event.currentTarget.checked;
          update((o) => setEntry(o, "target_on", target.key, on === target.default_on ? null : on));
        }}
      />
      <Notes notes={before} />
      {target.key === "capex_pct_revenue" && (
        <Caption>
          D&A stays at its starting ratio, as the margin does: the margin already deducts today's D&A, so move the two
          together or cash flow moves by the whole change in D&A rather than by its tax shield.
        </Caption>
      )}
      {target.on && target.input && (
        <>
          <NumberField
            label={`Target ${target.label} in year ${years}`}
            {...RATE}
            step={0.5}
            maw={320}
            value={target.input.value}
            fallback={target.seed}
            overridden={overrides.target_values[target.key] != null}
            onCommit={(value) => update((o) => setEntry(o, "target_values", target.key, value))}
          />
          <ProvenanceLine provenance={target.input.provenance} />
        </>
      )}
      <Notes notes={after} />
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// Discount rate
// ---------------------------------------------------------------------------

function DiscountRate({ panel, company, overrides, update }: Props) {
  const currency = company.valuation_currency;
  const regression = panel.regression;
  const wacc = panel.wacc;
  return (
    <Section title="Discount rate">
      <SimpleGrid cols={{ base: 1, md: 3 }} spacing="xl" verticalSpacing="lg">
        <Stack gap="sm">
          <Stack gap={6}>
            <Label>Risk-free rate</Label>
            <Notes notes={panel.risk_free.notes} />
            <NumberField
              aria-label="Risk-free rate (%)"
              description="Should match the tenor of the bond behind your cost of debt."
              {...RATE}
              step={0.05}
              value={panel.risk_free.value}
              fallback={panel.risk_free.default}
              overridden={overrides.risk_free != null}
              onCommit={(value) => update((o) => setScalar(o, "risk_free", value))}
            />
            <ProvenanceLine provenance={panel.risk_free.provenance} />
          </Stack>
          <Stack gap={6}>
            <Label>Equity risk premium</Label>
            <Notes notes={panel.erp.notes} />
            <NumberField
              aria-label="Equity risk premium (%)"
              {...RATE}
              step={0.05}
              value={panel.erp.value}
              fallback={panel.erp.default}
              overridden={overrides.erp != null}
              onCommit={(value) => update((o) => setScalar(o, "erp", value))}
            />
            <ProvenanceLine provenance={panel.erp.provenance} />
          </Stack>
        </Stack>

        <Stack gap={6}>
          <Label>Beta</Label>
          <SegmentedControl
            size="xs"
            data={[
              { value: "monthly", label: "5y monthly" },
              { value: "weekly", label: "2y weekly" },
            ]}
            value={panel.beta_frequency}
            // A new window is a new regression: a typed beta is dropped.
            onChange={(value) =>
              update((o) => setScalar(setScalar(o, "beta_frequency", value as "monthly" | "weekly"), "beta", null))
            }
          />
          <Checkbox
            label="Blume adjustment (0.67 × β + 0.33)"
            checked={panel.blume}
            onChange={(event) => {
              const blume = event.currentTarget.checked;
              update((o) => setScalar(setScalar(o, "blume", blume), "beta", null));
            }}
          />
          <Caption>
            A single stock's regressed beta carries a standard error of 0.2–0.3 over five years; pulling it toward the
            market's 1.0 is the usual correction for that noise.
          </Caption>
          {!regression && <Notes notes={panel.beta.notes} />}
          <NumberField
            aria-label="Beta"
            step={0.05}
            places={4}
            value={panel.beta.value}
            fallback={panel.beta.default}
            overridden={overrides.beta != null}
            onCommit={(value) => update((o) => setScalar(o, "beta", value))}
          />
          {regression && (
            <>
              <Group gap="lg" align="flex-end">
                <Figure label="Regressed β" value={fixed(regression.beta, 4)} hint={`R² ${fixed(regression.r_squared, 3)}`} />
                {overrides.beta != null && (
                  <Button variant="light" size="xs" onClick={() => update((o) => setScalar(o, "beta", null))}>
                    Reset to regressed
                  </Button>
                )}
              </Group>
              <Caption>
                {`Raw β ${fixed(regression.raw_beta, 4)} · ${regression.observations} ${panel.beta_frequency} observations vs ${regression.index_ticker}${
                  regression.standard_error !== null && regression.interval_low !== null && regression.interval_high !== null
                    ? ` · standard error ${fixed(regression.standard_error, 3)}, 95% interval ${fixed(regression.interval_low)}–${fixed(regression.interval_high)}`
                    : ""
                }`}
              </Caption>
              <Notes notes={panel.beta.notes} />
            </>
          )}
          <ProvenanceLine provenance={panel.beta.provenance} />
        </Stack>

        <Stack gap={6}>
          <Label>Cost of debt</Label>
          <Notes notes={panel.cost_of_debt.notes} />
          <NumberField
            aria-label="Cost of debt — yield to maturity (%)"
            description="Yield to maturity. A traded bond's yield is better than the default; see below."
            {...RATE}
            step={0.05}
            value={panel.cost_of_debt.value}
            fallback={panel.cost_of_debt.default ?? panel.risk_free.value}
            overridden={overrides.cost_of_debt != null}
            onCommit={(value) => update((o) => setScalar(o, "cost_of_debt", value))}
          />
          <ProvenanceLine provenance={panel.cost_of_debt.provenance} />
          {panel.implied_cost_of_debt !== null ? (
            <>
              <Figure label="Implied average cost of debt" value={percent(panel.implied_cost_of_debt)} />
              <Caption>
                Interest expense ÷ average total debt. This is a **backward-looking average coupon**, not a current
                marginal cost — it understates when rates have risen and overstates when they have fallen.
                Cross-check only; it never populates the model.
              </Caption>
            </>
          ) : (
            <Caption>
              The implied cost of debt cross-check is unavailable: Sharadar reports no interest expense for this company.
            </Caption>
          )}
          <NumberField
            label="Marginal tax rate"
            description="What the next dollar of interest saves — not the effective rate that feeds FCFF."
            {...RATE}
            step={0.5}
            mt="xs"
            value={panel.marginal_tax.value}
            fallback={panel.marginal_tax.default}
            overridden={overrides.marginal_tax != null}
            onCommit={(value) => update((o) => setScalar(o, "marginal_tax", value))}
          />
          <Notes notes={panel.marginal_tax.notes} />
          <ProvenanceLine provenance={panel.marginal_tax.provenance} />
        </Stack>
      </SimpleGrid>

      <Accordion variant="contained">
        <Accordion.Item value="kd">
          <Accordion.Control icon={<IconInfoCircle size={16} />}>
            <Text size="sm">Where to find a cost of debt</Text>
          </Accordion.Control>
          <Accordion.Panel>
            <Stack gap="xs">
              <Text size="sm">
                FINRA publishes TRACE-reported corporate bond trades and yields, free. Open <b>Corporate & Agency Bonds</b>,
                search by <b>issuer name</b> rather than CUSIP — a large issuer has dozens — then pick an issue{" "}
                <b>maturing about ten years out</b> and read its <b>yield to maturity</b> off the trade detail. The latest
                10-K's debt footnote lists every issue with its coupon and maturity.
              </Text>
              <Text size="sm">
                <b>Which bond:</b> a recently traded, senior unsecured issue. Secured, subordinated, convertible and
                thinly traded issues do not price the company's general cost of borrowing. <b>Which maturity:</b> the
                tenor of your risk-free rate — consistency between the two matters more than exactly ten years.
              </Text>
              <Anchor href={FINRA} target="_blank" rel="noreferrer" size="sm">
                <Group gap={4} wrap="nowrap">
                  Open FINRA's fixed income data
                  <IconExternalLink size={14} stroke={1.6} />
                </Group>
              </Anchor>
            </Stack>
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>

      <Stack gap={6}>
        <Radio.Group
          label="Share count basis for value per share"
          value={panel.share_basis}
          onChange={(value) => update((o) => setScalar(o, "share_basis", value))}
        >
          <Group gap="lg" mt={6}>
            {panel.share_bases.map((basis) => (
              <Radio key={basis} value={basis} label={basis} />
            ))}
          </Group>
        </Radio.Group>
        <Notes notes={panel.share_notes} />
        {panel.shares !== null && (
          <Caption>
            {`Using **${Math.round(panel.shares).toLocaleString("en-US")}** shares — ${panel.share_label}. Sharadar provides no treasury-stock-method calculation for outstanding options and RSUs, and no if-converted treatment of convertibles, so no basis is fully loaded.${
              company.share_factor !== 1
                ? ` Counts are per ADR: Sharadar's share factor of ${company.share_factor} is applied to every one.`
                : ""
            }`}
          </Caption>
        )}
      </Stack>

      {wacc && (
        <Stack gap={6}>
          <Label>Weighted average cost of capital</Label>
          <SimpleGrid cols={{ base: 2, sm: 3, lg: 5 }} spacing="md">
            <Figure label="WACC" value={percent(wacc.wacc)} />
            <Figure label="Cost of equity" value={percent(wacc.cost_of_equity)} />
            <Figure label="After-tax cost of debt" value={percent(wacc.after_tax_cost_of_debt)} />
            <Figure label="Equity weight" value={percent(wacc.equity_weight, 1)} />
            <Figure label="Debt weight" value={percent(wacc.debt_weight, 1)} />
          </SimpleGrid>
          <Caption>
            <Code fz="inherit">{wacc.formula}</Code> · E = last close × point-in-time shares (
            {formatMoney(panel.equity_value, currency)}) · D = book value of total debt (
            {formatMoney(panel.debt_value, currency)}), as of {panel.stocks_as_of ? isoDay(panel.stocks_as_of) : "—"}.
          </Caption>
        </Stack>
      )}
    </Section>
  );
}

// ---------------------------------------------------------------------------
// Terminal value and conventions
// ---------------------------------------------------------------------------

const RONIC_MANUAL = "Manual";

function TerminalValue({ panel, overrides, update }: Part) {
  const incremental =
    panel.incremental_return !== null ? percent(panel.incremental_return, 1) : panel.incremental_return_note;
  return (
    <Section title="Terminal value and conventions">
      <SimpleGrid cols={{ base: 1, md: 3 }} spacing="xl" verticalSpacing="lg">
        <Stack gap={6}>
          <NumberField
            label="Perpetuity growth rate"
            {...RATE}
            step={0.1}
            value={panel.terminal_growth.value}
            fallback={panel.terminal_growth.default}
            overridden={overrides.terminal_growth != null}
            onCommit={(value) => update((o) => setScalar(o, "terminal_growth", value))}
          />
          <ProvenanceLine provenance={panel.terminal_growth.provenance} />

          <Radio.Group
            label="Return on new invested capital (RONIC)"
            description="The terminal year reinvests g ÷ RONIC of NOPAT to grow at g. However high it goes, the Gordon value rises by at most WACC ÷ (WACC − g)."
            value={panel.ronic_basis}
            mt="sm"
            onChange={(basis) => update((o) => setScalar(setScalar(o, "ronic_basis", basis), "ronic", null))}
          >
            <Stack gap={6} mt={6}>
              {panel.ronic_bases.map((basis) => (
                <Radio key={basis} value={basis} label={basis} />
              ))}
            </Stack>
          </Radio.Group>
          <Notes notes={panel.ronic.notes} />
          {panel.ronic_basis === RONIC_MANUAL && (
            <NumberField
              aria-label="Return on new invested capital (%)"
              {...RATE}
              step={0.5}
              min={0.1}
              max={1000}
              clampBehavior="blur"
              value={panel.ronic.value}
              fallback={panel.ronic.default}
              overridden={overrides.ronic != null}
              onCommit={(value) => update((o) => setScalar(o, "ronic", value))}
            />
          )}
          <ProvenanceLine provenance={panel.ronic.provenance} />
          <Caption>
            {`**Forecast incremental return on capital: ${incremental}** — (NOPAT in year ${panel.years} − base NOPAT) ÷ Σ (capex − D&A + ΔNWC). The excess return the explicit forecast carries before the terminal year earns its RONIC.`}
          </Caption>

          <Switch
            mt="sm"
            label="Terminal tax at the marginal rate"
            checked={panel.terminal_at_marginal}
            onChange={(event) => update((o) => setScalar(o, "terminal_at_marginal", event.currentTarget.checked))}
          />
          <Caption>
            Low effective rates from credits, deferrals or a foreign mix rarely last forever. On: the terminal year pays
            the marginal rate. Off: it keeps the forecast's effective rate in perpetuity.
          </Caption>
          <Switch
            mt="xs"
            label="Terminal NOPAT before amortisation (EBITA proxy)"
            checked={panel.ebita_proxy}
            onChange={(event) => update((o) => setScalar(o, "ebita_proxy", event.currentTarget.checked))}
          />
          <Caption>
            Adds back max(D&A − capex, 0) from the final forecast year, grown at g — amortisation of acquired intangibles
            that capex never replaces. Moves the Gordon value, and the exit row's implied RONIC and zero-reinvestment
            growth; never the exit value.
          </Caption>
        </Stack>

        <Stack gap={6}>
          <Notes notes={panel.exit_multiple.notes} />
          <NumberField
            label="Exit EV/EBITDA multiple"
            suffix="×"
            step={0.25}
            min={0.01}
            value={panel.exit_multiple.value}
            fallback={panel.exit_multiple.default}
            overridden={overrides.exit_multiple != null}
            onCommit={(value) => update((o) => setScalar(o, "exit_multiple", value))}
          />
          <ProvenanceLine provenance={panel.exit_multiple.provenance} />
          <Caption>
            The default is the company's *own* recent multiple, earned while it grew at its historical rate. By year N
            the forecast has slowed it toward terminal growth, so a mature multiple — a peer or sector median — is often
            the more consistent choice.
          </Caption>
        </Stack>

        <Stack gap="sm">
          <div>
            <Label>Headline method</Label>
            <SegmentedControl
              mt={6}
              data={[
                { value: "gordon", label: "Gordon growth" },
                { value: "exit_multiple", label: "Exit multiple" },
              ]}
              value={panel.headline_method}
              onChange={(value) => update((o) => setScalar(o, "headline_method", value as "gordon" | "exit_multiple"))}
            />
            <Caption>
              Drives the headline value per share and the equity bridge. Both methods always compute and both are shown
              side by side.
            </Caption>
          </div>
          <Switch
            label="Mid-year discounting"
            checked={panel.mid_year}
            onChange={(event) => update((o) => setScalar(o, "mid_year", event.currentTarget.checked))}
          />
          <Caption>
            Each year's cash flow is discounted from the middle of the year. Applies to the Gordon value, a perpetuity of
            such flows, but not to an exit multiple, which is a sale at the end of year N.
          </Caption>
        </Stack>
      </SimpleGrid>
      <Notes notes={[panel.elapsed_note]} />
    </Section>
  );
}

// ---------------------------------------------------------------------------
// Cash, SBC and the bridge
// ---------------------------------------------------------------------------

/** Money is typed in millions: raw dollars run to twelve digits. */
const MILLIONS = 1e-6;

function CashAndBridge({ panel, company, overrides, update }: Props) {
  const currency = company.valuation_currency;
  const money = (value: number | null) => (value === null ? <NotReported /> : formatMoney(value, currency));
  const asOf = panel.stocks_as_of ? isoDay(panel.stocks_as_of) : "—";
  return (
    <Section title="Cash, SBC and bridge items">
      <SimpleGrid cols={{ base: 1, md: 3 }} spacing="xl" verticalSpacing="lg">
        <Stack gap={6}>
          <Switch
            label="Treat short-term investments as excess cash"
            checked={panel.include_sti}
            onChange={(event) => update((o) => setScalar(o, "include_sti", event.currentTarget.checked))}
          />
          <Caption>Off treats short-term investments as operating assets.</Caption>
          <NumberField
            label={`Operating cash floor to hold back (${currency} m)`}
            description="Cash the business needs to run, excluded from the bridge."
            scale={MILLIONS}
            places={0}
            step={1000}
            min={0}
            thousandSeparator=","
            value={panel.cash_floor}
            fallback={0}
            overridden={overrides.cash_floor != null}
            onCommit={(value) => update((o) => setScalar(o, "cash_floor", value))}
          />
          <Caption>
            Balance-sheet cash {money(panel.cash)} · short-term investments {money(panel.short_term_investments)} (as of{" "}
            {asOf})
          </Caption>
        </Stack>

        <Stack gap={6}>
          <Label>Stock-based compensation</Label>
          <Notes notes={panel.sbc.notes} />
          <NumberField
            aria-label={`Stock-based compensation (${currency} m)`}
            scale={MILLIONS}
            places={0}
            step={100}
            thousandSeparator=","
            suffix=" m"
            value={panel.sbc.value}
            fallback={panel.sbc.default}
            overridden={overrides.sbc != null}
            onCommit={(value) => update((o) => setScalar(o, "sbc", value))}
          />
          <ProvenanceLine provenance={panel.sbc.provenance} />
          {panel.sbc.value !== null && (
            <Alert variant="light" color="blue" p="xs" icon={<IconInfoCircle size={16} />}>
              <Text size="xs">
                <b>SBC is expensed, once.</b> The operating margin is taken from GAAP operating income, which already
                deducts it, so this figure is shown for reference and does not move the valuation. Most published FCF
                figures add it back, so this model's FCFF sits below theirs by about this amount.
              </Text>
            </Alert>
          )}
        </Stack>

        <Stack gap={6}>
          <Label>Non-operating assets</Label>
          <Caption>
            {panel.long_term_investments !== null
              ? `Sharadar reports **${formatMoney(panel.long_term_investments, currency)}** of non-current investments (as of ${asOf}), at book value. It mixes marketable securities with equity-method stakes and restricted balances, so it is not added unless you choose to.`
              : "Sharadar reports no non-current investments line for this company."}
          </Caption>
          <NumberField
            label={`Non-operating assets to add (${currency} m)`}
            scale={MILLIONS}
            places={0}
            step={100}
            thousandSeparator=","
            value={panel.non_operating}
            fallback={0}
            overridden={overrides.non_operating != null}
            onCommit={(value) => update((o) => setScalar(o, "non_operating", value))}
          />
        </Stack>
      </SimpleGrid>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// Base-period FCFF
// ---------------------------------------------------------------------------

function BaseFcff({ panel, company }: { panel: Panel; company: Company }) {
  const currency = company.valuation_currency;
  const fcff = panel.base_fcff;
  const money = (value: number | null) => (value === null ? <NotReported /> : formatMoney(value, currency));
  return (
    <Section title="Base-period free cash flow, built two independent ways">
      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="xl">
        <Stack gap={6}>
          <Label>Method A — cash flow statement build</Label>
          <Code block>FCFF = CFO + Interest Expense × (1 − tax rate) − CapEx</Code>
          <Figure label="FCFF (method A)" value={money(fcff.method_a)} />
        </Stack>
        <Stack gap={6}>
          <Label>Method B — income statement build</Label>
          <Code block>{"NOPAT = EBIT × (1 − tax rate)\nFCFF  = NOPAT + D&A − CapEx − ΔNWC"}</Code>
          <Figure label="FCFF (method B)" value={money(fcff.method_b)} />
        </Stack>
      </SimpleGrid>
      {fcff.divergence !== null && <Caption>{`Divergence between the two builds: **${percent(fcff.divergence)}**`}</Caption>}
      {fcff.notes.map((note, i) => (
        <Alert key={i} variant="light" color="yellow" p="xs">
          <Text size="xs">
            <Md text={note} />
          </Text>
        </Alert>
      ))}
      <Table.ScrollContainer minWidth={320}>
        <Table fz="sm" verticalSpacing={6} maw={520}>
          <Table.Tbody>
            {fcff.inputs.map((input) => (
              <Table.Tr key={input.name}>
                <Table.Td>{input.name}</Table.Td>
                <Table.Td ta="right" className="num">
                  {input.value === null ? (
                    <NotReported />
                  ) : input.name === "Tax rate" ? (
                    percent(input.value)
                  ) : (
                    formatMoney(input.value, currency)
                  )}
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
      {panel.etr !== null ? (
        <Caption>
          {`Effective tax rate in use: **${percent(panel.etr)}** — median of the last ${panel.etr_years} profitable year(s). Years with a pretax loss are excluded because the ratio is meaningless there, not set to zero.`}
        </Caption>
      ) : (
        <Alert variant="light" color="yellow" p="xs">
          <Text size="xs">
            No profitable year is available to compute an effective tax rate, so the forecast tax rate driver is used for
            the base period as well.
          </Text>
        </Alert>
      )}
      {panel.delta_nwc_span && <Caption>{`Change in net working capital measured over — ${panel.delta_nwc_span}.`}</Caption>}
      <Caption>
        Method A assumes interest paid sits in operating cash flow, which holds under US GAAP; an IFRS filer may classify
        it in financing. Sharadar exposes interest *expense* (accrual), not interest *paid* (cash). The two builds
        legitimately differ where working capital moves a lot: method B subtracts the balance-sheet movement, while method
        A picks up whatever the cash flow statement booked, including acquisitions and FX translation.
      </Caption>
    </Section>
  );
}


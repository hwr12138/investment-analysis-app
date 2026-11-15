// 💰 DCF — what the company is worth on its own cash flows, under assumptions
// the reader can see and change.
//
// The tab is one request: the reader's overrides go up, and the resolved panel
// and the valued run come back together, because one input can move another's
// default. Only what the reader changed is sent; the overrides and the
// acknowledgements are this company's, so a new ticker starts clean.
//
// The headline and the banners sit above the sub-tabs, so whichever one is open
// shows what an edit did to the value and anything that is wrong with it.
import { Alert, Box, Button, Group, Loader, Menu, SegmentedControl, SimpleGrid, Skeleton, Stack, Text } from "@mantine/core";
import { useMutation } from "@tanstack/react-query";
import { IconAlertTriangle, IconDownload, IconFileSpreadsheet, IconFileTypeCsv } from "@tabler/icons-react";
import { useEffect, useState } from "react";

import { api, ApiError } from "../../api/client";
import { Stat } from "../../components/Stat";
import { isoDay, percent } from "../../format";
import { price, signedPercent, toneColor, useDcf, useResearchParams } from "./common";
import { Assumptions } from "./dcf/Assumptions";
import { type Acknowledged, Banners, BlockedCard, ValidationPanel } from "./dcf/Checks";
import { HistoryPanel } from "./dcf/History";
import { NO_OVERRIDES, type Overrides, type Update } from "./dcf/overrides";
import { Caption, type Company, type Dcf, type Panel, type Valuation } from "./dcf/parts";
import { Results } from "./dcf/Results";
import { Reverse } from "./dcf/Reverse";
import { Segments } from "./parts";

const PARTS = [
  { value: "valuation", label: "Valuation" },
  { value: "assumptions", label: "Assumptions" },
  { value: "reverse", label: "Reverse DCF" },
  { value: "checks", label: "Checks" },
  { value: "history", label: "History" },
] as const;
type Part = (typeof PARTS)[number]["value"];

export function DcfTab({ ticker }: { ticker: string }) {
  // Keyed on the ticker, so the overrides and acknowledgements reset with it.
  return <DcfBody key={ticker} ticker={ticker} />;
}

function DcfBody({ ticker }: { ticker: string }) {
  const { params, set } = useResearchParams();
  const part: Part = PARTS.some((p) => p.value === params.get("part")) ? (params.get("part") as Part) : "valuation";
  const [overrides, setOverrides] = useState<Overrides>(NO_OVERRIDES);
  const [acknowledged, setAcknowledged] = useState<Acknowledged>({});
  const dcf = useDcf(ticker, overrides);

  // The last answer the API gave, and the overrides that produced it: a
  // refused input leaves it on screen with a way back.
  const [good, setGood] = useState<{ overrides: Overrides; data: Dcf } | null>(null);
  useEffect(() => {
    if (dcf.data && !dcf.isPlaceholderData) setGood({ overrides, data: dcf.data });
    // `overrides` is the key `dcf.data` was fetched for whenever it is not a placeholder.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dcf.data, dcf.isPlaceholderData]);

  const update: Update = (change) => setOverrides((current) => change(current));
  const data = dcf.data ?? good?.data;

  if (!data) {
    if (dcf.isError) {
      return (
        <Alert variant="light" color="red" icon={<IconAlertTriangle size={18} />} title={`Couldn't value ${ticker}`}>
          {dcf.error.message}
        </Alert>
      );
    }
    return (
      <Stack gap="xs">
        <Skeleton h={120} radius="md" />
        <Skeleton h={360} radius="md" />
        <Text size="xs" c="dimmed">
          Loading the statements, prices and rates, then valuing {ticker}. A company's first load can take half a minute;
          after that an assumption change answers in a fraction of a second.
        </Text>
      </Stack>
    );
  }

  const { company, panel, valuation, blocked } = data;
  if (!company || !panel) {
    return blocked ? <BlockedCard blocked={blocked} /> : null;
  }

  const refused = dcf.isError && dcf.error instanceof ApiError && dcf.error.status === 422;
  const stale = dcf.isPlaceholderData || dcf.isError;
  const unacknowledged = valuation
    ? valuation.issues.filter((issue) => issue.severity === "critical" && !(issue.key in acknowledged)).length
    : 0;

  return (
    <Stack gap="lg">
      {/* Dimmed with the rest while it describes the last run rather than the panel on screen. */}
      <Box style={{ opacity: stale ? 0.6 : 1, transition: "opacity 120ms" }}>
        <Headline company={company} panel={panel} valuation={valuation} unacknowledged={unacknowledged} updating={dcf.isFetching} />
      </Box>

      {dcf.isError && (
        <Alert
          variant="light"
          color={refused ? "yellow" : "red"}
          icon={<IconAlertTriangle size={18} />}
          title={refused ? "That input was refused" : "Couldn't revalue"}
        >
          <Stack gap="xs" align="flex-start">
            <Text size="sm">{dcf.error.message}</Text>
            {good && (
              <Button size="xs" variant="light" onClick={() => setOverrides(good.overrides)}>
                Undo the last change
              </Button>
            )}
          </Stack>
        </Alert>
      )}
      {blocked && <BlockedCard blocked={blocked} />}
      {valuation && (
        <Banners
          issues={valuation.issues}
          acknowledged={acknowledged}
          onAcknowledge={(key) =>
            setAcknowledged((current) => ({ ...current, [key]: new Date().toLocaleString("sv-SE") }))
          }
          onRestore={(key) =>
            setAcknowledged((current) => {
              const next = { ...current };
              delete next[key];
              return next;
            })
          }
        />
      )}

      <Group justify="space-between" align="center" wrap="wrap" gap="sm">
        <Segments>
          <SegmentedControl
            data={[...PARTS]}
            value={part}
            onChange={(value) => set({ part: value === "valuation" ? null : value })}
          />
        </Segments>
        {valuation && <Export ticker={ticker} overrides={overrides} acknowledged={acknowledged} />}
      </Group>

      <Box style={{ opacity: stale && part !== "assumptions" ? 0.6 : 1, transition: "opacity 120ms" }}>
        {part === "assumptions" && <Assumptions panel={panel} company={company} overrides={overrides} update={update} />}
        {part === "history" && data.history && <HistoryPanel history={data.history} company={company} />}
        {(part === "valuation" || part === "reverse" || part === "checks") && !valuation && (
          <Text size="sm" c="dimmed">
            No valuation to show — {blocked?.title ?? "the run did not complete"}. The assumption panel is still open, so
            the input that caused it can be changed back.
          </Text>
        )}
        {part === "valuation" && valuation && (
          <Results valuation={valuation} company={company} unacknowledged={unacknowledged} ebitaProxy={panel.ebita_proxy} />
        )}
        {part === "reverse" && valuation && (
          <Reverse ticker={ticker} overrides={overrides} valuation={valuation} panel={panel} company={company} />
        )}
        {part === "checks" && valuation && (
          <ValidationPanel valuation={valuation} reference={data.reference} yearsRetrieved={company.years_retrieved} />
        )}
      </Box>
    </Stack>
  );
}

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

function Headline({
  company,
  panel,
  valuation,
  unacknowledged,
  updating,
}: {
  company: Company;
  panel: Panel;
  valuation: Valuation | null;
  unacknowledged: number;
  updating: boolean;
}) {
  const currency = company.valuation_currency;
  const method = valuation?.headline_method === "exit_multiple" ? "Exit multiple" : "Gordon growth";
  const facts = [
    company.price_date && `Price as of ${isoDay(company.price_date)}`,
    company.shares_date && `share count as of ${isoDay(company.shares_date)}`,
    `statements ${company.asreported ? "as reported" : "restated"}`,
    company.template || "sector not classified",
    `${company.years_retrieved} fiscal years retrieved`,
    company.fiscal_year_end_month && `fiscal year ends in ${MONTHS[company.fiscal_year_end_month - 1]}`,
  ].filter(Boolean);

  return (
    <Stack gap="xs">
      <SimpleGrid cols={{ base: 2, md: 4 }} spacing="md">
        <Stat
          label={unacknowledged ? "DCF value per share ⚠" : "DCF value per share"}
          value={valuation ? price(valuation.value_per_share, currency) : "—"}
          color={unacknowledged ? "red" : undefined}
          hint={
            <Group gap={6}>
              <Text size="xs" c="dimmed">
                {valuation ? `Headline: ${method}` : "No valuation"}
              </Text>
              {updating && <Loader size={10} />}
            </Group>
          }
        />
        <Stat
          label="Against the last close"
          value={valuation?.upside != null ? signedPercent(valuation.upside) : "—"}
          color={valuation?.upside != null ? toneColor(valuation.upside > 0) : undefined}
          hint={company.last_price !== null ? `Last close ${price(company.last_price, currency)}` : "No close reported"}
        />
        <Stat
          label="WACC"
          value={panel.wacc ? percent(panel.wacc.wacc) : "—"}
          hint={panel.wacc ? `Cost of equity ${percent(panel.wacc.cost_of_equity)}` : undefined}
        />
        <Stat label="Forecast" value={`${panel.years} years`} hint={`From ${panel.base_label}`} />
      </SimpleGrid>
      <Caption>{facts.join(" · ")}</Caption>
      {company.converted && company.fx_rate !== null && (
        <Caption>
          {`**Reported in ${company.currency}, valued in ${currency}.** Drivers come from the ${company.currency} history, so growth and margins carry no exchange-rate noise; the base period is converted once at a spot rate of **${Number(company.fx_rate.toPrecision(6))} ${currency} per ${company.currency}**, which the valuation assumes holds. History tables stay in ${company.currency}; the projection, the bridge and value per share are in ${currency}.`}
        </Caption>
      )}
    </Stack>
  );
}

function Export({ ticker, overrides, acknowledged }: { ticker: string; overrides: Overrides; acknowledged: Acknowledged }) {
  const download = useMutation({
    mutationFn: async (format: "xlsx" | "csv") => {
      const { data, error, response } = await api.POST("/api/research/{ticker}/dcf/export", {
        params: { path: { ticker } },
        body: { overrides, acknowledged, format },
        parseAs: "blob",
      });
      if (error !== undefined || !data) throw new ApiError(response.status, `The export failed (${response.status}).`);
      const name = /filename="?([^";]+)"?/.exec(response.headers.get("content-disposition") ?? "")?.[1] ?? `${ticker}_dcf.${format}`;
      const url = URL.createObjectURL(data as Blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = name;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    },
  });

  return (
    <Group gap="xs">
      {download.isError && (
        <Text size="xs" c="red">
          {download.error.message}
        </Text>
      )}
      <Menu position="bottom-end" withinPortal>
        <Menu.Target>
          <Button variant="default" size="xs" leftSection={download.isPending ? <Loader size={12} /> : <IconDownload size={15} />}>
            Export
          </Button>
        </Menu.Target>
        <Menu.Dropdown>
          <Menu.Label>The run on screen, with every assumption's provenance</Menu.Label>
          <Menu.Item leftSection={<IconFileSpreadsheet size={16} />} onClick={() => download.mutate("xlsx")}>
            Excel workbook
          </Menu.Item>
          <Menu.Item leftSection={<IconFileTypeCsv size={16} />} onClick={() => download.mutate("csv")}>
            CSV
          </Menu.Item>
        </Menu.Dropdown>
      </Menu>
    </Group>
  );
}


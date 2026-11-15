// 🔎 Stock Research — one company, a tab per question.
//
// The shell: the company's header from the profile, the ticker box, and the tab
// strip. The profile is fetched first and every tab waits for it, so a cold
// search assembles the snapshot once. Only the open tab is mounted, so only it
// fetches; the tab, like every choice inside one, is in the URL.
import {
  Alert,
  Anchor,
  Badge,
  Center,
  Group,
  Loader,
  Skeleton,
  Stack,
  Tabs,
  Text,
  Title,
} from "@mantine/core";
import {
  IconAlertTriangle,
  IconBuildingBank,
  IconChartCandle,
  IconExternalLink,
  IconFileSpreadsheet,
  IconScale,
  IconUsers,
} from "@tabler/icons-react";
import { useNavigate } from "react-router";

import { ApiError } from "../../api/client";
import { TickerSearch } from "../../components/TickerSearch";
import { isoDay } from "../../format";
import { FinancialsTab } from "./FinancialsTab";
import { HoldingsTab } from "./HoldingsTab";
import { InsidersTab } from "./InsidersTab";
import { SummaryTab } from "./SummaryTab";
import { ValuationTab } from "./ValuationTab";
import { price, type Profile, type Tab, useProfile, useResearchParams } from "./common";

// Valuation follows Summary, Insiders and Holdings read as an ownership pair,
// and the statements close the strip as the reference the rest draw on.
const TAB_LABELS: { value: Tab; label: string; icon: typeof IconScale }[] = [
  { value: "summary", label: "Summary", icon: IconChartCandle },
  { value: "valuation", label: "Relative valuation", icon: IconScale },
  { value: "insiders", label: "Insiders", icon: IconUsers },
  { value: "holdings", label: "Holdings", icon: IconBuildingBank },
  { value: "financials", label: "Financials", icon: IconFileSpreadsheet },
];

export function ResearchPage() {
  const { ticker, params, tab, set } = useResearchParams();
  const navigate = useNavigate();
  const profile = useProfile(ticker);
  const data = profile.data;

  return (
    <Stack gap="lg" maw={1440} mx="auto">
      <Group justify="space-between" align="flex-end" wrap="wrap" gap="md">
        <div style={{ minWidth: 0 }}>
          <Text size="xs" fw={700} c="dimmed" tt="uppercase" lts={0.6}>
            Stock research
          </Text>
          <Group gap="sm" align="baseline" wrap="wrap">
            <Title order={1} fz={34} lh={1.15}>
              {data?.name || ticker}
            </Title>
            {data && data.name && (
              <Badge variant="light" size="lg">
                {data.ticker}
              </Badge>
            )}
          </Group>
          {data ? <ProfileLine profile={data} /> : profile.isPending && <Skeleton h={14} w={320} mt={8} />}
        </div>
        <Group gap="md" align="flex-end" wrap="wrap">
          {data?.last_close != null && data.last_session && (
            <Stack gap={0} align="flex-end">
              <Text fz={26} fw={700} lh={1.2} className="num">
                {price(data.last_close, data.currency)}
              </Text>
              <Text size="xs" c="dimmed">
                Close {isoDay(data.last_session)}
              </Text>
            </Stack>
          )}
          <TickerSearch
            key={ticker}
            initial={ticker}
            onSubmit={(next) => navigate({ pathname: `/research/${next}`, search: params.toString() })}
          />
        </Group>
      </Group>

      {profile.isPending && (
        <Center mih="40vh">
          <Stack align="center" gap="xs">
            <Loader />
            <Text size="sm" c="dimmed">
              Loading {ticker} from Sharadar…
            </Text>
          </Stack>
        </Center>
      )}
      {profile.isError && <LoadError ticker={ticker} error={profile.error} />}

      {data && (
        <Tabs value={tab} onChange={(value) => set({ tab: value === "summary" ? null : value })} keepMounted={false}>
          <Tabs.List mb="md">
            {TAB_LABELS.map(({ value, label, icon: Icon }) => (
              <Tabs.Tab key={value} value={value} leftSection={<Icon size={16} stroke={1.6} />}>
                {label}
              </Tabs.Tab>
            ))}
          </Tabs.List>
          <Tabs.Panel value="summary">
            <SummaryTab ticker={ticker} />
          </Tabs.Panel>
          <Tabs.Panel value="valuation">
            <ValuationTab ticker={ticker} />
          </Tabs.Panel>
          <Tabs.Panel value="insiders">
            <InsidersTab ticker={ticker} />
          </Tabs.Panel>
          <Tabs.Panel value="holdings">
            <HoldingsTab ticker={ticker} />
          </Tabs.Panel>
          <Tabs.Panel value="financials">
            <FinancialsTab ticker={ticker} />
          </Tabs.Panel>
        </Tabs>
      )}
    </Stack>
  );
}

function ProfileLine({ profile }: { profile: Profile }) {
  const facts = [profile.exchange, profile.sector, profile.industry, profile.currency].filter(Boolean);
  const links = [
    profile.sec_filings && { href: profile.sec_filings, label: "SEC filings" },
    profile.website && { href: profile.website, label: "Website" },
  ].filter((link): link is { href: string; label: string } => Boolean(link));
  return (
    <Group gap="md" mt={4} wrap="wrap">
      {facts.length > 0 && (
        <Text size="sm" c="dimmed">
          {facts.join(" · ")}
        </Text>
      )}
      {links.map(({ href, label }) => (
        <Anchor key={label} href={href} target="_blank" rel="noreferrer" size="sm">
          <Group gap={3} wrap="nowrap">
            {label}
            <IconExternalLink size={13} stroke={1.6} />
          </Group>
        </Anchor>
      ))}
    </Group>
  );
}

function LoadError({ ticker, error }: { ticker: string; error: Error }) {
  if (error instanceof ApiError && error.status === 404) {
    return (
      <Alert variant="light" color="yellow" icon={<IconAlertTriangle size={18} />} title={`No Sharadar record for ${ticker}`}>
        <Text size="sm">{error.message}</Text>
        <Text size="sm" mt="xs">
          This page covers US equities carried in Sharadar's SF1 fundamentals table. ETFs and funds have prices but no
          fundamentals, and are not supported yet.
        </Text>
      </Alert>
    );
  }
  return (
    <Alert variant="light" color="red" icon={<IconAlertTriangle size={18} />} title={`Couldn't load ${ticker}`}>
      {error.message}
    </Alert>
  );
}

// The severity banners, above the answer, and the validation panel.
//
// The banners never collapse: on this tab it must be impossible to look at the
// output and not notice something is wrong. An acknowledged issue leaves the
// banner but stays listed, and is stamped into the export.
import { Accordion, Alert, Badge, Button, Card, Code, Group, SimpleGrid, Spoiler, Stack, Table, Text } from "@mantine/core";
import { IconAlertOctagon, IconAlertTriangle, IconCircleCheck } from "@tabler/icons-react";

import type { Schemas } from "../../../api/client";
import { Stat } from "../../../components/Stat";
import { FrameTable, plural } from "../parts";
import { Caption, Md, type Valuation } from "./parts";

type Issue = Schemas["Issue"];
export type Acknowledged = Record<string, string>;

type BannerProps = {
  issues: Issue[];
  acknowledged: Acknowledged;
  onAcknowledge: (key: string) => void;
  onRestore: (key: string) => void;
};

export function Banners({ issues, acknowledged, onAcknowledge, onRestore }: BannerProps) {
  const live = issues.filter((issue) => !(issue.key in acknowledged));
  const critical = live.filter((issue) => issue.severity === "critical");
  const warnings = live.filter((issue) => issue.severity === "warning");
  const acked = issues.filter((issue) => issue.key in acknowledged);

  return (
    <Stack gap="sm">
      {critical.length > 0 && (
        <Alert
          variant="light"
          color="red"
          icon={<IconAlertOctagon size={20} />}
          title={plural(critical.length, "critical data problem")}
        >
          <Text size="sm" mb="sm">
            A number was computed, but a core input is suspect. Every figure below inherits these problems.
          </Text>
          <Stack gap="xs">
            {critical.map((issue) => (
              <IssueCard key={issue.key} issue={issue} onAcknowledge={onAcknowledge} />
            ))}
          </Stack>
        </Alert>
      )}
      {warnings.length > 0 && (
        <Alert variant="light" color="yellow" icon={<IconAlertTriangle size={20} />} title={plural(warnings.length, "warning")}>
          <Text size="sm" mb="sm">
            Worth knowing. Not necessarily wrong.
          </Text>
          <Stack gap="xs">
            {warnings.map((issue) => (
              <IssueCard key={issue.key} issue={issue} onAcknowledge={onAcknowledge} />
            ))}
          </Stack>
        </Alert>
      )}
      {acked.length > 0 && (
        <Accordion variant="separated">
          <Accordion.Item value="acknowledged">
            <Accordion.Control>
              <Text size="sm" fw={500}>
                Acknowledged ({acked.length})
              </Text>
            </Accordion.Control>
            <Accordion.Panel>
              <Caption>
                Acknowledged checks stay listed and are stamped into the export. They reset when the ticker changes.
              </Caption>
              <Stack gap="xs" mt="xs">
                {acked.map((issue) => (
                  <Group key={issue.key} justify="space-between" wrap="nowrap" align="flex-start" gap="md">
                    <div style={{ minWidth: 0 }}>
                      <Text size="sm">
                        <Badge size="xs" variant="light" color={issue.severity === "critical" ? "red" : "yellow"} mr={6}>
                          {issue.severity}
                        </Badge>
                        <b>{issue.headline}</b> — <Md text={issue.message} />
                      </Text>
                      <Text size="xs" c="dimmed">
                        Acknowledged at {acknowledged[issue.key]}
                      </Text>
                    </div>
                    <Button variant="default" size="xs" onClick={() => onRestore(issue.key)}>
                      Un-acknowledge
                    </Button>
                  </Group>
                ))}
              </Stack>
            </Accordion.Panel>
          </Accordion.Item>
        </Accordion>
      )}
      {live.length === 0 && acked.length === 0 && (
        <Alert variant="light" color="teal" icon={<IconCircleCheck size={20} />}>
          Every validation check passed.
        </Alert>
      )}
    </Stack>
  );
}

function IssueCard({ issue, onAcknowledge }: { issue: Issue; onAcknowledge: (key: string) => void }) {
  return (
    <Card padding="sm">
      <Group justify="space-between" align="flex-start" wrap="nowrap" gap="md">
        <Stack gap={4} style={{ minWidth: 0 }}>
          <Text size="sm" fw={700}>
            {issue.headline}
          </Text>
          {issue.severity === "critical" ? (
            <Text size="sm">
              <Md text={issue.message} />
            </Text>
          ) : (
            // A warning's explanation is two lines until asked for: the banner
            // itself never collapses, but three of them should not bury the answer.
            <Spoiler
              maxHeight={42}
              showLabel={<Text size="xs">Show more</Text>}
              hideLabel={<Text size="xs">Show less</Text>}
            >
              <Text size="sm">
                <Md text={issue.message} />
              </Text>
            </Spoiler>
          )}
          {issue.values.length > 0 && (
            <Text size="sm">
              {issue.values.map((value, i) => (
                <span key={value.label}>
                  {i > 0 && " · "}
                  <b>{value.label}:</b> <span className="num">{value.value}</span>
                </span>
              ))}
            </Text>
          )}
          <Text size="xs" c="dimmed">
            <Code fz="inherit">{issue.issue_type}</Code> · concept: {issue.concept}
            {issue.tolerance && ` · tolerance: ${issue.tolerance}`}
          </Text>
        </Stack>
        <Button variant="default" size="xs" onClick={() => onAcknowledge(issue.key)} style={{ flexShrink: 0 }}>
          Acknowledge
        </Button>
      </Group>
    </Card>
  );
}

const STATUS_COLOR: Record<Schemas["Check"]["status"], string> = {
  pass: "teal",
  fail: "red",
  "not evaluated": "gray",
};

export function ValidationPanel({
  valuation,
  reference,
  yearsRetrieved,
}: {
  valuation: Valuation;
  reference: Schemas["Reference"] | null;
  yearsRetrieved: number;
}) {
  const count = (status: Schemas["Check"]["status"]) => valuation.checks.filter((check) => check.status === status).length;
  const skipped = count("not evaluated");
  return (
    <Stack gap="lg">
      <SimpleGrid cols={{ base: 2, md: 4 }} spacing="md">
        <Stat label="Checks passed" value={String(count("pass"))} />
        <Stat label="Checks failed" value={String(count("fail"))} color={count("fail") ? "red" : undefined} />
        <Stat label="Not evaluated" value={String(skipped)} />
        <Stat label="Fiscal years retrieved" value={String(yearsRetrieved)} />
      </SimpleGrid>
      {skipped > 0 && <Caption>“Not evaluated” means the inputs a check needs were absent — it is not a pass.</Caption>}
      <Card p={0}>
        <Table.ScrollContainer minWidth={720}>
          <Table fz="sm" verticalSpacing={6} striped highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Check</Table.Th>
                <Table.Th>Status</Table.Th>
                <Table.Th>Severity</Table.Th>
                <Table.Th>Detail</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {valuation.checks.map((check) => (
                <Table.Tr key={check.name}>
                  <Table.Td style={{ whiteSpace: "nowrap" }}>{check.name}</Table.Td>
                  <Table.Td>
                    <Badge size="sm" variant="light" color={STATUS_COLOR[check.status]} style={{ minWidth: "max-content" }}>
                      {check.status}
                    </Badge>
                  </Table.Td>
                  <Table.Td>
                    <Text size="xs" c="dimmed">
                      {check.severity ?? ""}
                    </Text>
                  </Table.Td>
                  <Table.Td>
                    <Text size="xs">
                      <Md text={check.detail} />
                    </Text>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
      </Card>
      {reference && (
        <Accordion variant="separated" multiple>
          <Accordion.Item value="schema">
            <Accordion.Control>
              <Text size="sm" fw={500}>
                Resolved schema mappings
              </Text>
            </Accordion.Control>
            <Accordion.Panel>
              <Caption>
                Concept-to-column resolution against the columns Sharadar actually returned for this company. Nothing here
                is hardcoded at a call site.
              </Caption>
              {reference.ambiguous && (
                <Alert variant="light" color="yellow" p="xs" my="xs">
                  <Text size="xs">
                    Some concepts matched more than one column. The first alias in the preference order was used; the
                    alternates are listed.
                  </Text>
                </Alert>
              )}
              <FrameTable frame={reference.schema_map} minWidth={900} />
            </Accordion.Panel>
          </Accordion.Item>
          <Accordion.Item value="tolerances">
            <Accordion.Control>
              <Text size="sm" fw={500}>
                Tolerances in use
              </Text>
            </Accordion.Control>
            <Accordion.Panel>
              <FrameTable frame={reference.tolerances} minWidth={420} />
            </Accordion.Panel>
          </Accordion.Item>
        </Accordion>
      )}
    </Stack>
  );
}

// A terminal value that cannot be computed is an input to change, not absent data.
const INPUT_PROBLEM = "Terminal value cannot be computed";

/** Why there is no valuation: a refused company, or a run the model cannot build. */
export function BlockedCard({ blocked }: { blocked: Schemas["Blocked"] }) {
  const absent = blocked.title !== INPUT_PROBLEM;
  return (
    <Alert variant="light" color="red" icon={<IconAlertOctagon size={20} />} title={blocked.title}>
      <Stack gap="xs">
        <Text size="sm">
          <Md text={blocked.message} />
        </Text>
        {blocked.detail && (
          <Text size="xs" c="dimmed">
            {blocked.detail}
          </Text>
        )}
        {absent && (
        <Text size="xs">
          This is a <b>blocking</b> condition: the model cannot be computed, so no valuation is shown. It is not a
          judgement that the data looks wrong — it is that the data needed is absent.
        </Text>
        )}
      </Stack>
    </Alert>
  );
}


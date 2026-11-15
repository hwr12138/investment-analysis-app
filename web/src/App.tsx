import {
  ActionIcon,
  Alert,
  AppShell,
  Burger,
  Button,
  Group,
  NavLink,
  Stack,
  Text,
  Title,
  useComputedColorScheme,
  useMantineColorScheme,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router";

import { ApiError, api, unwrap } from "./api/client";
import { authEnabled, signOutUser, useUser } from "./auth";
import { MomentumPage } from "./pages/momentum/MomentumPage";

// Only built pages go in the navigation; the rest arrive as they are ported.
const PAGES = [{ path: "/momentum", label: "Momentum Analysis", icon: "🚀" }];

export function App() {
  const [opened, { toggle, close }] = useDisclosure();
  const { pathname } = useLocation();

  return (
    <AppShell
      header={{ height: 56 }}
      navbar={{ width: 240, breakpoint: "sm", collapsed: { mobile: !opened } }}
      padding="md"
    >
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between">
          <Group>
            <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
            <Title order={3}>Equiti</Title>
          </Group>
          <Group>
            <ColorSchemeToggle />
            {authEnabled && <SignedInAs />}
          </Group>
        </Group>
      </AppShell.Header>

      <AppShell.Navbar p="xs">
        {PAGES.map((page) => (
          <NavLink
            key={page.path}
            component={Link}
            to={page.path}
            label={page.label}
            leftSection={page.icon}
            active={pathname.startsWith(page.path)}
            onClick={close}
          />
        ))}
      </AppShell.Navbar>

      <AppShell.Main>
        <AccessCheck>
          <Routes>
            <Route path="/" element={<Navigate to="/momentum" replace />} />
            <Route path="/momentum/:ticker?" element={<MomentumPage />} />
            <Route path="*" element={<Text>No such page.</Text>} />
          </Routes>
        </AccessCheck>
      </AppShell.Main>
    </AppShell>
  );
}

/** Signing in proves who you are; the API's allowlist decides whether you may
 *  read anything. Ask it once, so a refusal reads as one clear message rather
 *  than as every request failing. */
function AccessCheck({ children }: { children: ReactNode }) {
  const me = useQuery({ queryKey: ["me"], queryFn: () => unwrap(api.GET("/api/me")) });
  if (me.error instanceof ApiError && me.error.status === 403) {
    return (
      <Alert color="red" title="No access">
        <Stack gap="xs" align="flex-start">
          <Text size="sm">{me.error.message}</Text>
          <Button size="xs" variant="default" onClick={() => void signOutUser()}>
            Sign in with another account
          </Button>
        </Stack>
      </Alert>
    );
  }
  return children;
}

function SignedInAs() {
  const user = useUser();
  if (!user) return null;
  return (
    <Group gap="xs">
      <Text size="sm" c="dimmed" visibleFrom="sm">
        {user.email}
      </Text>
      <Button size="xs" variant="default" onClick={() => void signOutUser()}>
        Sign out
      </Button>
    </Group>
  );
}

function ColorSchemeToggle() {
  const { setColorScheme } = useMantineColorScheme();
  const computed = useComputedColorScheme("dark");
  return (
    <ActionIcon
      variant="default"
      aria-label="Toggle colour scheme"
      onClick={() => setColorScheme(computed === "dark" ? "light" : "dark")}
    >
      {computed === "dark" ? "☀" : "☾"}
    </ActionIcon>
  );
}

import {
  ActionIcon,
  AppShell,
  Avatar,
  Burger,
  Button,
  Center,
  Group,
  Menu,
  NavLink,
  Paper,
  Stack,
  Text,
  ThemeIcon,
  Title,
  Tooltip,
  UnstyledButton,
  useComputedColorScheme,
  useMantineColorScheme,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { IconLockAccess, IconLogout, IconMoonStars, IconSun, IconTrendingUp } from "@tabler/icons-react";
import { useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router";

import { ApiError, api, unwrap } from "./api/client";
import { authEnabled, signOutUser, useUser } from "./auth";
import { Brand } from "./components/Brand";
import { MomentumPage } from "./pages/momentum/MomentumPage";

// Only built pages go in the navigation; the rest arrive as they are ported.
const PAGES = [{ path: "/momentum", label: "Momentum", icon: IconTrendingUp }];

export function App() {
  const [opened, { toggle, close }] = useDisclosure();
  const { pathname } = useLocation();

  return (
    <AppShell
      header={{ height: 60 }}
      navbar={{ width: 232, breakpoint: "sm", collapsed: { mobile: !opened } }}
      padding="lg"
    >
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between">
          <Group gap="sm">
            <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
            <Brand />
          </Group>
          <Group gap="xs">
            <ColorSchemeToggle />
            {authEnabled && <UserMenu />}
          </Group>
        </Group>
      </AppShell.Header>

      <AppShell.Navbar p="md">
        <Text size="xs" fw={700} c="dimmed" tt="uppercase" px="sm" mb="xs">
          Analysis
        </Text>
        {PAGES.map(({ path, label, icon: Icon }) => (
          <NavLink
            key={path}
            className="nav-link"
            component={Link}
            to={path}
            label={label}
            leftSection={<Icon size={18} stroke={1.6} />}
            active={pathname.startsWith(path)}
            variant="light"
            onClick={close}
          />
        ))}
      </AppShell.Navbar>

      <AppShell.Main className="app-main">
        <AccessCheck>
          <Routes>
            <Route path="/" element={<Navigate to="/momentum" replace />} />
            <Route path="/momentum/:ticker?" element={<MomentumPage />} />
            <Route path="*" element={<Text c="dimmed">No such page.</Text>} />
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
  const user = useUser();
  const me = useQuery({ queryKey: ["me"], queryFn: () => unwrap(api.GET("/api/me")) });
  if (!(me.error instanceof ApiError && me.error.status === 403)) return children;
  return (
    <Center mih="60vh">
      <Paper withBorder radius="lg" p="xl" maw={420}>
        <Stack align="center" gap="md" ta="center">
          <ThemeIcon size={48} radius="xl" variant="light" color="red">
            <IconLockAccess size={26} stroke={1.6} />
          </ThemeIcon>
          <Title order={3}>No access</Title>
          <Text size="sm" c="dimmed">
            {me.error.message}
            {user?.email && ` You're signed in as ${user.email}.`}
          </Text>
          <Button variant="default" onClick={() => void signOutUser()}>
            Use another account
          </Button>
        </Stack>
      </Paper>
    </Center>
  );
}

function UserMenu() {
  const user = useUser();
  if (!user) return null;
  return (
    <Menu position="bottom-end" width={240} shadow="md">
      <Menu.Target>
        <UnstyledButton aria-label="Account">
          <Avatar src={user.photoURL} alt={user.email ?? ""} radius="xl" size={32} color="indigo">
            {user.email?.[0]?.toUpperCase()}
          </Avatar>
        </UnstyledButton>
      </Menu.Target>
      <Menu.Dropdown>
        <Menu.Label>{user.email}</Menu.Label>
        <Menu.Item leftSection={<IconLogout size={16} />} onClick={() => void signOutUser()}>
          Sign out
        </Menu.Item>
      </Menu.Dropdown>
    </Menu>
  );
}

function ColorSchemeToggle() {
  const { setColorScheme } = useMantineColorScheme();
  const dark = useComputedColorScheme("dark") === "dark";
  return (
    <Tooltip label={dark ? "Light mode" : "Dark mode"}>
      <ActionIcon
        variant="default"
        size="lg"
        aria-label="Toggle colour scheme"
        onClick={() => setColorScheme(dark ? "light" : "dark")}
      >
        {dark ? <IconSun size={18} stroke={1.6} /> : <IconMoonStars size={18} stroke={1.6} />}
      </ActionIcon>
    </Tooltip>
  );
}

// Firebase sign-in, switched on by the VITE_FIREBASE_* variables (.env.example).
//
// The API checks every token itself (app/api/auth.py) — this file only obtains
// one. Without the variables the app calls the API with no token, which an API
// started with EQUITI_AUTH=off accepts locally and Cloud Run never does.
import { Button, Center, Loader, Stack, Text, Title } from "@mantine/core";
import { initializeApp } from "firebase/app";
import {
  GoogleAuthProvider,
  getAuth,
  onAuthStateChanged,
  signInWithPopup,
  signOut,
  type User,
} from "firebase/auth";
import { useEffect, useState, type ReactNode } from "react";

const env = import.meta.env;

export const authEnabled = Boolean(env.VITE_FIREBASE_API_KEY);

const auth = authEnabled
  ? getAuth(
      initializeApp({
        apiKey: env.VITE_FIREBASE_API_KEY,
        authDomain: env.VITE_FIREBASE_AUTH_DOMAIN,
        projectId: env.VITE_FIREBASE_PROJECT_ID,
        appId: env.VITE_FIREBASE_APP_ID,
      }),
    )
  : null;

/** The signed-in user's ID token (refreshed by Firebase when near expiry), or null. */
export async function idToken(): Promise<string | null> {
  return auth?.currentUser ? auth.currentUser.getIdToken() : null;
}

export function signOutUser(): Promise<void> {
  return auth ? signOut(auth) : Promise.resolve();
}

export function useUser(): User | null | undefined {
  const [user, setUser] = useState<User | null | undefined>(auth ? undefined : null);
  useEffect(() => (auth ? onAuthStateChanged(auth, setUser) : undefined), []);
  return user;
}

/** Renders its children only once someone is signed in (or sign-in is off). */
export function AuthGate({ children }: { children: ReactNode }) {
  const user = useUser();
  const [error, setError] = useState<string | null>(null);

  if (!auth) return children;
  if (user === undefined) {
    return (
      <Center h="100vh">
        <Loader />
      </Center>
    );
  }
  if (user === null) {
    return (
      <Center h="100vh">
        <Stack align="center">
          <Title order={2}>Equiti</Title>
          <Button
            onClick={() =>
              signInWithPopup(auth, new GoogleAuthProvider()).catch((e: Error) =>
                setError(e.message),
              )
            }
          >
            Sign in with Google
          </Button>
          {error && (
            <Text c="red" size="sm">
              {error}
            </Text>
          )}
        </Stack>
      </Center>
    );
  }
  return children;
}

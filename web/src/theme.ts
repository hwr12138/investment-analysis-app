import { Card, createTheme } from "@mantine/core";

export const theme = createTheme({
  primaryColor: "indigo",
  defaultRadius: "md",
  headings: { fontWeight: "700" },
  components: {
    Card: Card.extend({ defaultProps: { withBorder: true, padding: "lg", radius: "md" } }),
  },
});

/** Tones a reading can carry, and the colour each is drawn in everywhere. */
export type Tone = "bullish" | "bearish" | "neutral";
export const TONE_COLOR: Record<Tone, string> = {
  bullish: "teal",
  bearish: "red",
  neutral: "gray",
};

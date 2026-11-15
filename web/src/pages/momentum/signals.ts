// Reading the scanner and divergence strings for display.
//
// The logic layer returns sentences with an emoji in front ("📈 Bullish Momentum
// (MACD > Signal)"), written for the Streamlit page this app replaced. Here they
// become a tone, a headline and a detail. Tone is read from the words
// "Bull"/"Bear"; anything else, including a wording this does not recognise, is
// shown as neutral grey rather than guessed at. Returning a structured reading
// from the logic layer would remove the parsing; nothing else reads the sentences.
import type { Tone } from "../../theme";

export type Reading = {
  tone: Tone;
  headline: string;
  detail: string | null;
  /** A crossover that has just happened, or a pattern that is active now. */
  fresh: boolean;
};

export function read(text: string): Reading {
  const cleaned = text.replace(/^[^\p{L}]+/u, "").replace(/!+$/, "").trim();
  const parts = cleaned.match(/^(.*?)\s*\((.+)\)$/);
  return {
    tone: /\bbull/i.test(cleaned) ? "bullish" : /\bbear/i.test(cleaned) ? "bearish" : "neutral",
    headline: parts?.[1] ?? cleaned,
    detail: parts?.[2] ?? null,
    fresh: /just occurred|\bactive\b/i.test(cleaned),
  };
}

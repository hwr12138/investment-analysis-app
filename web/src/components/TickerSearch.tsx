// A ticker box that submits on Enter. Keyed on the ticker by its caller, so the
// draft resets whenever the page moves to another company.
import { TextInput } from "@mantine/core";
import { IconSearch } from "@tabler/icons-react";
import { useState } from "react";

export function TickerSearch({ initial, onSubmit }: { initial: string; onSubmit: (ticker: string) => void }) {
  const [draft, setDraft] = useState(initial);
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        const next = draft.trim().toUpperCase();
        if (next) onSubmit(next);
      }}
    >
      <TextInput
        aria-label="Ticker"
        placeholder="Ticker"
        value={draft}
        onChange={(event) => setDraft(event.currentTarget.value)}
        leftSection={<IconSearch size={16} />}
        w={150}
        styles={{ input: { textTransform: "uppercase" } }}
      />
    </form>
  );
}

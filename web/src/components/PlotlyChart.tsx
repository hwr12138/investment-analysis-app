// plotly.js drawn into a div, loaded on first use so the shell paints before
// the ~1.3 MB library arrives. A thin wrapper rather than react-plotly.js,
// which has not had a release in years.
import { Box, Center, Loader } from "@mantine/core";
import type { Config, Data, Layout } from "plotly.js";
import { useEffect, useRef, useState } from "react";

type Plotly = typeof import("plotly.js-finance-dist-min").default;

let loading: Promise<Plotly> | null = null;
let loaded: Plotly | null = null;

function loadPlotly(): Promise<Plotly> {
  loading ??= import("plotly.js-finance-dist-min").then(({ default: module }) => {
    loaded = module;
    return module;
  });
  return loading;
}

type Props = {
  data: Data[];
  layout: Partial<Layout>;
  config?: Partial<Config>;
  height: number;
};

export function PlotlyChart({ data, layout, config, height }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    let live = true;
    void loadPlotly().then((plotly) => {
      if (!live) return;
      void plotly.react(node, data, { ...layout, height }, {
        responsive: true,
        displaylogo: false,
        ...config,
      });
      setReady(true);
    });
    return () => {
      live = false;
    };
  }, [data, layout, config, height]);

  // Purged synchronously on unmount. Purging through a fresh import() raced
  // StrictMode's mount–unmount–mount in development and wiped the redraw.
  useEffect(() => {
    const node = ref.current;
    return () => {
      if (node && loaded) loaded.purge(node);
    };
  }, []);

  return (
    <Box pos="relative" h={height}>
      {!ready && (
        <Center pos="absolute" inset={0}>
          <Loader />
        </Center>
      )}
      <div ref={ref} style={{ width: "100%" }} />
    </Box>
  );
}

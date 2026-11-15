// The three-panel momentum chart — price, MACD, RSI on one shared date axis —
// built from the API's columns and drawn in the Mantine theme's colours.
import type { MantineTheme } from "@mantine/core";
import type { Annotations, Data, Layout, PlotData, Shape } from "plotly.js";

import type { Schemas } from "../../api/client";

export type Palette = {
  up: string;
  down: string;
  macd: string;
  signal: string;
  rsi: string;
  trend: string;
  grid: string;
  text: string;
  muted: string;
  font: string;
};

export function palette(theme: MantineTheme, dark: boolean): Palette {
  const c = theme.colors;
  return {
    up: c.teal[dark ? 5 : 6],
    down: c.red[dark ? 5 : 6],
    macd: c.blue[dark ? 4 : 6],
    signal: c.orange[dark ? 4 : 6],
    rsi: c.violet[dark ? 4 : 6],
    trend: c.yellow[dark ? 4 : 6],
    grid: dark ? c.dark[5] : c.gray[2],
    text: dark ? c.dark[1] : c.gray[7],
    muted: dark ? c.dark[2] : c.gray[6],
    font: theme.fontFamily ?? "sans-serif",
  };
}

/** A hex colour at a given opacity, for fills behind the lines. */
function alpha(hex: string, opacity: number): string {
  const n = Number.parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${opacity})`;
}

// Panel domains on the shared paper, top to bottom.
const PRICE: [number, number] = [0.48, 1];
const MACD: [number, number] = [0.25, 0.43];
const RSI: [number, number] = [0, 0.21];

export function momentumTraces(
  window: Schemas["MomentumWindow"],
  divergence: Schemas["Divergence"] | null,
  p: Palette,
): Data[] {
  const b = window.bars;
  const x = b.sessions;
  const traces: Data[] = [
    {
      type: "candlestick",
      name: "Price",
      x,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
      increasing: { line: { color: p.up, width: 1 }, fillcolor: p.up },
      decreasing: { line: { color: p.down, width: 1 }, fillcolor: p.down },
      // plotly.js draws the candle bodies from `fillcolor`; its types omit it.
    } as Data,
    {
      type: "bar",
      name: "Histogram",
      x,
      y: b.histogram,
      yaxis: "y2",
      marker: { color: b.histogram.map((v) => alpha(v >= 0 ? p.up : p.down, 0.45)) },
    },
    { type: "scatter", mode: "lines", name: "MACD", x, y: b.macd, yaxis: "y2", line: { color: p.macd, width: 1.6 } },
    { type: "scatter", mode: "lines", name: "Signal", x, y: b.signal, yaxis: "y2", line: { color: p.signal, width: 1.4 } },
    { type: "scatter", mode: "lines", name: "RSI", x, y: b.rsi, yaxis: "y3", line: { color: p.rsi, width: 1.6 } },
  ];
  if (divergence) {
    const { first: a, second: z } = divergence;
    const line: Partial<PlotData> = {
      type: "scatter",
      mode: "lines+markers",
      x: [a.session, z.session],
      line: { color: p.trend, width: 2, dash: "dot" },
      marker: { color: p.trend, size: 7 },
      hoverinfo: "skip",
    };
    traces.push(
      { ...line, name: "Trend", y: [a.close, z.close] },
      { ...line, name: "Trend", y: [a.macd, z.macd], yaxis: "y2" },
      { ...line, name: "Trend", y: [a.rsi, z.rsi], yaxis: "y3" },
    );
  }
  return traces;
}

/** A panel's title, set in the gap above it so it never sits on the data. */
function panelLabel(text: string, top: number, p: Palette): Partial<Annotations> {
  return {
    text,
    xref: "paper",
    yref: "paper",
    x: 0,
    y: top,
    xanchor: "left",
    yanchor: "bottom",
    showarrow: false,
    font: { size: 11, color: p.muted },
  };
}

function rsiLine(level: number, p: Palette): Partial<Shape> {
  return {
    type: "line", xref: "paper", x0: 0, x1: 1, yref: "y3", y0: level, y1: level,
    line: { color: alpha(p.rsi, 0.5), width: 1, dash: "dot" },
  };
}

export function momentumLayout(p: Palette): Partial<Layout> {
  const axis = { gridcolor: p.grid, zeroline: false, side: "right", tickfont: { size: 10 } } as const;
  return {
    margin: { t: 24, r: 52, b: 28, l: 8 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: p.font, color: p.text, size: 11 },
    showlegend: false,
    hovermode: "x unified",
    hoverlabel: { font: { family: p.font, size: 11 } },
    bargap: 0.25,
    xaxis: {
      anchor: "y3",
      showgrid: false,
      rangeslider: { visible: false },
      rangebreaks: [{ bounds: ["sat", "mon"] }],
      showspikes: true,
      spikemode: "across",
      spikethickness: 1,
      spikedash: "dot",
      spikecolor: p.muted,
      tickfont: { size: 10 },
    },
    yaxis: { ...axis, domain: PRICE, tickprefix: "$" },
    yaxis2: { ...axis, domain: MACD, zeroline: true, zerolinecolor: p.grid },
    yaxis3: { ...axis, domain: RSI, range: [0, 100], tickvals: [30, 50, 70] },
    shapes: [
      {
        type: "rect", xref: "paper", x0: 0, x1: 1, yref: "y3", y0: 30, y1: 70,
        fillcolor: alpha(p.rsi, 0.07), line: { width: 0 }, layer: "below",
      },
      rsiLine(30, p),
      rsiLine(70, p),
    ],
    annotations: [
      panelLabel("Price", PRICE[1], p),
      panelLabel(
        `MACD 12 · 26 · 9   <span style="color:${p.macd}">━ MACD</span>   <span style="color:${p.signal}">━ Signal</span>`,
        MACD[1],
        p,
      ),
      panelLabel("RSI 14", RSI[1], p),
    ],
  };
}

// The three-panel momentum chart — price, MACD, RSI on one shared date axis —
// built from the API's columns. The Streamlit page draws the same figure with
// plotly's Python make_subplots; this is its layout spelled out by hand.
import type { Data, Layout, Shape } from "plotly.js";

import type { Schemas } from "../../api/client";

const ANCHOR_LINE = { color: "#fab005", width: 2, dash: "dot" } as const;

export function momentumTraces(
  window: Schemas["MomentumWindow"],
  divergence: Schemas["Divergence"] | null,
): Data[] {
  const b = window.bars;
  const x = b.sessions;
  const traces: Data[] = [
    { type: "candlestick", x, open: b.open, high: b.high, low: b.low, close: b.close, name: "Price" },
    { type: "scatter", mode: "lines", x, y: b.macd, name: "MACD", yaxis: "y2", line: { color: "#228be6", width: 1.5 } },
    { type: "scatter", mode: "lines", x, y: b.signal, name: "Signal", yaxis: "y2", line: { color: "#fd7e14", width: 1.5 } },
    { type: "bar", x, y: b.histogram, name: "Histogram", yaxis: "y2", marker: { color: "gray" } },
    { type: "scatter", mode: "lines", x, y: b.rsi, name: "RSI", yaxis: "y3", line: { color: "#be4bdb", width: 1.5 } },
  ];
  if (divergence) {
    const { first: a, second: z } = divergence;
    const xs = [a.session, z.session];
    const line = { type: "scatter", mode: "lines+markers", x: xs, line: ANCHOR_LINE } as const;
    traces.push(
      { ...line, y: [a.close, z.close], name: "Price Trend" },
      { ...line, y: [a.macd, z.macd], yaxis: "y2", showlegend: false },
      { ...line, y: [a.rsi, z.rsi], yaxis: "y3", showlegend: false },
    );
  }
  return traces;
}

function rsiBand(level: number, color: string): Partial<Shape> {
  return {
    type: "line", xref: "paper", x0: 0, x1: 1, yref: "y3", y0: level, y1: level,
    line: { color, dash: "dash", width: 1 },
  };
}

export function momentumLayout(dark: boolean): Partial<Layout> {
  const grid = dark ? "#373a40" : "#e9ecef";
  return {
    margin: { t: 30, b: 30, l: 50, r: 10 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: dark ? "#c9c9c9" : "#212529" },
    legend: { orientation: "h", y: 1.05 },
    hovermode: "x unified",
    xaxis: {
      anchor: "y3",
      gridcolor: grid,
      rangeslider: { visible: false },
      rangebreaks: [{ bounds: ["sat", "mon"] }],
    },
    yaxis: { domain: [0.52, 1], gridcolor: grid },
    yaxis2: { domain: [0.27, 0.48], gridcolor: grid, zerolinecolor: grid },
    yaxis3: { domain: [0, 0.23], gridcolor: grid, range: [0, 100] },
    shapes: [rsiBand(70, "#fa5252"), rsiBand(30, "#40c057")],
  };
}

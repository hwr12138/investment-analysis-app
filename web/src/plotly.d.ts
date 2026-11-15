// The finance bundle (candlestick, scatter, bar) is plotly.js with fewer trace
// types compiled in; its API is the full library's.
declare module "plotly.js-finance-dist-min" {
  import * as Plotly from "plotly.js";
  export default Plotly;
}

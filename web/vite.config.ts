import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the API runs beside Vite (uvicorn, port 8600 by default) and
// is reached through this proxy on the same origin — the same shape Firebase
// Hosting gives production, where /api/** is rewritten to Cloud Run.
const apiPort = process.env.API_PORT ?? "8600";

export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.PORT ?? 5173),
    proxy: { "/api": `http://127.0.0.1:${apiPort}` },
  },
});

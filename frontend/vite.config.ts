import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev proxy for /api/*. A local API serves its routes at the root, so the
// prefix is stripped; set API_PROXY_TARGET (e.g. the live demo host) to
// forward to a deployment that already lives behind an /api/ prefix.
const target = process.env.API_PROXY_TARGET;

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": target
        ? { target, changeOrigin: true }
        : { target: "http://localhost:8000", rewrite: (path) => path.replace(/^\/api/, "") },
    },
  },
});

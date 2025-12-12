import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import { resolve } from "path";

export default defineConfig({
  plugins: [react()],
  root: ".",
  build: {
    outDir: "../frontend/static/frontend",
    emptyOutDir: false,
    rollupOptions: {
      input: resolve(__dirname, "index.html"),
      output: {
        // JS entry for the app
        entryFileNames: "assets/index.js",
        // Any JS chunks (if code-splitting happens)
        chunkFileNames: "assets/[name].js",
        // CSS and other assets – no hashes, stable names
        assetFileNames: "assets/[name][extname]",
      },
    },
  },

  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: process.env.VITE_API_BASE_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/admin": {
        target: process.env.VITE_API_BASE_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/static": {
        target: process.env.VITE_API_BASE_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: "node",
  },
});

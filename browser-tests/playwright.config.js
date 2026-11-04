import { defineConfig } from "@playwright/test";
import { fileURLToPath } from "node:url";

const port = Number(process.env.PANELCHEMY_TEST_PORT || 8766);
if (!Number.isInteger(port) || port < 1 || port > 65535) {
  throw new Error("PANELCHEMY_TEST_PORT must be an integer from 1 to 65535");
}
const baseURL = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: ".",
  testMatch: "*.spec.js",
  fullyParallel: false,
  workers: 1,
  timeout: 30_000,
  reporter: "list",
  outputDir: process.env.PANELCHEMY_TEST_OUTPUT || "test-results",
  use: { baseURL, browserName: "chromium", headless: true },
  webServer: {
    command: `uv run --locked panelchemy serve --port ${port}`,
    cwd: fileURLToPath(new URL("..", import.meta.url)),
    url: `${baseURL}/api/health`,
    reuseExistingServer: false,
    timeout: 30_000,
  },
});

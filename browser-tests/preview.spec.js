import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";

// Generated synthetic 12x16 panel: enclosed black rim, white mark, orange block.
const panel = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAwAAAAQCAIAAACtAwlQAAAAQElEQVR4nGP8//8/A0FAUNH///+ZCBvDwMCC1UhGRkZkRUSZNGQVIYIAzdvYFX1dpowmxx11l9puIiqCGYlJKgDAUxgSqwmtBwAAAABJRU5ErkJggg==", "base64");
const upload = { name: "synthetic-panel.png", mimeType: "image/png", buffer: panel };

async function generate(page) {
  await page.getByRole("button", { name: "Generate previews" }).click();
  await expect(page.getByRole("button", { name: "Download outputs" })).toBeEnabled();
  await expect(page.locator("#normalized-preview")).toHaveJSProperty("naturalWidth", 12);
}

test("upload, adjust, inspect, download, invalid input and recovery", async ({ page }, info) => {
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/");
  await page.locator("#source-file").setInputFiles(upload);
  await generate(page);
  await expect(page.locator("#record-source")).toHaveText("synthetic-panel.png");
  await page.getByLabel("Mask preview").selectOption("support-orange");
  await expect(page.locator("#mask-preview")).toHaveJSProperty("naturalHeight", 16);

  await page.getByLabel("Neutral tolerance").fill("30");
  await expect(page.getByRole("button", { name: "Download outputs" })).toBeDisabled();
  await generate(page);
  await expect(page.locator("#manifest-json")).toContainText('"neutral_tolerance": 30');
  await page.screenshot({ path: info.outputPath("workspace-desktop.png"), fullPage: true });

  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download outputs" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("panelchemy-separation.zip");
  const path = info.outputPath("outputs.zip");
  await download.saveAs(path);
  const bytes = await readFile(path);
  expect(bytes.subarray(0, 4).toString("hex")).toBe("504b0304");
  expect(bytes.includes(Buffer.from("manifest.json"))).toBeTruthy();
  expect(bytes.includes(Buffer.from("config.json"))).toBeTruthy();

  await page.locator("#source-file").setInputFiles({ name: "invalid.png", mimeType: "image/png", buffer: Buffer.from("invalid image") });
  await page.getByRole("button", { name: "Generate previews" }).click();
  await expect(page.getByRole("alert")).toBeVisible();
  await expect(page.getByRole("button", { name: "Download outputs" })).toBeDisabled();
  await page.locator("#source-file").setInputFiles(upload);
  await generate(page);
  await expect(page.getByRole("alert")).toBeHidden();
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  await page.screenshot({ path: info.outputPath("workspace-mobile.png"), fullPage: true });
  expect(errors).toEqual([]);
});

test("a failed repeat conversion clears old ready state", async ({ page }) => {
  await page.goto("/");
  await page.locator("#source-file").setInputFiles(upload);
  await generate(page);
  await page.route("**/api/separate", route => route.fulfill({ status: 503, contentType: "application/json", body: "null" }));
  await page.getByRole("button", { name: "Generate previews" }).click();
  await expect(page.getByRole("alert")).toContainText("could not be processed");
  await expect(page.locator("#result-badge")).toHaveText("Not generated");
  await expect(page.locator("#normalized-preview")).toBeHidden();
  await expect(page.getByRole("button", { name: "Download outputs" })).toBeDisabled();
});

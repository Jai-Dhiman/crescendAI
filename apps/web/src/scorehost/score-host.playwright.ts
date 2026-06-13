import { test, expect } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

test("ScoreHost facade is defined after bundle loads", async ({ page }) => {
  const indexHtml = path.resolve(
    __dirname,
    "../../dist-scorehost/index.html",
  );
  await page.goto(`file://${indexHtml}`);
  await page.waitForFunction(() => typeof (window as any).ScoreHost !== "undefined", { timeout: 5000 });
  const defined = await page.evaluate(() => typeof (window as any).ScoreHost);
  expect(defined).toBe("object");
});

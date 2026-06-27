// Verify a LilyPond-engraved Mutopia .svg displays through the REAL scorehost
// worker (the SVG branch in score-worker.loadPiece, which bypasses Verovio).
// Opt-in: set SVG_FILE to a rendered .svg; skips cleanly otherwise.
import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const SVG_FILE = process.env.SVG_FILE ?? "";
const SHOTS = process.env.SHOTS_DIR ?? "";
const have = SVG_FILE !== "" && fs.existsSync(SVG_FILE);

test("Mutopia LilyPond .svg displays via scorehost static tier", async ({ page }) => {
  test.skip(!have, "set SVG_FILE to a rendered Mutopia .svg to run");
  const pid = "mutopia.test.piece";
  await page.route("**/api/scores/*/data", async (route) => {
    await route.fulfill({ status: 200, contentType: "image/svg+xml", body: fs.readFileSync(SVG_FILE) });
  });
  await page.goto("/");
  await page.waitForFunction(() => typeof (window as any).ScoreHost !== "undefined", { timeout: 10000 });
  await page.evaluate(() => { (window as any).__SCOREHOST_API_BASE = "http://localhost:5173"; });

  const loaded = await page.evaluate(async (id) => await (window as any).ScoreHost.load(id), pid);
  expect(loaded).toEqual({ ok: true });

  const artifact = JSON.stringify({ type: "score_highlight", config: { pieceId: pid, highlights: [] } });
  await page.evaluate(async (j) => { await (window as any).ScoreHost.showArtifact(j); }, artifact);
  await page.waitForSelector("#scorehost-container svg", { timeout: 15000 });

  const stats = await page.evaluate(() => {
    const c = document.getElementById("scorehost-container")!;
    return { svgCount: c.querySelectorAll("svg").length, pathCount: c.querySelectorAll("svg path").length };
  });
  console.log(`Mutopia SVG display: svg=${stats.svgCount} paths=${stats.pathCount}`);
  expect(stats.svgCount).toBeGreaterThanOrEqual(1);
  expect(stats.pathCount).toBeGreaterThan(50); // LilyPond glyphs are inline <path>s
  if (SHOTS) {
    fs.mkdirSync(SHOTS, { recursive: true });
    await page.locator("#scorehost-container").screenshot({ path: path.join(SHOTS, "mutopia_svg_display.png") });
  }
});

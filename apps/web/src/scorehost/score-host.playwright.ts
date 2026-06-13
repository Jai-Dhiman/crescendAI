import { test, expect } from "@playwright/test";

test("ScoreHost facade is defined after bundle loads", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => typeof (window as any).ScoreHost !== "undefined", { timeout: 5000 });
  const defined = await page.evaluate(() => typeof (window as any).ScoreHost);
  expect(defined).toBe("object");
});

test("ScoreHost.load + showArtifact(score_highlight) engraves note glyphs", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => typeof (window as any).ScoreHost !== "undefined", { timeout: 5000 });

  const loadResult = await page.evaluate(async () => {
    return await (window as any).ScoreHost.load("czerny-op299-no1");
  });
  expect(loadResult).toEqual({ ok: true });

  const artifactJson = JSON.stringify({
    type: "score_highlight",
    config: {
      pieceId: "czerny-op299-no1",
      highlights: [{ bars: [1, 4], dimension: "dynamics", annotation: "forte" }],
    },
  });
  await page.evaluate(async (json: string) => {
    await (window as any).ScoreHost.showArtifact(json);
  }, artifactJson);

  await page.waitForSelector("svg use", { timeout: 15000 });
  const useCount = await page.evaluate(() => document.querySelectorAll("svg use").length);
  expect(useCount).toBeGreaterThan(0);

  // Clip renders exactly ONE svg element (not the full multi-page score)
  const svgCount = await page.evaluate(() => document.querySelectorAll("#scorehost-container > div > svg").length);
  expect(svgCount).toBe(1);
});

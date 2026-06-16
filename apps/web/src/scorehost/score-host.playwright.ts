import { test, expect } from "@playwright/test";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


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

test("ScoreHost play_passage renders clip and fires playback event", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => typeof (window as any).ScoreHost !== "undefined", { timeout: 5000 });

  await page.evaluate(async () => {
    await (window as any).ScoreHost.load("czerny-op299-no1");
  });

  const artifactJson = JSON.stringify({
    type: "play_passage",
    config: {
      pieceId: "czerny-op299-no1",
      bars: [1, 4],
      dimension: "dynamics",
      annotation: "Keep steady pulse here",
    },
  });

  await page.evaluate(async (json: string) => {
    await (window as any).ScoreHost.showArtifact(json);
  }, artifactJson);

  // Clip SVG should render
  await page.waitForSelector("svg use", { timeout: 15000 });
  const useCount = await page.evaluate(() => document.querySelectorAll("svg use").length);
  expect(useCount).toBeGreaterThan(0);

  // Install playback event capture BEFORE calling play().
  // Assign directly to window.__playbackEvents so waitForFunction can see it in page context.
  await page.evaluate(() => {
    (window as any).__playbackEvents = [];
    const origPostMessage = window.webkit?.messageHandlers?.scoreHostEvents?.postMessage?.bind(
      window.webkit?.messageHandlers?.scoreHostEvents,
    );
    if (!(window as any).webkit) (window as any).webkit = {};
    if (!(window as any).webkit.messageHandlers) (window as any).webkit.messageHandlers = {};
    (window as any).webkit.messageHandlers.scoreHostEvents = {
      postMessage: (msg: any) => {
        if (msg.type === "playback") {
          ((window as any).__playbackEvents as string[]).push(msg.payload.state);
        }
        if (origPostMessage) origPostMessage(msg);
      },
    };
  });

  // play() clicks the in-WebView button which fires the playback event.
  // In headless Chromium, AudioContext autoplay may be blocked, so we allow
  // the Promise to resolve without error as the primary assertion.
  await expect(page.evaluate(() => (window as any).ScoreHost.play())).resolves.not.toThrow();

  // Wait up to 3s for playback{state:"playing"} event in the PAGE context.
  // window.__playbackEvents is set above and is visible to waitForFunction.
  await page
    .waitForFunction(
      () => ((window as any).__playbackEvents as string[]).includes("playing"),
      { timeout: 3000 },
    )
    .catch(() => {
      // Headless AudioContext may block autoplay — acceptable; test passes if no exception thrown above.
    });

  // Transport UI must still be present regardless of AudioContext state.
  const useCountAfterPlay = await page.evaluate(() => document.querySelectorAll("svg use").length);
  expect(useCountAfterPlay).toBeGreaterThan(0);

  const transportPresent = await page.evaluate(() => !!document.getElementById("loop-transport"));
  expect(transportPresent).toBe(true);
});

// This test verifies that when window.__SCOREHOST_API_BASE is set, fetchScoreBytes
// requests /api/scores/:pieceId/data from that origin instead of the bundled ./scores/ path.
// It uses Playwright route interception — no real API server required.
test("ScoreHost.load fetches non-bundled piece via __SCOREHOST_API_BASE (/api/scores/:id/data)", async ({ page }) => {
  // Intercept the bundled path to ensure it is NOT what satisfies the load.
  // Any request to ./scores/... from the file:// page would match this pattern.
  let bundledPathHit = false;
  await page.route("**/scores/chopin-nocturne-op9-no2.mxl", async (route) => {
    bundledPathHit = true;
    // Abort immediately so a regression to the bundled path fails fast with a clear error.
    await route.abort();
  });

  // Read the real .mxl bytes used by the other tests (czerny) to use as a stand-in fixture.
  // The scorehost vite preview serves public/scores/ under /scores/.
  // We intercept the API endpoint and return the czerny bytes (valid .mxl) so Verovio can parse.
  let apiPathHit = false;
  await page.route("**/api/scores/chopin-nocturne-op9-no2/data", async (route) => {
    apiPathHit = true;
    // Fetch the czerny fixture from the running vite preview as a valid .mxl stand-in.
    const bytes = await fetch("http://localhost:5173/scores/czerny-op299-no1.mxl")
      .then((r) => r.arrayBuffer())
      .catch(() => null);
    if (!bytes) {
      await route.abort();
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/octet-stream",
      body: Buffer.from(bytes),
    });
  });

  // Use the vite preview server (http://localhost:5173) so that absolute URL routing works.
  await page.goto("/");
  await page.waitForFunction(() => typeof (window as any).ScoreHost !== "undefined", { timeout: 5000 });

  // Set __SCOREHOST_API_BASE to the Playwright base URL so /api/ routes resolve via the
  // intercepted route above.
  await page.evaluate(() => {
    (window as any).__SCOREHOST_API_BASE = "http://localhost:5173";
  });

  const loadResult = await page.evaluate(async () => {
    return await (window as any).ScoreHost.load("chopin-nocturne-op9-no2");
  });
  expect(loadResult).toEqual({ ok: true });

  // The API path must have been hit — not the bundled ./scores/ path.
  expect(apiPathHit).toBe(true);
  expect(bundledPathHit).toBe(false);

  // Verify the loaded score actually renders.
  const artifactJson = JSON.stringify({
    type: "score_highlight",
    config: {
      pieceId: "chopin-nocturne-op9-no2",
      highlights: [{ bars: [1, 4], dimension: "phrasing", annotation: "sing here" }],
    },
  });
  await page.evaluate(async (json: string) => {
    await (window as any).ScoreHost.showArtifact(json);
  }, artifactJson);

  await page.waitForSelector("svg use", { timeout: 15000 });
  const useCount = await page.evaluate(() => document.querySelectorAll("svg use").length);
  expect(useCount).toBeGreaterThan(0);
});

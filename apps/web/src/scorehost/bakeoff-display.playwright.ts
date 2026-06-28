// Bake-off DISPLAY verification: render arbitrary candidate engravings
// (.mxl / .musicxml / .svg) through the REAL scorehost Verovio worker -- the
// exact engine the web ScorePanel and iOS WKWebView use. This is how we judge
// notation readability and catch the partitura-segfaults-Verovio failure mode
// that a python verovio.loadData() pass cannot see.
//
// Opt-in (skips cleanly otherwise): set BAKEOFF_SAMPLE to a JSON array of
//   { pid, file, label }  (file = absolute path to the engraving asset)
// and SHOTS_DIR for screenshots. Content-type is irrelevant -- score-worker
// detects format by magic bytes (ZIP -> mxl interactive; <svg> -> static).
import { test } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const SAMPLE = process.env.BAKEOFF_SAMPLE ?? "";
const SHOTS = process.env.SHOTS_DIR ?? "";
const haveData = SAMPLE !== "" && fs.existsSync(SAMPLE);

interface Row { pid: string; file: string; label: string }
const sample: Row[] = haveData ? JSON.parse(fs.readFileSync(SAMPLE, "utf8")) : [];
const byPid = new Map(sample.map((r) => [r.pid, r]));

test("BAKEOFF: candidate engravings render in real scorehost Verovio worker", async ({ page }) => {
  test.skip(!haveData, "set BAKEOFF_SAMPLE (+ SHOTS_DIR) to run the engine bake-off sweep");
  if (SHOTS) fs.mkdirSync(SHOTS, { recursive: true });
  const out = fs.createWriteStream(path.join(SHOTS || ".", "bakeoff_display_results.jsonl"), { flags: "w" });

  await page.route("**/api/scores/*/data", async (route) => {
    const url = new URL(route.request().url());
    const m = url.pathname.match(/\/api\/scores\/(.+)\/data$/);
    const pid = m ? decodeURIComponent(m[1]) : "";
    const row = byPid.get(pid);
    if (!row || !fs.existsSync(row.file)) { await route.fulfill({ status: 404, body: "no asset" }); return; }
    await route.fulfill({ status: 200, contentType: "application/octet-stream", body: fs.readFileSync(row.file) });
  });

  await page.goto("/");
  await page.waitForFunction(() => typeof (window as any).ScoreHost !== "undefined", { timeout: 10000 });
  await page.evaluate(() => { (window as any).__SCOREHOST_API_BASE = "http://localhost:5173"; });

  for (const row of sample) {
    const rec: any = { pid: row.pid, label: row.label, file: path.basename(row.file) };
    try {
      const loaded = await page.evaluate(async (id) => await (window as any).ScoreHost.load(id), row.pid);
      rec.loaded = JSON.stringify(loaded);
      const artifact = JSON.stringify({ type: "score_highlight", config: { pieceId: row.pid, highlights: [] } });
      await page.evaluate(async (j) => { await (window as any).ScoreHost.showArtifact(j); }, artifact);
      await page.waitForSelector("#scorehost-container svg", { timeout: 25000 });
      const stats = await page.evaluate(() => {
        const c = document.getElementById("scorehost-container")!;
        return {
          svgCount: c.querySelectorAll("svg").length,
          useCount: c.querySelectorAll("svg use").length,         // Verovio glyph refs
          pathCount: c.querySelectorAll("svg path").length,       // LilyPond inline glyphs
          noteCount: c.querySelectorAll("svg g.note").length,
          measureCount: c.querySelectorAll("svg g.measure").length,
        };
      });
      Object.assign(rec, stats);
      rec.ok = stats.svgCount >= 1 && (stats.useCount > 0 || stats.pathCount > 50);
      if (SHOTS) {
        const safe = row.label.replace(/[^a-z0-9_.-]/gi, "_");
        await page.locator("#scorehost-container").screenshot({ path: path.join(SHOTS, `${safe}.png`) });
      }
    } catch (e: any) {
      rec.ok = false;
      rec.error = e?.message ?? String(e);
      await page.evaluate(() => { document.getElementById("scorehost-container")!.innerHTML = ""; }).catch(() => {});
    }
    out.write(JSON.stringify(rec) + "\n");
    // eslint-disable-next-line no-console
    console.log(`${rec.ok ? "OK " : "FAIL"} ${row.label}  use=${rec.useCount ?? "-"} path=${rec.pathCount ?? "-"} note=${rec.noteCount ?? "-"} meas=${rec.measureCount ?? "-"} ${rec.error ?? ""}`);
  }
  out.end();
});

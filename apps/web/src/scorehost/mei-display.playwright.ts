// DISPLAY verification: render a stratified sample of the 618 PD-clean .mei
// through the REAL scorehost Verovio worker (the exact engine the web ScorePanel
// and the iOS WKWebView use). MEI bytes are fulfilled from disk via the same
// /api/scores/:id/data path the app uses (route interception), so this exercises
// fetchScoreBytes -> worker load -> Verovio engrave end-to-end.
//
// Per piece: load, render full page 1, assert note glyphs engraved, count
// measures on page 1, screenshot. Failures are collected (never aborts mid-sweep)
// and written to mei_display_results.jsonl; the final assertion fails if any
// piece failed to engrave.
import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

// Opt-in local verification harness (not a CI test): set MEI_DIR + MEI_SAMPLE to
// the on-disk .mei dir and a stratified sample JSON; SHOTS_DIR for screenshots.
// Skips cleanly when unset so the scorehost CI suite is unaffected.
const SRC_MEI = process.env.MEI_DIR ?? "";
const SAMPLE = process.env.MEI_SAMPLE ?? "";
const SHOTS = process.env.SHOTS_DIR ?? "";
const haveData = SRC_MEI !== "" && SAMPLE !== "" && fs.existsSync(SAMPLE) && fs.existsSync(SRC_MEI);

interface SampleRow { pid: string; total_bars: number | null; composer: string | null; key: string | null; ts: string | null; coll: string; }
const sample: SampleRow[] = haveData ? JSON.parse(fs.readFileSync(SAMPLE, "utf8")) : [];

test("DISPLAY: stratified .mei sample engraves in real scorehost Verovio worker", async ({ page }) => {
  test.skip(!haveData, "set MEI_DIR + MEI_SAMPLE (+ SHOTS_DIR) to run the .mei display sweep");
  if (SHOTS) fs.mkdirSync(SHOTS, { recursive: true });
  const out = fs.createWriteStream(path.join(SHOTS || SRC_MEI, "mei_display_results.jsonl"), { flags: "w" });

  // Fulfill the app's score-data endpoint from the on-disk .mei for any sampled piece.
  await page.route("**/api/scores/*/data", async (route) => {
    const url = new URL(route.request().url());
    const m = url.pathname.match(/\/api\/scores\/(.+)\/data$/);
    const pid = m ? decodeURIComponent(m[1]) : "";
    const file = path.join(SRC_MEI, `${pid}.mei`);
    if (!fs.existsSync(file)) { await route.fulfill({ status: 404, body: "no mei" }); return; }
    await route.fulfill({ status: 200, contentType: "application/mei+xml", body: fs.readFileSync(file) });
  });

  await page.goto("/");
  await page.waitForFunction(() => typeof (window as any).ScoreHost !== "undefined", { timeout: 10000 });
  await page.evaluate(() => { (window as any).__SCOREHOST_API_BASE = "http://localhost:5173"; });

  const results: any[] = [];
  for (const row of sample) {
    const pid = row.pid;
    const rec: any = { pid, total_bars: row.total_bars, key: row.key, ts: row.ts, coll: row.coll };
    try {
      const loaded = await page.evaluate(async (id) => await (window as any).ScoreHost.load(id), pid);
      rec.loaded = JSON.stringify(loaded);
      const artifact = JSON.stringify({ type: "score_highlight", config: { pieceId: pid, highlights: [] } });
      await page.evaluate(async (json) => { await (window as any).ScoreHost.showArtifact(json); }, artifact);
      await page.waitForSelector("#scorehost-container svg", { timeout: 20000 });
      const stats = await page.evaluate(() => {
        const c = document.getElementById("scorehost-container")!;
        return {
          svgCount: c.querySelectorAll("svg").length,
          useCount: c.querySelectorAll("svg use").length,
          page1Measures: c.querySelectorAll("svg g.measure").length,
          noteCount: c.querySelectorAll("svg g.note").length,
        };
      });
      Object.assign(rec, stats);
      rec.ok = stats.svgCount >= 1 && stats.useCount > 0;
      if (SHOTS) {
        const safe = pid.replace(/[^a-z0-9_.-]/gi, "_");
        await page.locator("#scorehost-container").screenshot({ path: path.join(SHOTS, `${safe}.png`) });
      }
    } catch (e: any) {
      rec.ok = false;
      rec.error = e?.message ?? String(e);
      // reset worker state on crash so the next piece can still load
      await page.evaluate(() => { document.getElementById("scorehost-container")!.innerHTML = ""; }).catch(() => {});
    }
    out.write(JSON.stringify(rec) + "\n");
    results.push(rec);
    // eslint-disable-next-line no-console
    console.log(`${rec.ok ? "OK " : "FAIL"} ${pid}  bars=${row.total_bars} use=${rec.useCount ?? "-"} m1=${rec.page1Measures ?? "-"} ${rec.error ?? ""}`);
  }
  out.end();

  const failed = results.filter((r) => !r.ok);
  console.log(`\nDISPLAY SUMMARY: ${results.length - failed.length}/${results.length} engraved; ${failed.length} failed`);
  if (failed.length) console.log("FAILURES:", failed.map((r) => r.pid).join(", "));
  expect(failed, `pieces that failed to engrave: ${failed.map((r) => r.pid).join(", ")}`).toEqual([]);
});

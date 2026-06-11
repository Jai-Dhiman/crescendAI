// apps/web/scripts/fetch-soundfont.ts
// Fetches the acoustic grand piano SoundFont sample pack from the smplr CDN
// and writes it into apps/web/public/soundfonts/ for self-hosted playback.
//
// Usage: bun apps/web/scripts/fetch-soundfont.ts
//
// The smplr package serves samples from:
//   https://gleitz.github.io/midi-js-soundfonts/MusyngKite/{instrument}-mp3.js
// Each file is a JS object mapping note names to base64 data URIs.
// We download the acoustic_grand_piano instrument and write:
//   apps/web/public/soundfonts/acoustic_grand_piano-mp3.js

import { writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const INSTRUMENT = "acoustic_grand_piano";
const CDN_URL = `https://gleitz.github.io/midi-js-soundfonts/MusyngKite/${INSTRUMENT}-mp3.js`;
const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(__dirname, "../public/soundfonts");
const OUT_FILE = resolve(OUT_DIR, `${INSTRUMENT}-mp3.js`);

async function main() {
  console.log(`Fetching ${CDN_URL} ...`);
  const res = await fetch(CDN_URL);
  if (!res.ok) {
    throw new Error(`Fetch failed: ${res.status} ${res.statusText}`);
  }
  const text = await res.text();
  mkdirSync(OUT_DIR, { recursive: true });
  writeFileSync(OUT_FILE, text, "utf-8");
  console.log(`Written to ${OUT_FILE} (${text.length} bytes)`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

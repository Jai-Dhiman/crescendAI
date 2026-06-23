// apps/api/src/services/keys.ts
// Best-effort TS port of model/src/exercise_corpus/keys.py.
//
// parseKeyToPc is a strict SUPERSET of the Python parse_key_to_pc: identical
// results for every input the oracle accepts (uppercase tonics), plus correct
// tonic extraction for lowercase-minor tags the oracle rejects (it raises). The
// added normalization is the single first-char capitalization in step 3 below;
// _PC, the mode-stripping, and transposeInterval are byte-for-byte the oracle.
// Returns null (never throws) so the caller can degrade to transpose=0 + warn.

// _PC replicated verbatim from keys.py (uppercase tonic entries only).
const PC: Record<string, number> = {
  C: 0,
  "C#": 1, Db: 1,
  D: 2,
  "D#": 3, Eb: 3,
  E: 4,
  F: 5,
  "F#": 6, Gb: 6,
  G: 7,
  "G#": 8, Ab: 8,
  A: 9,
  "A#": 10, Bb: 10,
  B: 11,
};

export function parseKeyToPc(keySignature: string): number | null {
  let s = keySignature.trim();
  // Strip trailing " major" / " minor" (case-insensitive), like the oracle.
  for (const suffix of [" major", " minor"]) {
    if (s.toLowerCase().endsWith(suffix)) {
      s = s.slice(0, -suffix.length).trim();
      break;
    }
  }
  // Strip trailing "m" (minor shorthand) — only if not the entire string.
  if (s.endsWith("m") && s.length > 1) {
    s = s.slice(0, -1);
  }
  // Superset normalization the oracle lacks: capitalize ONLY the first char of
  // the remaining tonic so "c"->"C", "eb"->"Eb", "f#"->"F#", "C#"->"C#".
  if (s.length > 0) {
    s = s[0].toUpperCase() + s.slice(1);
  }
  return s in PC ? PC[s] : null;
}

export function transposeInterval(fromPc: number, toPc: number): number {
  // Nearest-octave semitone shift, range [-5, +6], tritone resolves to +6.
  let d = ((toPc - fromPc) % 12 + 12) % 12;
  if (d > 6) d -= 12;
  return d;
}

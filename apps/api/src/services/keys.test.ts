// apps/api/src/services/keys.test.ts
import { describe, expect, it, test } from "vitest";
import fixture from "./keys-parity-fixture.json";
import { parseKeyToPc, transposeInterval } from "./keys";

describe("keys TS port — parity with the Python oracle (uppercase domain)", () => {
  // The fixture is generated from model/src/exercise_corpus/keys.py. The TS port
  // must match it byte-for-byte for every input the oracle accepts.
  for (const [key, pc] of Object.entries(fixture.pc as Record<string, number>)) {
    it(`parseKeyToPc(${JSON.stringify(key)}) === ${pc} (oracle parity)`, () => {
      expect(parseKeyToPc(key)).toBe(pc);
    });
  }

  for (const { from, to, expected } of fixture.intervals as Array<{
    from: string;
    to: string;
    expected: number;
  }>) {
    it(`transposeInterval(${from} -> ${to}) === ${expected} (oracle parity)`, () => {
      const f = parseKeyToPc(from);
      const t = parseKeyToPc(to);
      expect(f).not.toBeNull();
      expect(t).not.toBeNull();
      expect(transposeInterval(f as number, t as number)).toBe(expected);
    });
  }
});

describe("keys TS port — intentional supersets (oracle RAISES, TS resolves)", () => {
  // These diverge from the Python oracle BY DESIGN: keys.py._PC is uppercase-only
  // and raises on lowercase-minor tags, which is latently broken for its own corpus
  // (e.g. czerny_001 = "c"). The TS port capitalizes the first tonic char before
  // lookup, so it resolves the correct tonic pitch class. Mode is irrelevant to
  // semitone transposition.
  test('parseKeyToPc("c") === 0 (lowercase minor, oracle would raise)', () => {
    expect(parseKeyToPc("c")).toBe(0);
  });
  test('parseKeyToPc("eb") === 3 (oracle would raise)', () => {
    expect(parseKeyToPc("eb")).toBe(3);
  });
  test('parseKeyToPc("f#") === 6 (oracle would raise)', () => {
    expect(parseKeyToPc("f#")).toBe(6);
  });
  test("parseKeyToPc returns null on genuine garbage", () => {
    expect(parseKeyToPc("zzz")).toBeNull();
  });
});

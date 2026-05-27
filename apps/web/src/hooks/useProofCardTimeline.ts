// apps/web/src/hooks/useProofCardTimeline.ts
import { useCallback, useRef, useState } from "react";
import type { RefObject } from "react";
import type { ScoreIR } from "../lib/score-ir";

type BarTimeline = Array<{ bar: number; tSec: number }>;

export function useProofCardTimeline(
  _audioRef: RefObject<HTMLAudioElement | null>,
  scoreIR: ScoreIR | null,
  barTimeline: BarTimeline,
) {
  const [currentTime, setCurrentTimeState] = useState(0);
  const barTimelineRef = useRef(barTimeline);
  barTimelineRef.current = barTimeline;
  const scoreIRRef = useRef(scoreIR);
  scoreIRRef.current = scoreIR;

  const setCurrentTime = useCallback((t: number) => {
    setCurrentTimeState(t);
  }, []);

  // Returns the qstampStart of the bar whose tSec window contains the given time.
  // ScoreCursor.findBar() binary-searches bar.qstampStart/qstampEnd (quaternote floats),
  // so we must return a qstamp float, not a raw bar number.
  const qstampForTime = useCallback((tSec: number): number | null => {
    const timeline = barTimelineRef.current;
    if (timeline.length === 0) return null;

    // Find the last entry whose tSec <= tSec (sorted ascending)
    let matchedEntry = timeline[0];
    if (!matchedEntry) return null;
    for (let i = 0; i < timeline.length; i++) {
      const e = timeline[i];
      if (!e) continue;
      if (e.tSec <= tSec) matchedEntry = e;
      else break;
    }

    // Look up the bar's qstampStart from ScoreIR.bars by barNumber
    const ir = scoreIRRef.current;
    if (ir) {
      const barIR = ir.bars.find((b) => b.barNumber === matchedEntry!.bar);
      if (barIR) return barIR.qstampStart;
    }

    // Fallback: scoreIR not yet loaded — return null so cursor stays hidden
    return null;
  }, []);

  return { currentTime, setCurrentTime, qstampForTime };
}

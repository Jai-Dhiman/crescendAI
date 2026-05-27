// apps/web/src/components/ProofCard.tsx
import { useCallback, useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import type { ProofCardManifest } from "../types/landing";
import type { ScoreIR } from "../lib/score-ir";
import { ScoreCursor } from "../lib/score-cursor";
import { useProofCardTimeline } from "../hooks/useProofCardTimeline";
import { trackLandingEvent } from "../lib/landing-analytics";
import { BarScoreChip } from "./BarScoreChip";

interface ProofCardProps {
  manifest: ProofCardManifest;
  cardIndex: number;
}

type LoadState = "loading" | "ready" | "score-failed" | "audio-failed";

export function ProofCard({ manifest, cardIndex }: ProofCardProps) {
  const scoreContainerRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const [scoreIR, setScoreIR] = useState<ScoreIR | null>(null);
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [exerciseComponent, setExerciseComponent] = useState<unknown>(null);
  const [activeBar, setActiveBar] = useState<number | null>(null);
  const reducedMotion = useRef(
    typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );

  const [duration, setDuration] = useState(30);
  const [isPlaying, setIsPlaying] = useState(false);

  const { currentTime, setCurrentTime, qstampForTime } = useProofCardTimeline(
    audioRef,
    scoreIR,
    manifest.barTimeline,
  );

  // Load scoreIR, score SVG, and exercise JSON
  useEffect(() => {
    let cancelled = false;

    async function load() {
      // Load score SVG
      if (scoreContainerRef.current) {
        try {
          const svgRes = await fetch(manifest.scoreSvgUrl);
          if (!cancelled && svgRes.ok) {
            const svgText = await svgRes.text();
            if (!cancelled && scoreContainerRef.current) {
              scoreContainerRef.current.textContent = "";
              // biome-ignore lint/security/noDomManipulation: static SVG from prebaked landing asset on same origin, not user input
              scoreContainerRef.current.insertAdjacentHTML("beforeend", svgText);
            }
          }
        } catch {
          // Score failed; continue — graceful degradation handled by loadState
        }
      }

      // Load scoreIR
      try {
        const irRes = await fetch(manifest.scoreIRUrl);
        if (!cancelled) {
          if (irRes.ok) {
            const ir = (await irRes.json()) as ScoreIR;
            if (!cancelled) setScoreIR(ir);
          }
          // non-ok response: scoreIR remains null; ScoreCursor will not animate
        }
      } catch {
        // fetch threw (network error); scoreIR remains null
      }

      // Load exercise
      try {
        const exRes = await fetch(manifest.exerciseUrl);
        if (!cancelled && exRes.ok) {
          const ex = await exRes.json();
          if (!cancelled) setExerciseComponent(ex);
        }
      } catch {
        // exercise fetch failed; exercise section will not render
      }

      if (!cancelled) setLoadState("ready");
    }

    load();
    return () => { cancelled = true; };
  }, [manifest.scoreIRUrl, manifest.scoreSvgUrl, manifest.exerciseUrl]);

  // IntersectionObserver for scroll-autoplay
  useEffect(() => {
    const card = cardRef.current;
    const audio = audioRef.current;
    if (!card || !audio || reducedMotion.current) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        if (!entry) return;
        if (entry.intersectionRatio >= 0.6) {
          audio.play().catch(() => {
            // Autoplay blocked by browser; user must interact
          });
          trackLandingEvent("landing_proof_card_enter", { cardIndex });
        } else {
          audio.pause();
        }
      },
      { threshold: [0, 0.6] },
    );

    observer.observe(card);
    return () => observer.disconnect();
  }, [cardIndex]);

  // Sync currentTime → audio element when set from scrubber
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (Math.abs(audio.currentTime - currentTime) > 0.1) {
      audio.currentTime = currentTime;
    }
  }, [currentTime]);

  // Audio timeupdate, ended, error → state
  // Using addEventListener (not React onError prop) because React 18 does not
  // reliably fire synthetic events for HTMLMediaElement in jsdom.
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    function onTimeUpdate() {
      setCurrentTime(audio!.currentTime);
    }
    function onEnded() {
      trackLandingEvent("landing_proof_card_played_to_end", { cardIndex });
    }
    function onError() {
      setLoadState("audio-failed");
    }
    function onLoadedMetadata() {
      setDuration(audio!.duration);
    }
    function onPlay() {
      setIsPlaying(true);
    }
    function onPause() {
      setIsPlaying(false);
    }
    audio.addEventListener("timeupdate", onTimeUpdate);
    audio.addEventListener("ended", onEnded);
    audio.addEventListener("error", onError);
    audio.addEventListener("loadedmetadata", onLoadedMetadata);
    audio.addEventListener("durationchange", onLoadedMetadata);
    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    return () => {
      audio.removeEventListener("timeupdate", onTimeUpdate);
      audio.removeEventListener("ended", onEnded);
      audio.removeEventListener("error", onError);
      audio.removeEventListener("loadedmetadata", onLoadedMetadata);
      audio.removeEventListener("durationchange", onLoadedMetadata);
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
    };
  }, [cardIndex, setCurrentTime]);

  // ScoreCursor — instantiate once when scoreIR loads; the rAF loop reads audio
  // position live on every tick via audioRef.current so currentTime state never
  // needs to be in the dependency array (adding it would destroy/recreate the
  // cursor at ~4Hz, cancelling the rAF before it can self-reschedule).
  const cursorQstampSource = useCallback(() => {
    const t = audioRef.current?.currentTime ?? 0;
    return qstampForTime(t) ?? 0;
  }, [qstampForTime]);

  useEffect(() => {
    if (scoreIR === null || scoreContainerRef.current === null) return;
    const cursor = new ScoreCursor({
      pieceId: manifest.pieceId,
      container: scoreContainerRef.current,
      ir: scoreIR,
      qstampSource: cursorQstampSource,
    });
    cursor.start();
    return () => {
      cursor.stop();
    };
  }, [scoreIR, manifest.pieceId, cursorQstampSource]);

  // Keyboard navigation: Tab cycles bars, Enter opens chip, Escape closes
  const barNumbers = Object.keys(manifest.perBarScores).map(Number).sort((a, b) => a - b);

  function handleBarKeyDown(e: KeyboardEvent<HTMLButtonElement>, barNumber: number) {
    if (e.key === "Enter") {
      setActiveBar(barNumber);
      trackLandingEvent("landing_bar_tap", { cardIndex, barNumber });
    } else if (e.key === "Escape") {
      setActiveBar(null);
    }
  }

  function handleBarClick(barNumber: number) {
    setActiveBar((prev) => (prev === barNumber ? null : barNumber));
    trackLandingEvent("landing_bar_tap", { cardIndex, barNumber });
  }

  // Asset prefetch for subsequent cards
  useEffect(() => {
    if (cardIndex !== 0) return;
    const card = cardRef.current;
    if (!card) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries[0]?.isIntersecting) return;
        for (const n of [2, 3]) {
          for (const asset of ["scoreir.json", "score.svg", "exercise.json", "recording.opus"]) {
            const link = document.createElement("link");
            link.rel = "prefetch";
            link.href = `/landing/card-${n}/${asset}`;
            document.head.appendChild(link);
          }
        }
        observer.disconnect();
      },
      { threshold: 0.1 },
    );
    observer.observe(card);
    return () => observer.disconnect();
  }, [cardIndex]);

  const showPlayButton = reducedMotion.current || loadState === "audio-failed";

  return (
    <div ref={cardRef} className="w-full bg-surface border border-border rounded-xl overflow-hidden">
      {/* Score area */}
      <div className="relative" data-testid="proof-card-score">
        <div
          ref={scoreContainerRef}
          className="score-container w-full"
          style={{ position: "relative" }}
          role="img"
          aria-label={`Score for ${manifest.title}`}
        />
        {/* Bar tap targets rendered over score */}
        <div
          className="absolute inset-0"
          aria-label="Bar score inspection overlay"
        >
          {barNumbers.map((barNumber, idx) => (
            <button
              key={barNumber}
              type="button"
              data-bar={barNumber}
              tabIndex={0}
              className="absolute focus:outline-none focus:ring-2 focus:ring-accent"
              style={{
                left: `${(idx / barNumbers.length) * 100}%`,
                width: `${(1 / barNumbers.length) * 100}%`,
                top: 0,
                bottom: 0,
                background: barNumber === manifest.focusBar ? "rgba(255,255,255,0.05)" : "transparent",
              }}
              onClick={() => handleBarClick(barNumber)}
              onKeyDown={(e) => handleBarKeyDown(e, barNumber)}
              aria-label={`Bar ${barNumber} — tap to inspect quality scores`}
              aria-pressed={activeBar === barNumber}
            />
          ))}
        </div>
        {/* BarScoreChip overlay */}
        {activeBar !== null && manifest.perBarScores[activeBar] && (
          <div className="absolute top-2 right-2 z-20">
            <BarScoreChip
              scores={manifest.perBarScores[activeBar]}
              barNumber={activeBar}
              onClose={() => setActiveBar(null)}
            />
          </div>
        )}
      </div>

      {/* Controls + diagnosis */}
      <div className="px-6 py-5 space-y-4">
        {/* Audio scrubber */}
        <div data-testid="proof-card-scrubber" className="flex items-center gap-3">
          {showPlayButton && (
            <button
              type="button"
              aria-label={isPlaying ? "Pause" : "Play"}
              className="shrink-0 w-8 h-8 flex items-center justify-center rounded-full bg-accent text-cream"
              onClick={() => {
                const audio = audioRef.current;
                if (!audio) return;
                if (audio.paused) {
                  audio.play().catch(() => {});
                } else {
                  audio.pause();
                }
              }}
            >
              <span aria-hidden="true">&#9654;</span>
            </button>
          )}
          <input
            type="range"
            min={0}
            max={duration}
            step={0.1}
            value={currentTime}
            onChange={(e) => setCurrentTime(Number(e.target.value))}
            className="flex-1 h-1 accent-accent"
            aria-label="Playback position"
          />
          <audio
            ref={audioRef}
            src={manifest.audioUrl}
            preload={cardIndex === 0 ? "auto" : "none"}
          />
        </div>

        {/* Piece title and era */}
        <div>
          <span className="text-label-sm text-text-tertiary uppercase tracking-wide capitalize">
            {manifest.era}
          </span>
          <h3 className="font-display text-display-sm text-cream mt-0.5">{manifest.title}</h3>
        </div>

        {/* Teacher diagnosis for focus bar */}
        <p className="text-body-md text-text-secondary">{manifest.diagnosis}</p>

        {/* Generated exercise */}
        {exerciseComponent && (
          <div data-testid="proof-card-exercise">
            {/* Render exercise as a static summary — not using Artifact store lifecycle on landing */}
            <div className="bg-surface-2 border border-border rounded-lg p-4">
              <p className="text-body-sm font-medium text-cream">
                {(exerciseComponent as { config?: { targetSkill?: string } }).config?.targetSkill ?? "Exercise"}
              </p>
              <p className="text-body-xs text-text-secondary mt-1">
                {(exerciseComponent as { config?: { sourcePassage?: string } }).config?.sourcePassage ?? ""}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

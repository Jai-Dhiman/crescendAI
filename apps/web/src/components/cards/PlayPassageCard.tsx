// apps/web/src/components/cards/PlayPassageCard.tsx
import { useEffect, useRef, useState } from "react";
import { api } from "../../lib/api";
import { DIMENSION_COLORS } from "../../lib/mock-session";
import { PassagePlayer } from "../../lib/passage-player";
import type { ClipResult } from "../../lib/score-renderer";
import { scoreRenderer } from "../../lib/score-renderer";
import type { PassageManifest, PlayPassageConfig } from "../../lib/types";
import { SvgClip } from "../SvgClip";

interface PlayPassageCardProps {
  config: PlayPassageConfig;
  onExpand?: () => void;
  artifactId?: string;
  /** Sandbox only: skip the manifest API fetch and use this directly. */
  _mockManifest?: PassageManifest;
  /** Sandbox only: skip scoreRenderer.getClip and use this directly. */
  _mockClip?: ClipResult;
  /** Sandbox only: when true with _mockManifest+_mockClip, still runs PassagePlayer (real audio). */
  _playable?: boolean;
}

// "audio_error" = manifest+score loaded fine but audio chunks unavailable.
// Score + annotation still render; only the play button indicates the failure.
type LoadState = "loading" | "ready" | "audio_error" | "error";

export function PlayPassageCard({
  config,
  onExpand,
  artifactId: _artifactId,
  _mockManifest,
  _mockClip,
  _playable,
}: PlayPassageCardProps) {
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [clip, setClip] = useState<ClipResult | null>(null);
  const [manifest, setManifest] = useState<PassageManifest | null>(null);
  const playerRef = useRef<PassagePlayer | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    // Sandbox fast-path: both mocks provided and audio not needed → skip all async.
    // playerRef stays null; play button renders but clicking is a no-op.
    if (_mockManifest && _mockClip && !_playable) {
      setManifest(_mockManifest);
      setClip(_mockClip);
      setLoadState("ready");
      return;
    }

    let cancelled = false;
    (async () => {
      // Phase 1: fetch manifest + score clip (data failure = full error state).
      // When mocks are provided, use them directly and skip the await.
      let m: PassageManifest;
      let c: ClipResult;
      try {
        m = _mockManifest ?? await api.sessions.getPassage(config.sessionId, config.bars);
        if (!_mockManifest && cancelled) return;
        c = _mockClip ?? await scoreRenderer.getClip(m.pieceId, config.bars[0], config.bars[1]);
        if (!_mockClip && cancelled) return;
      } catch (err) {
        console.error("PlayPassageCard fetch failed", err);
        if (!cancelled) setLoadState("error");
        return;
      }
      // Score data is ready — commit it so the card can render even if audio fails.
      setManifest(m);
      setClip(c);

      // Phase 2: load audio (failure = audio_error, score still renders)
      try {
        const ctx = new AudioContext();
        ctxRef.current = ctx;
        const player = new PassagePlayer(m, ctx);
        await player.load();
        if (cancelled) {
          player.destroy();
          ctxRef.current = null;
          return;
        }
        playerRef.current = player;
        setLoadState("ready");
      } catch (err) {
        console.error("PlayPassageCard audio load failed", err);
        if (!cancelled) setLoadState("audio_error");
      }
    })();
    return () => {
      cancelled = true;
      playerRef.current?.destroy();
      ctxRef.current?.close();
      ctxRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.sessionId, config.bars[0], config.bars[1], _mockManifest, _mockClip, _playable]);

  const color =
    DIMENSION_COLORS[config.dimension as keyof typeof DIMENSION_COLORS] ?? "#7a9a82";

  return (
    <div className="bg-surface-card border border-border rounded-xl overflow-hidden mt-3" onClick={onExpand}>
      {loadState === "loading" && (
        <div className="h-10 flex items-center justify-center">
          <div className="w-3.5 h-3.5 rounded-full border-2 border-text-tertiary/50 border-t-transparent animate-spin" />
        </div>
      )}
      {(loadState === "ready" || loadState === "audio_error") && clip && manifest && (
        <div className="px-3 pt-3">
          <div
            style={{
              position: "relative",
              borderRadius: "6px",
              border: `1.5px solid ${color}40`,
              backgroundColor: "white",
              overflow: "hidden",
            }}
          >
            <SvgClip
              svgMarkup={clip.svg}
              startMeasureId={clip.startMeasureId}
              endMeasureId={clip.endMeasureId}
            />
          </div>
          {loadState === "ready" ? (
            <button
              type="button"
              aria-label="Play passage"
              onClick={() => void playerRef.current?.play()}
              className="mt-3 px-3 py-1.5 rounded-md border border-border text-body-sm text-text-primary hover:bg-surface transition-colors"
            >
              Play
            </button>
          ) : (
            <span className="mt-3 inline-block text-body-sm text-text-tertiary">
              Audio unavailable
            </span>
          )}
        </div>
      )}
      {loadState === "error" && (
        <div className="p-4 text-body-sm text-text-tertiary">couldn't load audio</div>
      )}
      <div className="p-4 flex flex-col gap-3.5">
        <div className="flex items-center gap-1.5 shrink-0">
          <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: color }} />
          <span className="text-label-sm text-text-tertiary uppercase tracking-wide">
            {config.dimension}
          </span>
        </div>
        <span className="text-body-xs text-text-tertiary">
          bars {config.bars[0]}–{config.bars[1]}
        </span>
        <p className="text-body-sm text-text-primary mt-0.5 leading-snug">
          {config.annotation}
        </p>
      </div>
    </div>
  );
}

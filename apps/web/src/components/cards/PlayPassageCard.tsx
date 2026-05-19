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
}

type LoadState = "loading" | "ready" | "error";

export function PlayPassageCard({ config }: PlayPassageCardProps) {
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [clip, setClip] = useState<ClipResult | null>(null);
  const [manifest, setManifest] = useState<PassageManifest | null>(null);
  const playerRef = useRef<PassagePlayer | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const m = await api.sessions.getPassage(config.sessionId, config.bars);
        if (cancelled) return;
        const c = await scoreRenderer.getClip(m.pieceId, config.bars[0], config.bars[1]);
        if (cancelled) return;
        const ctx = new AudioContext();
        const player = new PassagePlayer(m, ctx);
        await player.load();
        if (cancelled) {
          player.destroy();
          return;
        }
        playerRef.current = player;
        setManifest(m);
        setClip(c);
        setLoadState("ready");
      } catch (err) {
        console.error("PlayPassageCard load failed", err);
        if (!cancelled) setLoadState("error");
      }
    })();
    return () => {
      cancelled = true;
      playerRef.current?.destroy();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.sessionId, config.bars[0], config.bars[1]]);

  const color =
    DIMENSION_COLORS[config.dimension as keyof typeof DIMENSION_COLORS] ?? "#7a9a82";

  return (
    <div className="bg-surface-card border border-border rounded-xl overflow-hidden mt-3">
      {loadState === "loading" && (
        <div className="h-10 flex items-center justify-center">
          <div className="w-3.5 h-3.5 rounded-full border-2 border-text-tertiary/50 border-t-transparent animate-spin" />
        </div>
      )}
      {loadState === "ready" && clip && manifest && (
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
          <button
            type="button"
            aria-label="Play passage"
            onClick={() => playerRef.current?.play()}
            className="mt-3 px-3 py-1.5 rounded-md border border-border text-body-sm text-text-primary hover:bg-surface transition-colors"
          >
            Play
          </button>
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

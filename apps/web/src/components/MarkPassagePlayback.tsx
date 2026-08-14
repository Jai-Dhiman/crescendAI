import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { DIMENSION_COLOR_VAR } from "../lib/dimension-colors";
import type { Mark } from "../lib/mark";
import { createPassageQstampSource } from "../lib/passage-cursor";
import { PassagePlayer } from "../lib/passage-player";
import { ScoreCursor } from "../lib/score-cursor";
import type { ScoreIR } from "../lib/score-ir";
import { scoreRenderer } from "../lib/score-renderer";
import type { PassageManifest } from "../lib/types";
import { ClipSvg } from "./ClipSvg";

interface MarkPassagePlaybackProps {
	mark: Mark;
	sessionId: string;
}

/** "no_bars" is distinct from "error": a timestamp-only mark's anchor has no
 * bars to fetch a passage for at all -- that's an honest "unavailable," not
 * a failure, matching mark.ts's silent-degradation precedent. */
type LoadState = "no_bars" | "loading" | "ready" | "audio_error" | "error";

/**
 * Cursor-tracked own-audio playback of exactly a mark's bars: fetches the
 * passage manifest + clip, decodes the audio, and drives a ScoreCursor via
 * passage-cursor.ts's tSec -> qstamp bridge.
 */
export function MarkPassagePlayback({
	mark,
	sessionId,
}: MarkPassagePlaybackProps) {
	const bars: [number, number] | null =
		mark.anchor.type === "bars"
			? [mark.anchor.bars[0], mark.anchor.bars[1]]
			: null;

	const [loadState, setLoadState] = useState<LoadState>(
		bars ? "loading" : "no_bars",
	);
	const [clipSvg, setClipSvg] = useState<string | null>(null);
	const [manifest, setManifest] = useState<PassageManifest | null>(null);
	const [clipIr, setClipIr] = useState<ScoreIR | null>(null);
	const playerRef = useRef<PassagePlayer | null>(null);
	const ctxRef = useRef<AudioContext | null>(null);
	const containerRef = useRef<HTMLDivElement>(null);
	const currentTSecRef = useRef<number | null>(null);

	useEffect(() => {
		if (!bars) return;

		let cancelled = false;
		(async () => {
			let m: PassageManifest;
			let svg: string;
			let ir: ScoreIR;
			try {
				m = await api.sessions.getPassage(sessionId, bars);
				if (cancelled) return;
				const clip = await scoreRenderer.getClipPlayback(
					m.pieceId,
					bars[0],
					bars[1],
				);
				svg = clip.svg;
				ir = clip.ir;
				if (cancelled) return;
			} catch (err) {
				console.error("MarkPassagePlayback fetch failed", err);
				if (!cancelled) setLoadState("error");
				return;
			}
			setManifest(m);
			setClipSvg(svg);
			setClipIr(ir);

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
				player.onTick((tSec) => {
					currentTSecRef.current = tSec;
				});
				playerRef.current = player;
				setLoadState("ready");
			} catch (err) {
				console.error("MarkPassagePlayback audio load failed", err);
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
	}, [sessionId, bars?.[0], bars?.[1]]);

	useEffect(() => {
		if (!manifest || !clipIr || !containerRef.current) return;
		const qstampSource = createPassageQstampSource(
			manifest.barTimeline,
			clipIr.bars,
			() => currentTSecRef.current,
		);
		const cursor = new ScoreCursor({
			pieceId: manifest.pieceId,
			container: containerRef.current,
			ir: clipIr,
			qstampSource,
		});
		cursor.start();
		return () => cursor.stop();
	}, [manifest, clipIr]);

	if (loadState === "no_bars") {
		return (
			<p className="text-body-sm text-ink-tertiary">
				No passage audio for this mark.
			</p>
		);
	}

	const color = DIMENSION_COLOR_VAR[mark.dimension];

	return (
		<div>
			{loadState === "loading" && (
				<div className="h-10 flex items-center justify-center">
					<div className="w-3.5 h-3.5 rounded-full border-2 border-ink-tertiary/50 border-t-transparent animate-spin" />
				</div>
			)}
			{(loadState === "ready" || loadState === "audio_error") && clipSvg && (
				<div className="px-3 pt-3">
					<div
						ref={containerRef}
						style={{
							position: "relative",
							borderRadius: "6px",
							border: `1.5px solid ${color}40`,
							backgroundColor: "var(--color-score-canvas)",
							overflow: "hidden",
						}}
					>
						<ClipSvg svg={clipSvg} />
					</div>
					{loadState === "ready" ? (
						<button
							type="button"
							aria-label="Play passage"
							onClick={() => void playerRef.current?.play()}
							className="mt-3 px-3 py-1.5 rounded-md border border-border-subtle text-body-sm text-ink-primary hover:bg-surface-raised transition-colors"
						>
							Play
						</button>
					) : (
						<span className="mt-3 inline-block text-body-sm text-ink-tertiary">
							Audio unavailable
						</span>
					)}
				</div>
			)}
			{loadState === "error" && (
				<p className="p-4 text-body-sm text-ink-tertiary">
					couldn't load audio
				</p>
			)}
		</div>
	);
}

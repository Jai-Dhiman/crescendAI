// apps/web/src/components/cards/PlayPassageCard.tsx
import { useEffect, useRef, useState } from "react";
import { api } from "../../lib/api";
import { DIMENSION_COLOR_VAR } from "../../lib/dimension-colors";
import { PassagePlayer } from "../../lib/passage-player";
import { scoreRenderer } from "../../lib/score-renderer";
import type { PassageManifest, PlayPassageConfig } from "../../lib/types";
import { ClipSvg } from "../ClipSvg";

interface PlayPassageCardProps {
	config: PlayPassageConfig;
	onExpand?: () => void;
	artifactId?: string;
	_mockManifest?: PassageManifest;
	_mockClip?: string;
	_playable?: boolean;
}

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
	const [clipSvg, setClipSvg] = useState<string | null>(null);
	const [manifest, setManifest] = useState<PassageManifest | null>(null);
	const playerRef = useRef<PassagePlayer | null>(null);
	const ctxRef = useRef<AudioContext | null>(null);

	useEffect(() => {
		if (_mockManifest && _mockClip && !_playable) {
			setManifest(_mockManifest);
			setClipSvg(_mockClip);
			setLoadState("ready");
			return;
		}

		let cancelled = false;
		(async () => {
			let m: PassageManifest;
			let svg: string;
			try {
				m =
					_mockManifest ??
					(await api.sessions.getPassage(config.sessionId, config.bars));
				if (!_mockManifest && cancelled) return;
				svg =
					_mockClip ??
					(await scoreRenderer.getClip(
						m.pieceId,
						config.bars[0],
						config.bars[1],
					));
				if (!_mockClip && cancelled) return;
			} catch (err) {
				console.error("PlayPassageCard fetch failed", err);
				if (!cancelled) setLoadState("error");
				return;
			}
			setManifest(m);
			setClipSvg(svg);

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
	}, [
		config.sessionId,
		config.bars[0],
		config.bars[1],
		_mockManifest,
		_mockClip,
		_playable,
	]);

	const color =
		DIMENSION_COLOR_VAR[config.dimension as keyof typeof DIMENSION_COLOR_VAR] ??
		"var(--color-accent)";

	return (
		// The explicit aria-label matters: without it the row's accessible name is
		// the concatenated text of its children, including the nested play button.
		// biome-ignore lint/a11y/useSemanticElements: the card contains its own play <button>; a <button> wrapper would nest interactive controls.
		<div
			role="button"
			tabIndex={0}
			aria-label={`Expand passage, bars ${config.bars[0]}-${config.bars[1]}`}
			className="bg-surface-raised border border-border-subtle rounded-xl overflow-hidden mt-3"
			onClick={onExpand}
			onKeyDown={(e) => {
				if (e.key === "Enter" || e.key === " ") {
					e.preventDefault();
					onExpand?.();
				}
			}}
		>
			{loadState === "loading" && (
				<div className="h-10 flex items-center justify-center">
					<div className="w-3.5 h-3.5 rounded-full border-2 border-ink-tertiary/50 border-t-transparent animate-spin" />
				</div>
			)}
			{(loadState === "ready" || loadState === "audio_error") &&
				clipSvg &&
				manifest && (
					<div className="px-3 pt-3">
						<div
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
				<div className="p-4 text-body-sm text-ink-tertiary">
					couldn't load audio
				</div>
			)}
			<div className="p-4 flex flex-col gap-3.5">
				<div className="flex items-center gap-1.5 shrink-0">
					<span
						className="w-1.5 h-1.5 rounded-full"
						style={{ backgroundColor: color }}
					/>
					<span className="text-label-sm text-ink-tertiary uppercase tracking-wide">
						{config.dimension}
					</span>
				</div>
				<span className="text-body-xs text-ink-tertiary">
					bars {config.bars[0]}–{config.bars[1]}
				</span>
				<p className="text-body-sm text-ink-primary mt-0.5 leading-snug">
					{config.annotation}
				</p>
			</div>
		</div>
	);
}

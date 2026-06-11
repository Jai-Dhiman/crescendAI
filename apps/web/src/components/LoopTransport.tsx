interface LoopTransportProps {
	isPlaying: boolean;
	isCounting: boolean;
	audioUnavailable: boolean;
	tempoFactor: number;
	onPlay: () => void;
	onPause: () => void;
	onStop: () => void;
	onTempoChange: (f: number) => void;
}

export function LoopTransport({
	isPlaying,
	isCounting,
	audioUnavailable,
	tempoFactor,
	onPlay,
	onPause,
	onStop,
	onTempoChange,
}: LoopTransportProps) {
	return (
		<div
			data-testid="loop-transport"
			className="flex items-center gap-3 px-4 py-2 bg-surface-card border-t border-border/60"
		>
			<button
				type="button"
				onClick={isPlaying || isCounting ? onPause : onPlay}
				className="shrink-0 w-8 h-8 flex items-center justify-center rounded-full border border-border hover:border-accent text-text-secondary hover:text-cream transition-colors"
				aria-label={isPlaying || isCounting ? "Pause" : "Play"}
			>
				{isPlaying || isCounting ? (
					<svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" aria-hidden="true">
						<rect x="1" y="1" width="4" height="10" rx="1" />
						<rect x="7" y="1" width="4" height="10" rx="1" />
					</svg>
				) : (
					<svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" aria-hidden="true">
						<polygon points="2,1 11,6 2,11" />
					</svg>
				)}
			</button>

			<button
				type="button"
				onClick={onStop}
				disabled={!isPlaying && !isCounting}
				className="shrink-0 w-8 h-8 flex items-center justify-center rounded-full border border-border hover:border-accent text-text-secondary hover:text-cream transition-colors disabled:opacity-40"
				aria-label="Stop"
			>
				<svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor" aria-hidden="true">
					<rect x="0" y="0" width="10" height="10" rx="1" />
				</svg>
			</button>

			<div className="flex items-center gap-2 flex-1 min-w-0">
				<span className="text-label-sm text-text-tertiary shrink-0">Speed</span>
				<input
					type="range"
					min={0.25}
					max={1.0}
					step={0.05}
					value={tempoFactor}
					onChange={(e) => onTempoChange(Number.parseFloat(e.target.value))}
					className="flex-1 accent-accent"
					aria-label="Tempo factor"
				/>
				<span className="text-label-sm text-text-tertiary shrink-0 w-10 text-right">
					{Math.round(tempoFactor * 100)}%
				</span>
			</div>

			{audioUnavailable && (
				<span className="text-label-xs text-amber-400 shrink-0">Audio unavailable</span>
			)}

			{isCounting && (
				<span className="text-label-xs text-text-tertiary shrink-0 animate-pulse">Count in...</span>
			)}
		</div>
	);
}

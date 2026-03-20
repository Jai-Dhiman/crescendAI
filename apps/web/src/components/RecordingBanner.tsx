import type { PracticeMode } from "../lib/practice-api";
import { ResonanceRipples } from "./ResonanceRipples";

interface RecordingBannerProps {
	elapsedSeconds: number;
	energy: number;
	isPlaying: boolean;
	onStop: () => void;
	practiceMode: PracticeMode | null;
}

function formatTime(seconds: number): string {
	const m = Math.floor(seconds / 60);
	const s = seconds % 60;
	return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

export function RecordingBanner({
	elapsedSeconds,
	energy,
	isPlaying,
	onStop,
	practiceMode,
}: RecordingBannerProps) {
	return (
		<div className="flex items-center gap-4 px-4 py-2 border-t border-border bg-surface-2">
			<div className="w-[60px] h-[40px] flex-shrink-0">
				<ResonanceRipples energy={energy} isPlaying={isPlaying} active={true} />
			</div>

			<div className="flex items-center gap-3 flex-1 min-w-0">
				<span className="font-mono text-sm text-primary tabular-nums">
					{formatTime(elapsedSeconds)}
				</span>
				{practiceMode && practiceMode !== "regular" && (
					<span className="text-xs px-2 py-0.5 rounded-full bg-surface-3 text-secondary">
						{practiceMode}
					</span>
				)}
			</div>

			<button
				onClick={onStop}
				className="w-9 h-9 rounded-full bg-red-500/15 hover:bg-red-500/25 flex items-center justify-center transition-colors flex-shrink-0"
				aria-label="Stop recording"
			>
				<div className="w-3 h-3 rounded-sm bg-red-500" />
			</button>
		</div>
	);
}

import { useMetronome } from "../hooks/useMetronome";
import type { Mark } from "../lib/mark";
import { SessionTimelineStrip } from "./SessionTimelineStrip";

interface PieceLessModeProps {
	marks: readonly Mark[];
	durationSeconds: number;
	elapsedSeconds: number;
	isRecording: boolean;
}

function formatElapsed(totalSeconds: number): string {
	const minutes = Math.floor(totalSeconds / 60);
	const seconds = Math.floor(totalSeconds % 60);
	return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

/**
 * The permanent pieceless surface (docs/apps/05-ui-system.md#2): a calm,
 * near-empty screen. No score to hide behind means this component has no
 * logic of its own beyond formatting elapsed time — everything else is
 * SessionTimelineStrip, which is the complete canvas by design.
 */
export function PieceLessMode({
	marks,
	durationSeconds,
	elapsedSeconds,
	isRecording,
}: PieceLessModeProps) {
	const metronome = useMetronome();

	return (
		<div className="flex h-full flex-col items-center justify-between px-6 py-12">
			<div className="flex flex-1 flex-col items-center justify-center gap-2">
				{isRecording && (
					<span className="h-2 w-2 rounded-full bg-danger" aria-hidden="true" />
				)}
				<span className="text-display-md tabular-nums text-ink-primary">
					{formatElapsed(elapsedSeconds)}
				</span>
				<button
					type="button"
					onClick={metronome.toggle}
					className="text-label-sm text-ink-tertiary underline"
				>
					{metronome.isPlaying ? `Metronome ${metronome.bpm}` : "Metronome"}
				</button>
			</div>
			<div className="w-full max-w-2xl">
				<SessionTimelineStrip durationSeconds={durationSeconds} marks={marks} />
			</div>
		</div>
	);
}

import { CircleNotch, Stop } from "@phosphor-icons/react";
import { useCallback, useState } from "react";
import type { PracticeState } from "../hooks/usePracticeSession";
import type { ObservationEvent } from "../lib/practice-api";
import { FlowingWaves } from "./FlowingWaves";
import { ObservationToast } from "./ObservationToast";

interface RecordingBarProps {
	state: PracticeState;
	elapsedSeconds: number;
	observations: ObservationEvent[];
	analyserNode: AnalyserNode | null;
	error: string | null;
	chunksProcessed: number;
	onStop: () => void;
}

function formatTime(seconds: number): string {
	const m = Math.floor(seconds / 60);
	const s = seconds % 60;
	return `${m}:${s.toString().padStart(2, "0")}`;
}

export function RecordingBar({
	state,
	elapsedSeconds,
	observations,
	analyserNode,
	error,
	chunksProcessed,
	onStop,
}: RecordingBarProps) {
	const [dismissedIds, setDismissedIds] = useState<Set<number>>(new Set());

	const handleDismiss = useCallback((idx: number) => {
		setDismissedIds((prev) => new Set(prev).add(idx));
	}, []);

	const visibleObservations = observations
		.map((obs, idx) => ({ ...obs, idx }))
		.filter(({ idx }) => !dismissedIds.has(idx))
		.slice(-3);

	const isConnecting = state === "requesting-mic" || state === "connecting";
	const isSummarizing = state === "summarizing";
	const isRecording = state === "recording";

	return (
		<>
			{/* Top bar */}
			<div className="fixed top-0 left-0 right-0 z-50 h-14 bg-espresso/95 backdrop-blur-md border-b border-border animate-bar-slide-down">
				<div className="h-full flex items-center px-5 gap-4">
					{/* Left: status / timer */}
					<div className="shrink-0 flex items-center gap-2 min-w-[100px]">
						{isConnecting && (
							<>
								<CircleNotch size={16} className="animate-spin text-accent" />
								<span className="text-body-sm text-text-secondary">
									Connecting...
								</span>
							</>
						)}

						{isSummarizing && (
							<>
								<CircleNotch size={16} className="animate-spin text-accent" />
								<span className="text-body-sm text-text-secondary">
									Summarizing...
								</span>
							</>
						)}

						{isRecording && (
							<>
								<span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
								<span className="font-display text-body-md text-cream tabular-nums tracking-wide">
									{formatTime(elapsedSeconds)}
								</span>
								{chunksProcessed > 0 && (
									<span className="text-body-xs text-text-tertiary ml-1 hidden sm:inline">
										({chunksProcessed})
									</span>
								)}
							</>
						)}

						{error && (
							<span className="text-body-sm text-red-400 truncate max-w-[200px]">
								{error}
							</span>
						)}
					</div>

					{/* Center: flowing waves */}
					<div className="flex-1 h-full py-1 overflow-hidden">
						<FlowingWaves analyserNode={analyserNode} active={isRecording} />
					</div>

					{/* Right: stop button */}
					<div className="shrink-0">
						{isRecording && (
							<button
								type="button"
								onClick={onStop}
								className="w-9 h-9 flex items-center justify-center rounded-full bg-red-600 hover:bg-red-500 text-on-accent transition-colors"
								aria-label="Stop recording"
							>
								<Stop size={16} weight="fill" />
							</button>
						)}
					</div>
				</div>
			</div>

			{/* Observation toasts -- below the bar */}
			<div className="fixed top-16 right-6 z-50 flex flex-col gap-3">
				{visibleObservations.map(({ idx, text, dimension }) => (
					<ObservationToast
						key={idx}
						text={text}
						dimension={dimension}
						onDismiss={() => handleDismiss(idx)}
					/>
				))}
			</div>
		</>
	);
}

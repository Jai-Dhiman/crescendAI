interface SessionEndedBannerProps {
	onResume: () => void;
}

/**
 * The soft auto-stop state at 60s of silence. This is presentation only:
 * nothing about the recording session, WebSocket, or mic changes underneath
 * it (see spec, "Why the auto-stop is UI-only") — onResume only dismisses
 * this banner and resets the silence clock.
 */
export function SessionEndedBanner({ onResume }: SessionEndedBannerProps) {
	return (
		<div className="flex flex-col items-center justify-center gap-4 rounded-lg border border-border-subtle bg-surface-raised px-6 py-8 text-center">
			<p className="text-body-md text-ink-primary">
				Session ended — keep playing?
			</p>
			<button
				type="button"
				onClick={onResume}
				className="rounded-full bg-accent px-5 py-2 text-body-sm text-on-accent"
			>
				Keep playing
			</button>
		</div>
	);
}

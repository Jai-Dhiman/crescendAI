import type { VerdictState } from "../lib/session-review";

interface VerdictBannerProps {
	verdict: VerdictState;
	onRetry: () => void;
}

/**
 * "Loud, never blank": VerdictState is a closed union, so an unhandled
 * variant here is a TypeScript error, not a silently-missing render.
 */
export function VerdictBanner({ verdict, onRetry }: VerdictBannerProps) {
	switch (verdict.status) {
		case "loading":
			return (
				<output className="text-body-md text-ink-secondary">
					Putting together your review...
				</output>
			);
		case "ready":
			return <p className="text-body-md text-ink-primary">{verdict.text}</p>;
		case "failed":
			return (
				<div
					role="alert"
					className="flex items-center justify-between gap-3 rounded-md border border-danger/40 bg-danger/10 px-3 py-2"
				>
					<p className="text-body-sm text-danger">
						Couldn't put together a review this time.
					</p>
					<button
						type="button"
						onClick={onRetry}
						className="shrink-0 rounded-full border border-border-subtle px-3 py-1 text-body-sm text-ink-primary"
					>
						Try again
					</button>
				</div>
			);
	}
}

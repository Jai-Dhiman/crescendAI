import type { ConfidentGuess } from "../lib/piece-ladder";

interface ConfirmPieceChipProps {
	guess: ConfidentGuess;
	onDismiss: () => void;
}

/**
 * Step 2 of the piece ladder: a confident but unpicked guess, shown as a
 * dismissible banner over whichever practice surface is active. Dismissal is
 * one-way — resolvePieceLadderState never re-shows a dismissed guess.
 */
export function ConfirmPieceChip({ guess, onDismiss }: ConfirmPieceChipProps) {
	return (
		<div className="flex items-center justify-between gap-3 rounded-lg border border-border-subtle bg-surface-raised px-4 py-2">
			<p className="text-body-sm text-ink-primary">
				Looks like <span className="font-medium">{guess.title}</span> — is that
				right?
			</p>
			<button
				type="button"
				onClick={onDismiss}
				className="text-label-sm text-ink-tertiary underline"
				aria-label="Dismiss piece guess"
			>
				Dismiss
			</button>
		</div>
	);
}

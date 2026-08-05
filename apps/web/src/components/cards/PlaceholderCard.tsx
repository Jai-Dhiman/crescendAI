interface PlaceholderCardProps {
	type: string;
}

export function PlaceholderCard({ type }: PlaceholderCardProps) {
	return (
		<div className="bg-surface-raised border border-border-subtle rounded-xl p-4 mt-3">
			<p className="text-body-sm text-ink-tertiary italic">
				{type.replace(/_/g, " ")} (coming soon)
			</p>
		</div>
	);
}

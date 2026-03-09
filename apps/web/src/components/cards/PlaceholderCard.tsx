interface PlaceholderCardProps {
	type: string;
}

export function PlaceholderCard({ type }: PlaceholderCardProps) {
	return (
		<div className="bg-surface-card border border-border rounded-xl p-4 mt-3">
			<p className="text-body-sm text-text-tertiary italic">
				{type.replace(/_/g, " ")} (coming soon)
			</p>
		</div>
	);
}

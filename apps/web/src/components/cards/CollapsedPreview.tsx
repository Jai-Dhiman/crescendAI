import { CaretRight } from "@phosphor-icons/react";

interface CollapsedPreviewProps {
	title: string;
	subtitle: string;
	badge: string;
	onRestore: () => void;
	onExpand: () => void;
}

export function CollapsedPreview({
	title,
	subtitle,
	badge,
	onRestore,
	onExpand,
}: CollapsedPreviewProps) {
	return (
		<div
			role="button"
			tabIndex={0}
			onClick={onRestore}
			onKeyDown={(e) => {
				if (e.key === "Enter" || e.key === " ") {
					e.preventDefault();
					onRestore();
				}
			}}
			className="bg-surface-card border border-border rounded-xl px-3 py-2 mt-3 flex items-center gap-3 cursor-pointer hover:bg-surface transition group"
		>
			{/* Accent bar */}
			<div className="w-1 self-stretch rounded-full bg-accent shrink-0" />

			{/* Content */}
			<div className="flex-1 min-w-0">
				<div className="flex items-center gap-2">
					<span className="text-body-sm font-medium text-cream truncate">
						{title}
					</span>
					<span className="text-body-xs text-text-tertiary shrink-0">
						{badge}
					</span>
				</div>
				<p className="text-body-xs text-text-secondary truncate">
					{subtitle}
				</p>
			</div>

			{/* Expand chevron */}
			<button
				type="button"
				onClick={(e) => {
					e.stopPropagation();
					onExpand();
				}}
				className="shrink-0 w-7 h-7 flex items-center justify-center rounded-md text-text-tertiary hover:text-cream hover:bg-surface-2 transition"
				aria-label="Expand artifact"
			>
				<CaretRight size={14} weight="bold" />
			</button>
		</div>
	);
}

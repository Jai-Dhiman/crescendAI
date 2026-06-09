import { Hand } from "@phosphor-icons/react";
import type { KeyboardGuideConfig } from "../../lib/types";

interface KeyboardGuideCardProps {
	config: KeyboardGuideConfig;
}

const HANDS_LABEL: Record<KeyboardGuideConfig["hands"], string> = {
	left: "Left hand",
	right: "Right hand",
	both: "Both hands",
};

export function KeyboardGuideCard({ config }: KeyboardGuideCardProps) {
	return (
		<div className="bg-surface-card border border-border rounded-xl overflow-hidden mt-3">
			{/* Header */}
			<div className="px-4 pt-4 pb-3 flex items-start justify-between gap-3">
				<h4 className="font-display text-body-md text-text-primary leading-snug min-w-0">
					{config.title}
				</h4>
				<span className="shrink-0 flex items-center gap-1.5 text-label-sm text-text-tertiary uppercase tracking-wider">
					<Hand size={12} weight="bold" />
					{HANDS_LABEL[config.hands]}
				</span>
			</div>

			{/* Divider */}
			<div className="border-t border-border/60" />

			{/* Body */}
			<div className="px-4 py-3 flex flex-col gap-3">
				<p className="text-body-sm text-text-secondary leading-relaxed">
					{config.description}
				</p>
				{config.fingering && (
					<div className="flex flex-col gap-1">
						<span className="text-label-sm text-text-tertiary uppercase tracking-wider">
							Fingering
						</span>
						<code className="text-body-sm text-text-primary font-mono bg-surface/40 rounded-lg px-3 py-2 whitespace-pre-wrap break-words">
							{config.fingering}
						</code>
					</div>
				)}
			</div>
		</div>
	);
}

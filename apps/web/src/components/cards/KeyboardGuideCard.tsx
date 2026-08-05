import { HandPalm } from "@phosphor-icons/react";
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
		<div className="bg-surface-raised border border-border-subtle rounded-xl p-4 mt-3 flex flex-col gap-3">
			<div className="flex items-center justify-between gap-3">
				<h4 className="text-body-sm text-ink-primary font-medium leading-snug min-w-0">
					{config.title}
				</h4>
				<div className="flex items-center gap-1.5 shrink-0">
					<HandPalm size={13} className="text-ink-tertiary" />
					<span className="text-label-sm text-ink-tertiary uppercase tracking-wide">
						{HANDS_LABEL[config.hands]}
					</span>
				</div>
			</div>

			<p className="text-body-sm text-ink-secondary leading-snug whitespace-pre-line">
				{config.description}
			</p>

			{config.fingering && (
				<div className="flex flex-col gap-1">
					<span className="text-label-sm text-ink-tertiary uppercase tracking-wide">
						Fingering
					</span>
					<span className="text-body-sm text-ink-primary font-mono">
						{config.fingering}
					</span>
				</div>
			)}
		</div>
	);
}

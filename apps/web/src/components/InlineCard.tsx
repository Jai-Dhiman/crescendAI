import type { InlineComponent } from "../lib/types";
import { ExerciseSetCard } from "./cards/ExerciseSetCard";
import { PlaceholderCard } from "./cards/PlaceholderCard";
import { ScoreHighlightCard } from "./cards/ScoreHighlightCard";

interface InlineCardProps {
	component: InlineComponent;
	onExpand?: () => void;
	artifactId?: string;
}

export function InlineCard({
	component,
	onExpand,
	artifactId,
}: InlineCardProps) {
	switch (component.type) {
		case "exercise_set":
			return (
				<ExerciseSetCard
					config={component.config}
					onExpand={onExpand}
					artifactId={artifactId}
				/>
			);
		case "score_highlight":
			return (
				<ScoreHighlightCard
					config={component.config}
					onExpand={onExpand}
					artifactId={artifactId}
				/>
			);
		default:
			return <PlaceholderCard type={component.type} />;
	}
}

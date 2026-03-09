import type { InlineComponent } from "../lib/types";
import { ExerciseSetCard } from "./cards/ExerciseSetCard";
import { PlaceholderCard } from "./cards/PlaceholderCard";

interface InlineCardProps {
	component: InlineComponent;
}

export function InlineCard({ component }: InlineCardProps) {
	switch (component.type) {
		case "exercise_set":
			return <ExerciseSetCard config={component.config} />;
		default:
			return <PlaceholderCard type={component.type} />;
	}
}

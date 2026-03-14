import { DIMENSION_COLORS, DIMENSION_LABELS } from "../lib/mock-session";

interface ScoreAnnotationProps {
	dimension: string;
	barRange: [number, number];
	index: number;
	isActive: boolean;
	style: React.CSSProperties;
	onClick: (index: number) => void;
}

export function ScoreAnnotation({
	dimension,
	barRange,
	index,
	isActive,
	style,
	onClick,
}: ScoreAnnotationProps) {
	const color =
		DIMENSION_COLORS[dimension as keyof typeof DIMENSION_COLORS] ?? "#7a9a82";
	const label =
		DIMENSION_LABELS[dimension as keyof typeof DIMENSION_LABELS] ?? dimension;

	return (
		<button
			type="button"
			onClick={() => onClick(index)}
			className={`absolute z-10 flex items-center gap-1 px-2 py-0.5 rounded-full text-label-sm font-medium cursor-pointer transition-all duration-200 ${
				isActive
					? "ring-2 ring-cream scale-110 shadow-lg"
					: "hover:scale-105 hover:shadow-md"
			}`}
			style={{
				...style,
				backgroundColor: color,
				color: "#fdf8f0",
				opacity: isActive ? 1 : 0.85,
			}}
			aria-label={`${label} observation at bars ${barRange[0]}-${barRange[1]}`}
			title={`${label}: bars ${barRange[0]}-${barRange[1]}`}
		>
			<span className="w-1.5 h-1.5 rounded-full bg-cream/60" />
			<span>{label}</span>
		</button>
	);
}

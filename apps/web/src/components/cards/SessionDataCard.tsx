import { DIMENSION_COLORS } from "../../lib/mock-session";
import type {
	SessionDataConfig,
	SessionDataObservationRow,
	SessionDataSessionRow,
} from "../../lib/types";

interface SessionDataCardProps {
	config: SessionDataConfig;
}

const QUERY_LABEL: Record<SessionDataConfig["queryType"], string> = {
	dimension_history: "Dimension history",
	recent_sessions: "Recent sessions",
	session_detail: "Session detail",
};

const DIM_FIELDS: Array<{
	dim: keyof typeof DIMENSION_COLORS;
	field: keyof SessionDataSessionRow;
}> = [
	{ dim: "dynamics", field: "avgDynamics" },
	{ dim: "timing", field: "avgTiming" },
	{ dim: "pedaling", field: "avgPedaling" },
	{ dim: "articulation", field: "avgArticulation" },
	{ dim: "phrasing", field: "avgPhrasing" },
	{ dim: "interpretation", field: "avgInterpretation" },
];

function formatScore(value: number | null | undefined): string {
	return typeof value === "number" && Number.isFinite(value)
		? value.toFixed(1)
		: "—";
}

function formatDate(value: string | null | undefined): string {
	if (!value) return "";
	const d = new Date(value);
	return Number.isNaN(d.getTime()) ? "" : d.toLocaleDateString();
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

function DimensionScores({ row }: { row: SessionDataSessionRow }) {
	return (
		<div className="flex flex-wrap gap-1.5">
			{DIM_FIELDS.map(({ dim, field }) => (
				<span
					key={dim}
					className="flex items-center gap-1 rounded bg-surface px-1.5 py-0.5"
				>
					<span
						className="w-1.5 h-1.5 rounded-full shrink-0"
						style={{ backgroundColor: DIMENSION_COLORS[dim] }}
					/>
					<span className="text-label-sm text-text-tertiary uppercase tracking-wide">
						{dim.slice(0, 3)}
					</span>
					<span className="text-body-xs text-text-primary tabular-nums">
						{formatScore(row[field] as number | null)}
					</span>
				</span>
			))}
		</div>
	);
}

function EmptyState() {
	return (
		<p className="text-body-sm text-text-tertiary italic">
			No session data yet.
		</p>
	);
}

export function SessionDataCard({ config }: SessionDataCardProps) {
	const rows = Array.isArray(config.data) ? config.data : [];

	return (
		<div className="bg-surface-card border border-border rounded-xl p-4 mt-3 flex flex-col gap-3">
			<span className="text-label-sm text-text-tertiary uppercase tracking-wide">
				{QUERY_LABEL[config.queryType]}
			</span>

			{config.queryType === "dimension_history" && (
				<div className="flex flex-col gap-2.5">
					{rows.length === 0 && <EmptyState />}
					{rows.filter(isRecord).map((raw, i) => {
						const row = raw as unknown as SessionDataObservationRow;
						const color =
							DIMENSION_COLORS[
								row.dimension as keyof typeof DIMENSION_COLORS
							] ?? "#7a9a82";
						return (
							<div
								key={row.id ?? `obs-${i}`}
								className="flex items-start gap-3"
							>
								<div className="flex items-center gap-1.5 shrink-0 mt-0.5">
									<span
										className="w-1.5 h-1.5 rounded-full shrink-0"
										style={{ backgroundColor: color }}
									/>
									<span className="text-label-sm text-text-tertiary uppercase tracking-wide">
										{row.dimension}
									</span>
								</div>
								<div className="min-w-0">
									<span className="text-body-xs text-text-tertiary">
										{formatScore(row.dimensionScore)}
										{formatDate(row.createdAt) &&
											` · ${formatDate(row.createdAt)}`}
									</span>
									{row.observationText && (
										<p className="text-body-sm text-text-primary mt-0.5 leading-snug">
											{row.observationText}
										</p>
									)}
								</div>
							</div>
						);
					})}
				</div>
			)}

			{config.queryType === "recent_sessions" && (
				<div className="flex flex-col gap-3">
					{rows.length === 0 && <EmptyState />}
					{rows.filter(isRecord).map((raw, i) => {
						const row = raw as unknown as SessionDataSessionRow;
						return (
							<div
								key={row.id ?? `sess-${i}`}
								className="flex flex-col gap-1.5"
							>
								<span className="text-body-xs text-text-tertiary">
									{formatDate(row.startedAt) || "Session"}
								</span>
								<DimensionScores row={row} />
							</div>
						);
					})}
				</div>
			)}

			{config.queryType === "session_detail" &&
				(isRecord(config.data) ? (
					<div className="flex flex-col gap-1.5">
						<span className="text-body-xs text-text-tertiary">
							{formatDate(
								(config.data as unknown as SessionDataSessionRow).startedAt,
							) || "Session"}
						</span>
						<DimensionScores
							row={config.data as unknown as SessionDataSessionRow}
						/>
					</div>
				) : (
					<EmptyState />
				))}
		</div>
	);
}

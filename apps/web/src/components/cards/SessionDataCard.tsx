import { ChartLine } from "@phosphor-icons/react";
import type {
	SessionDataConfig,
	SessionDataObservationRow,
	SessionDataSessionRow,
} from "../../lib/types";

interface SessionDataCardProps {
	config: SessionDataConfig;
}

const QUERY_TITLE: Record<SessionDataConfig["queryType"], string> = {
	dimension_history: "Dimension history",
	recent_sessions: "Recent sessions",
	session_detail: "Session detail",
};

const SESSION_DIMENSIONS: Array<{
	key: keyof SessionDataSessionRow;
	label: string;
}> = [
	{ key: "avgDynamics", label: "Dynamics" },
	{ key: "avgTiming", label: "Timing" },
	{ key: "avgPedaling", label: "Pedaling" },
	{ key: "avgArticulation", label: "Articulation" },
	{ key: "avgPhrasing", label: "Phrasing" },
	{ key: "avgInterpretation", label: "Interpretation" },
];

function formatScore(value: number | null | undefined): string {
	return value === null || value === undefined ? "—" : value.toFixed(1);
}

function formatDate(iso: string | null | undefined): string {
	if (!iso) return "—";
	const d = new Date(iso);
	return Number.isNaN(d.getTime())
		? "—"
		: d.toLocaleDateString(undefined, {
				month: "short",
				day: "numeric",
				year: "numeric",
			});
}

function EmptyState({ label }: { label: string }) {
	return (
		<p className="px-4 py-3 text-body-sm text-text-tertiary italic">{label}</p>
	);
}

function SessionScores({ session }: { session: SessionDataSessionRow }) {
	return (
		<div className="grid grid-cols-3 gap-x-4 gap-y-2 px-4 py-3">
			{SESSION_DIMENSIONS.map((dim) => (
				<div key={dim.key} className="flex flex-col">
					<span className="text-label-sm text-text-tertiary uppercase tracking-wider">
						{dim.label}
					</span>
					<span className="text-body-sm text-text-primary tabular-nums">
						{formatScore(session[dim.key] as number | null)}
					</span>
				</div>
			))}
		</div>
	);
}

function SessionRow({ session }: { session: SessionDataSessionRow }) {
	return (
		<div className="px-4 py-3 flex flex-col gap-2">
			<span className="text-body-xs text-text-tertiary">
				{formatDate(session.startedAt)}
			</span>
			<div className="flex flex-wrap gap-x-4 gap-y-1">
				{SESSION_DIMENSIONS.map((dim) => (
					<span key={dim.key} className="text-body-xs text-text-secondary">
						<span className="text-text-tertiary">{dim.label}</span>{" "}
						<span className="tabular-nums text-text-primary">
							{formatScore(session[dim.key] as number | null)}
						</span>
					</span>
				))}
			</div>
		</div>
	);
}

function ObservationRow({ obs }: { obs: SessionDataObservationRow }) {
	return (
		<div className="px-4 py-3 flex flex-col gap-1">
			<div className="flex items-center justify-between gap-3">
				<span className="text-label-sm text-text-tertiary uppercase tracking-wider">
					{obs.dimension}
				</span>
				<span className="text-body-xs text-text-secondary tabular-nums shrink-0">
					{formatScore(obs.dimensionScore)} · {formatDate(obs.createdAt)}
				</span>
			</div>
			<p className="text-body-sm text-text-secondary leading-relaxed">
				{obs.observationText}
			</p>
		</div>
	);
}

function Body({ config }: { config: SessionDataConfig }) {
	const { queryType, data } = config;

	if (queryType === "dimension_history") {
		const rows = (data as SessionDataObservationRow[] | null) ?? [];
		if (rows.length === 0) {
			return <EmptyState label="No observations recorded yet." />;
		}
		return (
			<div className="divide-y divide-border/50">
				{rows.map((obs) => (
					<ObservationRow key={obs.id} obs={obs} />
				))}
			</div>
		);
	}

	if (queryType === "recent_sessions") {
		const rows = (data as SessionDataSessionRow[] | null) ?? [];
		if (rows.length === 0) {
			return <EmptyState label="No sessions recorded yet." />;
		}
		return (
			<div className="divide-y divide-border/50">
				{rows.map((session) => (
					<SessionRow key={session.id} session={session} />
				))}
			</div>
		);
	}

	// session_detail
	const session = data as SessionDataSessionRow | null;
	if (!session) {
		return <EmptyState label="Session not found." />;
	}
	return (
		<div>
			<div className="px-4 pt-3 text-body-xs text-text-tertiary">
				{formatDate(session.startedAt)}
			</div>
			<SessionScores session={session} />
		</div>
	);
}

export function SessionDataCard({ config }: SessionDataCardProps) {
	return (
		<div className="bg-surface-card border border-border rounded-xl overflow-hidden mt-3">
			{/* Header */}
			<div className="px-4 pt-4 pb-3 flex items-center gap-2">
				<ChartLine size={14} className="text-text-tertiary shrink-0" />
				<h4 className="font-display text-body-md text-text-primary leading-snug">
					{QUERY_TITLE[config.queryType]}
				</h4>
			</div>

			{/* Divider */}
			<div className="border-t border-border/60" />

			{/* Body */}
			<Body config={config} />
		</div>
	);
}

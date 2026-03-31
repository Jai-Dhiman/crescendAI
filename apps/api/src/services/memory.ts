import { and, eq, isNull, desc } from "drizzle-orm";
import type { ServiceContext } from "../lib/types";
import { synthesizedFacts } from "../db/schema/memory";
import { observations } from "../db/schema/observations";

export async function buildMemoryContext(
	ctx: ServiceContext,
	studentId: string,
): Promise<string> {
	const facts = await ctx.db
		.select({
			factText: synthesizedFacts.factText,
			factType: synthesizedFacts.factType,
			dimension: synthesizedFacts.dimension,
			confidence: synthesizedFacts.confidence,
		})
		.from(synthesizedFacts)
		.where(
			and(
				eq(synthesizedFacts.studentId, studentId),
				isNull(synthesizedFacts.invalidAt),
				isNull(synthesizedFacts.expiredAt),
			),
		)
		.orderBy(desc(synthesizedFacts.validAt))
		.limit(12);

	const recentObs = await ctx.db
		.select({
			dimension: observations.dimension,
			observationText: observations.observationText,
			framing: observations.framing,
		})
		.from(observations)
		.where(eq(observations.studentId, studentId))
		.orderBy(desc(observations.createdAt))
		.limit(5);

	const parts: string[] = [];

	if (facts.length > 0) {
		parts.push("## Known Facts About This Student");
		for (const f of facts) {
			const dim = f.dimension ? ` [${f.dimension}]` : "";
			parts.push(`- (${f.factType}${dim}, ${f.confidence}) ${f.factText}`);
		}
	}

	if (recentObs.length > 0) {
		parts.push("\n## Recent Practice Observations");
		for (const o of recentObs) {
			const framing = o.framing ? ` (${o.framing})` : "";
			parts.push(`- [${o.dimension}${framing}] ${o.observationText}`);
		}
	}

	return parts.join("\n");
}

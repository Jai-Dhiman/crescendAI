import { waitlist } from "../db/schema/catalog";
import type { ServiceContext } from "../lib/types";

export async function addToWaitlist(
	ctx: ServiceContext,
	data: { email: string; context?: string; source?: string },
) {
	await ctx.db
		.insert(waitlist)
		.values({
			email: data.email,
			context: data.context ?? null,
			source: data.source ?? "web",
		})
		.onConflictDoNothing({ target: waitlist.email });

	return { success: true };
}

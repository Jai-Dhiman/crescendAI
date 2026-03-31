import { Sentry } from "./sentry";
import { client } from "./api-client";

const API_BASE = import.meta.env.PROD
	? "https://api.crescend.ai"
	: "http://localhost:8787";

export class ApiError extends Error {
	constructor(
		public status: number,
		message: string,
	) {
		super(message);
		this.name = "ApiError";
	}
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
	const response = await fetch(`${API_BASE}${path}`, {
		...options,
		credentials: "include",
		headers: {
			"Content-Type": "application/json",
			...options.headers,
		},
	});

	if (!response.ok) {
		const body = (await response
			.json()
			.catch(() => ({ error: response.statusText }))) as Record<string, string>;
		const err = new ApiError(response.status, body.error ?? response.statusText);
		Sentry.captureException(err, {
			extra: { status: response.status, path },
		});
		throw err;
	}

	Sentry.addBreadcrumb({
		category: "api",
		message: `${options.method ?? "GET"} ${path}`,
		level: "info",
	});

	return response.json();
}

export interface AuthUser {
	studentId: string;
	email: string | null;
	displayName: string | null;
}

export interface AuthResult {
	studentId: string;
	email: string | null;
	displayName: string | null;
	isNewUser: boolean;
}

// --- Chat types ---

export interface ConversationSummary {
	id: string;
	title: string | null;
	updatedAt: string;
}

export interface MessageRow {
	id: string;
	role: "user" | "assistant";
	content: string;
	createdAt: string;
	messageType?: string;
	dimension?: string;
	framing?: string;
	componentsJson?: string;
	sessionId?: string;
}

export interface ConversationWithMessages {
	conversation: {
		id: string;
		title: string | null;
		createdAt: string;
	};
	messages: MessageRow[];
}

export interface ChatStreamEvent {
	type: "start" | "delta" | "done";
	conversationId?: string;
	messageId?: string;
	text?: string;
}

// --- Exercise types ---

export interface Exercise {
	id: string;
	title: string;
	description: string;
	instructions: string;
	difficulty: string;
	category: string;
	repertoire_tags: string | null;
	source: string;
	dimensions: string[];
}

export interface StudentExercise {
	id: string;
	studentId: string;
	exerciseId: string;
	sessionId: string | null;
	assignedAt: string;
	completed: boolean;
	response: string | null;
	timesAssigned: number;
}

export const api = {
	auth: {
		apple(
			identityToken: string,
			userId: string,
			email?: string,
			displayName?: string,
		): Promise<AuthResult> {
			return request("/api/auth/apple", {
				method: "POST",
				body: JSON.stringify({
					identityToken,
					userId,
					email,
					displayName,
				}),
			});
		},

		me(): Promise<AuthUser> {
			return request("/api/auth/me");
		},

		signout(): Promise<void> {
			return request("/api/auth/signout", { method: "POST" });
		},

		google: (credential: string) =>
			request<AuthResult>(
				"/api/auth/google",
				{
					method: "POST",
					body: JSON.stringify({ credential }),
				},
			),

		debug(): Promise<AuthResult> {
			return request("/api/auth/debug", { method: "POST" });
		},
	},

	chat: {
		async send(
			message: string,
			conversationId: string | null,
			onEvent: (event: ChatStreamEvent) => void,
		): Promise<void> {
			const response = await fetch(`${API_BASE}/api/chat`, {
				method: "POST",
				credentials: "include",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					conversationId,
					message,
				}),
			});

			if (!response.ok) {
				const body = (await response
					.json()
					.catch(() => ({ error: response.statusText }))) as Record<
					string,
					string
				>;
				const err = new ApiError(response.status, body.error ?? response.statusText);
				Sentry.captureException(err, {
					extra: { status: response.status, path: "/api/chat" },
				});
				throw err;
			}

			if (!response.body) throw new Error("Response body is empty");

			const reader = response.body.getReader();
			const decoder = new TextDecoder();
			let lineBuffer = "";

			try {
				while (true) {
					const { done, value } = await reader.read();
					if (done) break;

					lineBuffer += decoder.decode(value, { stream: true });
					const lines = lineBuffer.split("\n");
					// Keep the last (possibly incomplete) line in the buffer
					lineBuffer = lines.pop() ?? "";

					for (const line of lines) {
						if (line.startsWith("data: ")) {
							try {
								const event: ChatStreamEvent = JSON.parse(line.slice(6));
								onEvent(event);
							} catch {
								// Skip unparseable lines
							}
						}
					}
				}

				// Process any remaining data in the buffer
				if (lineBuffer.startsWith("data: ")) {
					try {
						const event: ChatStreamEvent = JSON.parse(lineBuffer.slice(6));
						onEvent(event);
					} catch {
						// Skip unparseable lines
					}
				}
			} finally {
				reader.releaseLock();
			}
		},

		async list(): Promise<{ conversations: ConversationSummary[] }> {
			const res = await client.api.conversations.$get();
			if (!res.ok) {
				const body = (await res.json().catch(() => ({ error: res.statusText }))) as Record<string, string>;
				const err = new ApiError(res.status, body.error ?? res.statusText);
				Sentry.captureException(err, { extra: { status: res.status, path: "/api/conversations" } });
				throw err;
			}
			return res.json() as unknown as Promise<{ conversations: ConversationSummary[] }>;
		},

		async get(conversationId: string): Promise<ConversationWithMessages> {
			const res = await client.api.conversations[":id"].$get({ param: { id: conversationId } } as never);
			if (!res.ok) {
				const body = (await res.json().catch(() => ({ error: res.statusText }))) as Record<string, string>;
				const err = new ApiError(res.status, body.error ?? res.statusText);
				Sentry.captureException(err, { extra: { status: res.status, path: `/api/conversations/${conversationId}` } });
				throw err;
			}
			return res.json() as unknown as Promise<ConversationWithMessages>;
		},

		async delete(conversationId: string): Promise<void> {
			const res = await client.api.conversations[":id"].$delete({ param: { id: conversationId } } as never);
			if (!res.ok) {
				const err = new ApiError(res.status, "Failed to delete conversation");
				Sentry.captureException(err, { extra: { status: res.status, conversationId } });
				throw err;
			}
		},
	},

	exercises: {
		async fetch(params?: {
			dimension?: string;
			level?: string;
			repertoire?: string;
		}): Promise<{ exercises: Exercise[] }> {
			const query: { dimension?: string; level?: string; repertoire?: string } = {};
			if (params?.dimension) query.dimension = params.dimension;
			if (params?.level) query.level = params.level;
			if (params?.repertoire) query.repertoire = params.repertoire;
			const res = await client.api.exercises.$get({ query } as never);
			if (!res.ok) {
				const body = (await res.json().catch(() => ({ error: res.statusText }))) as Record<string, string>;
				const err = new ApiError(res.status, body.error ?? res.statusText);
				Sentry.captureException(err, { extra: { status: res.status, path: "/api/exercises" } });
				throw err;
			}
			return res.json() as unknown as Promise<{ exercises: Exercise[] }>;
		},

		async assign(body: {
			exerciseId: string;
			sessionId?: string;
		}): Promise<StudentExercise> {
			const res = await client.api.exercises.assign.$post({ json: body } as never);
			if (!res.ok) {
				const errBody = (await res.json().catch(() => ({ error: res.statusText }))) as Record<string, string>;
				const err = new ApiError(res.status, errBody.error ?? res.statusText);
				Sentry.captureException(err, { extra: { status: res.status, path: "/api/exercises/assign" } });
				throw err;
			}
			return res.json() as unknown as Promise<StudentExercise>;
		},

		async complete(body: {
			studentExerciseId: string;
			response?: string;
			dimensionBeforeJson?: string;
			dimensionAfterJson?: string;
			notes?: string;
		}): Promise<StudentExercise> {
			const res = await client.api.exercises.complete.$post({ json: body } as never);
			if (!res.ok) {
				const errBody = (await res.json().catch(() => ({ error: res.statusText }))) as Record<string, string>;
				const err = new ApiError(res.status, errBody.error ?? res.statusText);
				Sentry.captureException(err, { extra: { status: res.status, path: "/api/exercises/complete" } });
				throw err;
			}
			return res.json() as unknown as Promise<StudentExercise>;
		},
	},

	waitlist: {
		async join(
			email: string,
			context?: string,
			_name?: string,
		): Promise<{ ok: boolean }> {
			const res = await client.api.waitlist.$post({ json: { email, context } } as never);
			if (!res.ok) {
				const body = (await res.json().catch(() => ({ error: res.statusText }))) as Record<string, string>;
				const err = new ApiError(res.status, body.error ?? res.statusText);
				Sentry.captureException(err, { extra: { status: res.status, path: "/api/waitlist" } });
				throw err;
			}
			return res.json() as unknown as Promise<{ ok: boolean }>;
		},
	},
};

/** Check if any sessions in this conversation need deferred synthesis. */
export async function checkNeedsSynthesis(conversationId: string): Promise<string[]> {
	try {
		const res = await client.api.practice["needs-synthesis"].$get({ query: { conversationId } } as never);
		if (!res.ok) {
			throw new ApiError(res.status, res.statusText);
		}
		const data = await res.json() as unknown as { sessionIds: string[] };
		return data.sessionIds ?? [];
	} catch {
		return [];
	}
}

/** Trigger deferred synthesis for a specific session. */
export async function triggerDeferredSynthesis(sessionId: string): Promise<{ status: string; isFallback?: boolean } | null> {
	try {
		const res = await client.api.practice.synthesize.$post({ json: { sessionId } } as never);
		if (!res.ok) {
			throw new ApiError(res.status, res.statusText);
		}
		return res.json() as unknown as Promise<{ status: string; isFallback?: boolean }>;
	} catch {
		return null;
	}
}

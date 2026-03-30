import { Sentry } from "./sentry";

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

		list(): Promise<{ conversations: ConversationSummary[] }> {
			return request("/api/conversations");
		},

		get(conversationId: string): Promise<ConversationWithMessages> {
			return request(`/api/conversations/${conversationId}`);
		},

		async delete(conversationId: string): Promise<void> {
			const response = await fetch(
				`${API_BASE}/api/conversations/${conversationId}`,
				{
					method: "DELETE",
					credentials: "include",
				},
			);
			if (!response.ok && response.status !== 204) {
				const err = new ApiError(response.status, "Failed to delete conversation");
				Sentry.captureException(err, {
					extra: { status: response.status, conversationId },
				});
				throw err;
			}
		},
	},

	exercises: {
		fetch(params?: {
			dimension?: string;
			level?: string;
			repertoire?: string;
		}): Promise<{ exercises: Exercise[] }> {
			const searchParams = new URLSearchParams();
			if (params?.dimension) searchParams.set("dimension", params.dimension);
			if (params?.level) searchParams.set("level", params.level);
			if (params?.repertoire) searchParams.set("repertoire", params.repertoire);
			const qs = searchParams.toString();
			return request(`/api/exercises${qs ? `?${qs}` : ""}`);
		},

		assign(body: {
			exerciseId: string;
			sessionId?: string;
		}): Promise<StudentExercise> {
			return request("/api/exercises/assign", {
				method: "POST",
				body: JSON.stringify(body),
			});
		},

		complete(body: {
			studentExerciseId: string;
			response?: string;
			dimensionBeforeJson?: string;
			dimensionAfterJson?: string;
			notes?: string;
		}): Promise<StudentExercise> {
			return request("/api/exercises/complete", {
				method: "POST",
				body: JSON.stringify(body),
			});
		},
	},

	waitlist: {
		join(
			email: string,
			context?: string,
			name?: string,
		): Promise<{ ok: boolean }> {
			return request("/api/waitlist", {
				method: "POST",
				body: JSON.stringify({ email, context, name }),
			});
		},
	},
};

/** Check if any sessions in this conversation need deferred synthesis. */
export async function checkNeedsSynthesis(conversationId: string): Promise<string[]> {
	try {
		const data = await request<{ sessionIds: string[] }>(
			`/api/practice/needs-synthesis?conversationId=${encodeURIComponent(conversationId)}`
		);
		return data.sessionIds ?? [];
	} catch {
		return [];
	}
}

/** Trigger deferred synthesis for a specific session. */
export async function triggerDeferredSynthesis(sessionId: string): Promise<{ status: string; isFallback?: boolean } | null> {
	try {
		return await request<{ status: string; isFallback?: boolean }>(
			'/api/practice/synthesize',
			{
				method: 'POST',
				body: JSON.stringify({ sessionId }),
			}
		);
	} catch {
		return null;
	}
}

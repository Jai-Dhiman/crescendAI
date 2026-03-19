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
	student_id: string;
	email: string | null;
	display_name: string | null;
}

export interface AuthResult {
	student_id: string;
	email: string | null;
	display_name: string | null;
	is_new_user: boolean;
}

// --- Chat types ---

export interface ConversationSummary {
	id: string;
	title: string | null;
	updated_at: string;
}

export interface MessageRow {
	id: string;
	role: "user" | "assistant";
	content: string;
	created_at: string;
	components?: import("./types").InlineComponent[];
}

export interface ConversationWithMessages {
	conversation: {
		id: string;
		title: string | null;
		created_at: string;
	};
	messages: MessageRow[];
}

export interface ChatStreamEvent {
	type: "start" | "delta" | "done";
	conversation_id?: string;
	message_id?: string;
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
	student_id: string;
	exercise_id: string;
	session_id: string | null;
	assigned_at: string;
	completed: boolean;
	response: string | null;
	times_assigned: number;
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
					identity_token: identityToken,
					user_id: userId,
					email,
					display_name: displayName,
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
					conversation_id: conversationId,
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
			exercise_id: string;
			session_id?: string;
		}): Promise<StudentExercise> {
			return request("/api/exercises/assign", {
				method: "POST",
				body: JSON.stringify(body),
			});
		},

		complete(body: {
			student_exercise_id: string;
			response?: string;
			dimension_before_json?: string;
			dimension_after_json?: string;
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

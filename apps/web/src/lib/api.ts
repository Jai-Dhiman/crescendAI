import { client } from "./api-client";
import { API_BASE } from "./config";
import { Sentry } from "./sentry";

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
		const err = new ApiError(
			response.status,
			body.error ?? response.statusText,
		);
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
			request<AuthResult>("/api/auth/google", {
				method: "POST",
				body: JSON.stringify({ credential }),
			}),

		debug(): Promise<AuthResult> {
			return request("/api/auth/debug", { method: "POST" });
		},
	},

	waitlist: {
		async join(
			email: string,
			context?: string,
			_name?: string,
		): Promise<{ ok: boolean }> {
			const res = await client.api.waitlist.$post({
				json: { email, context },
			} as never);
			if (!res.ok) {
				const body = (await res
					.json()
					.catch(() => ({ error: res.statusText }))) as Record<string, string>;
				const err = new ApiError(res.status, body.error ?? res.statusText);
				Sentry.captureException(err, {
					extra: { status: res.status, path: "/api/waitlist" },
				});
				throw err;
			}
			return res.json() as unknown as Promise<{ ok: boolean }>;
		},
	},

	sessions: {
		async getPassage(
			sessionId: string,
			bars: [number, number],
		): Promise<import("./types").PassageManifest> {
			return request<import("./types").PassageManifest>(
				`/api/sessions/${sessionId}/passage?bars=${bars[0]}-${bars[1]}`,
			);
		},
	},

	scores: {
		async getData(pieceId: string): Promise<ArrayBuffer> {
			const res = await fetch(`${API_BASE}/api/scores/${pieceId}/data`, {
				credentials: "include",
			});
			if (!res.ok) {
				const err = new ApiError(
					res.status,
					`Failed to fetch score data for ${pieceId}`,
				);
				Sentry.captureException(err, {
					extra: { status: res.status, path: `/api/scores/${pieceId}/data` },
				});
				throw err;
			}
			return res.arrayBuffer();
		},
	},
};

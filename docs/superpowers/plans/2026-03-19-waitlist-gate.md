# Waitlist Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the sign-in page with a beta waitlist that collects emails, controlled by a `VITE_AUTH_MODE` feature flag.

**Architecture:** New D1 `waitlist` table + unauthenticated `POST /api/waitlist` Rust endpoint. The web app's `/signin` route conditionally renders `WaitlistPage` or the existing `SignInPage` based on `VITE_AUTH_MODE`. `AuthProvider` skips the `/api/auth/me` call in waitlist mode.

**Tech Stack:** Rust (Cloudflare Workers), D1 (SQLite), React 19, TanStack Router, Tailwind CSS v4

**Spec:** `docs/superpowers/specs/2026-03-19-waitlist-gate-design.md`

---

### Task 1: D1 Migration

**Files:**
- Create: `apps/api/migrations/0006_waitlist.sql`

- [ ] **Step 1: Write the migration file**

```sql
-- Waitlist for beta signups
CREATE TABLE IF NOT EXISTS waitlist (
    email TEXT PRIMARY KEY,
    context TEXT,
    source TEXT NOT NULL DEFAULT 'web',
    created_at TEXT NOT NULL
);

CREATE INDEX idx_waitlist_created ON waitlist(created_at);
```

- [ ] **Step 2: Apply migration locally**

Run: `cd apps/api && bunx wrangler d1 execute crescendai-db --local --file=migrations/0006_waitlist.sql`
Expected: Completes without errors.

- [ ] **Step 3: Verify table exists**

Run: `cd apps/api && bunx wrangler d1 execute crescendai-db --local --command "SELECT name FROM sqlite_master WHERE type='table' AND name='waitlist';"`
Expected: Returns one row: `waitlist`

- [ ] **Step 4: Commit**

```bash
git add apps/api/migrations/0006_waitlist.sql
git commit -m "feat: add waitlist D1 migration (0006)"
```

---

### Task 2: API Waitlist Endpoint

**Files:**
- Create: `apps/api/src/services/waitlist.rs`
- Modify: `apps/api/src/services/mod.rs:10` (add `pub mod waitlist;`)
- Modify: `apps/api/src/server.rs` (add route before the health check)

- [ ] **Step 1: Write the API test**

Create `apps/api/tests/waitlist.test.ts`:

```typescript
import { describe, test, expect } from "bun:test";

const BASE = "http://localhost:8787";

describe("POST /api/waitlist", () => {
	test("accepts valid email", async () => {
		const res = await fetch(`${BASE}/api/waitlist`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ email: "test@example.com" }),
		});
		expect(res.status).toBe(200);
		const data = (await res.json()) as { ok: boolean };
		expect(data.ok).toBe(true);
	});

	test("accepts email with context", async () => {
		const res = await fetch(`${BASE}/api/waitlist`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				email: "pianist@example.com",
				context: "Working through Chopin Nocturnes, intermediate level",
			}),
		});
		expect(res.status).toBe(200);
		const data = (await res.json()) as { ok: boolean };
		expect(data.ok).toBe(true);
	});

	test("rejects missing email", async () => {
		const res = await fetch(`${BASE}/api/waitlist`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({}),
		});
		expect(res.status).toBe(400);
	});

	test("rejects invalid email format", async () => {
		const res = await fetch(`${BASE}/api/waitlist`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ email: "not-an-email" }),
		});
		expect(res.status).toBe(400);
	});

	test("rejects email without TLD", async () => {
		const res = await fetch(`${BASE}/api/waitlist`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ email: "user@localhost" }),
		});
		expect(res.status).toBe(400);
	});

	test("duplicate email returns 200 (no leak)", async () => {
		// First submission
		await fetch(`${BASE}/api/waitlist`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ email: "dupe@example.com" }),
		});
		// Second submission -- same email
		const res = await fetch(`${BASE}/api/waitlist`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ email: "dupe@example.com" }),
		});
		expect(res.status).toBe(200);
		const data = (await res.json()) as { ok: boolean };
		expect(data.ok).toBe(true);
	});

	test("honeypot field triggers silent accept", async () => {
		const res = await fetch(`${BASE}/api/waitlist`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				email: "bot@spam.com",
				name: "I am a bot",
			}),
		});
		// Returns 200 but does NOT insert into DB
		expect(res.status).toBe(200);
		const data = (await res.json()) as { ok: boolean };
		expect(data.ok).toBe(true);
	});
});
```

- [ ] **Step 2: Run test to verify it fails**

Start the local dev server first (if not already running):
```bash
cd apps/api && bunx wrangler d1 execute crescendai-db --local --file=migrations/0006_waitlist.sql 2>/dev/null; bunx wrangler dev --local --port 8787 &
```

Run: `cd apps/api && bun test tests/waitlist.test.ts`
Expected: FAIL -- 404 Not found (endpoint does not exist yet).

- [ ] **Step 3: Create `waitlist.rs` handler**

Create `apps/api/src/services/waitlist.rs`:

```rust
use js_sys;
use wasm_bindgen::JsValue;
use worker::{console_error, Env};

/// Validate email: non-empty local@domain.tld, no whitespace, 3-254 chars.
fn is_valid_email(email: &str) -> bool {
    if email.len() < 3 || email.len() > 254 {
        return false;
    }
    let parts: Vec<&str> = email.splitn(2, '@').collect();
    if parts.len() != 2 {
        return false;
    }
    let local = parts[0];
    let domain = parts[1];
    if local.is_empty() || domain.is_empty() {
        return false;
    }
    if email.contains(char::is_whitespace) {
        return false;
    }
    // Domain must contain a dot with non-empty TLD
    match domain.rfind('.') {
        Some(dot_pos) => dot_pos > 0 && dot_pos < domain.len() - 1,
        None => false,
    }
}

pub async fn handle_waitlist(
    env: &Env,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    let parsed: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => {
            return http::Response::builder()
                .status(http::StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(r#"{"error":"Invalid JSON"}"#))
                .unwrap();
        }
    };

    // Honeypot: if "name" field is non-empty, silently accept
    if let Some(name) = parsed.get("name").and_then(|v| v.as_str()) {
        if !name.is_empty() {
            return http::Response::builder()
                .status(http::StatusCode::OK)
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(r#"{"ok":true}"#))
                .unwrap();
        }
    }

    let email = match parsed.get("email").and_then(|v| v.as_str()) {
        Some(e) => e.trim().to_lowercase(),
        None => {
            return http::Response::builder()
                .status(http::StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(r#"{"error":"Invalid email"}"#))
                .unwrap();
        }
    };

    if !is_valid_email(&email) {
        return http::Response::builder()
            .status(http::StatusCode::BAD_REQUEST)
            .header("Content-Type", "application/json")
            .body(axum::body::Body::from(r#"{"error":"Invalid email"}"#))
            .unwrap();
    }

    // Truncate context to 500 chars (char-boundary safe)
    let context: Option<String> = parsed
        .get("context")
        .and_then(|v| v.as_str())
        .map(|s| {
            let trimmed = s.trim();
            if trimmed.chars().count() > 500 {
                trimmed.chars().take(500).collect::<String>()
            } else {
                trimmed.to_string()
            }
        })
        .filter(|s| !s.is_empty());

    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_error!("D1 binding failed: {:?}", e);
            return http::Response::builder()
                .status(http::StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(r#"{"error":"Internal error"}"#))
                .unwrap();
        }
    };

    let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();

    let context_val = match &context {
        Some(c) => JsValue::from_str(c),
        None => JsValue::NULL,
    };

    let stmt = match db
        .prepare("INSERT OR IGNORE INTO waitlist (email, context, source, created_at) VALUES (?1, ?2, ?3, ?4)")
        .bind(&[
            JsValue::from_str(&email),
            context_val,
            JsValue::from_str("web"),
            JsValue::from_str(&now),
        ]) {
        Ok(stmt) => stmt,
        Err(e) => {
            console_error!("Waitlist bind failed: {:?}", e);
            return http::Response::builder()
                .status(http::StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(r#"{"error":"Internal error"}"#))
                .unwrap();
        }
    };

    if let Err(e) = stmt.run().await {
        console_error!("Waitlist insert failed: {:?}", e);
        return http::Response::builder()
            .status(http::StatusCode::INTERNAL_SERVER_ERROR)
            .header("Content-Type", "application/json")
            .body(axum::body::Body::from(r#"{"error":"Internal error"}"#))
            .unwrap();
    }

    http::Response::builder()
        .status(http::StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(axum::body::Body::from(r#"{"ok":true}"#))
        .unwrap()
}
```

- [ ] **Step 4: Register module in `services/mod.rs`**

Add after line 10 (`pub mod sync;`):
```rust
pub mod waitlist;
```

- [ ] **Step 5: Add route in `server.rs`**

Add this block before the health check (before `if path == "/health"`):

```rust
    // Waitlist signup (unauthenticated)
    if path == "/api/waitlist" && method == http::Method::POST {
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return into_worker_response(with_cors(
            crate::services::waitlist::handle_waitlist(&env, &body).await,
            origin.as_deref(),
        )).await;
    }
```

- [ ] **Step 6: Build and verify compilation**

Run: `cd apps/api && bunx wrangler deploy --dry-run`
Expected: Compiles without errors (dry-run does not deploy).

- [ ] **Step 7: Run tests**

Run: `cd apps/api && bun test tests/waitlist.test.ts`
Expected: All 7 tests PASS.

- [ ] **Step 8: Update `tests/run.sh` to discover all test files**

In `apps/api/tests/run.sh`, replace line 37:

```bash
bun test tests/exercises.test.ts
```

with:

```bash
bun test tests/
```

This ensures all test files in the directory are discovered automatically.

- [ ] **Step 9: Commit**

```bash
git add apps/api/src/services/waitlist.rs apps/api/src/services/mod.rs apps/api/src/server.rs apps/api/tests/waitlist.test.ts apps/api/tests/run.sh
git commit -m "feat: add POST /api/waitlist endpoint with honeypot + email validation"
```

---

### Task 3: Feature Flag + AuthProvider Update

**Files:**
- Modify: `apps/web/.env` (add `VITE_AUTH_MODE=waitlist`)
- Modify: `apps/web/src/lib/auth.tsx:25-37` (skip auth check in waitlist mode)

- [ ] **Step 1: Add env var to `.env`**

Add to the end of `apps/web/.env`:
```
VITE_AUTH_MODE=waitlist
```

- [ ] **Step 2: Update AuthProvider to skip auth check in waitlist mode**

In `apps/web/src/lib/auth.tsx`, replace the `useEffect` (lines 25-38):

```tsx
	useEffect(() => {
		// In waitlist mode, nobody can authenticate -- skip the API call
		// to avoid 401 Sentry noise on every visitor page load.
		if (import.meta.env.VITE_AUTH_MODE !== "live") {
			setIsLoading(false);
			return;
		}

		api.auth
			.me()
			.then(setUser)
			.catch((err) => {
				if (err instanceof ApiError && err.status === 401) {
					setUser(null);
				} else {
					console.error("Auth check failed:", err);
					setUser(null);
				}
			})
			.finally(() => setIsLoading(false));
	}, []);
```

- [ ] **Step 3: Verify dev server starts**

Run: `cd apps/web && bun run dev`
Expected: Starts without errors. The landing page loads. Navigating to `/signin` should still show the old sign-in page (we haven't added the conditional render yet).

- [ ] **Step 4: Commit**

```bash
git add apps/web/.env apps/web/src/lib/auth.tsx
git commit -m "feat: add VITE_AUTH_MODE flag, skip auth check in waitlist mode"
```

---

### Task 4: API Client Method

**Files:**
- Modify: `apps/web/src/lib/api.ts` (add `waitlist` namespace)

- [ ] **Step 1: Add waitlist API method**

In `apps/web/src/lib/api.ts`, add a new `waitlist` namespace after the `exercises` object (before the closing `};` of the `api` const, after line 292):

```typescript
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
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd apps/web && bunx tsc --noEmit`
Expected: No type errors.

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/lib/api.ts
git commit -m "feat: add api.waitlist.join() client method"
```

---

### Task 5: Waitlist Page UI + Conditional Render

**Files:**
- Modify: `apps/web/src/routes/signin.tsx` (add WaitlistPage, conditional render)

This task uses the `/frontend-design` skill for the WaitlistPage component to ensure polished, non-generic design. The component lives inline in `signin.tsx` alongside the existing `SignInPage` (same pattern -- all sign-in route components in one file).

- [ ] **Step 1: Build WaitlistPage component using `/frontend-design`**

Invoke the `/frontend-design` skill with these constraints:

- Same visual shell as `SignInPage`: full-bleed `Image5.jpg` background, radial gradient overlay, centered card (`bg-surface/80 backdrop-blur-xl border border-border rounded-2xl`, `max-w-sm`, `px-8 py-14`)
- Same `fade-in-up` animation (use inline style: `animation: "fade-in-up 600ms cubic-bezier(0.16, 1, 0.3, 1) both"`)
- Card content: wordmark, tagline, hook text, email input, hidden honeypot, optional textarea, submit button, footer text
- States: default form, submitting (disabled button "Joining..."), success (fade to "You're on the list"), error (red inline message)
- Uses `api.waitlist.join()` for submission
- Design tokens from existing codebase: `font-display`, `text-display-sm`, `text-cream`, `text-body-md`, `text-body-sm`, `text-body-xs`, `text-text-secondary`, `text-text-tertiary`, `bg-accent`, `bg-surface`, `border-border`
- Success state includes a link back to `/` (landing page)
- Add the component function ABOVE the existing `SignInPage` function in the same file
- Add necessary imports at the top (`useState` is already imported)

- [ ] **Step 2: Add conditional render at the route level**

In `apps/web/src/routes/signin.tsx`, replace line 6:

```tsx
export const Route = createFileRoute("/signin")({ component: SignInPage });
```

with:

```tsx
const isWaitlistMode = import.meta.env.VITE_AUTH_MODE !== "live";

export const Route = createFileRoute("/signin")({
	component: isWaitlistMode ? WaitlistPage : SignInPage,
});
```

- [ ] **Step 3: Verify the page renders in dev**

Run: `cd apps/web && bun run dev`
Navigate to `http://localhost:3000/signin`
Expected: The waitlist page renders with the email form (not the sign-in buttons). The background image and card styling match the sign-in page.

- [ ] **Step 4: Test form submission manually (requires API dev server on port 8787)**

1. Enter an email and click "Join the Waitlist" (requires API dev server running on port 8787)
2. Expected: Success state shows "You're on the list"
3. Try submitting without email -- expected: validation prevents submission (HTML `required` + type `email`)
4. Verify the landing page "Start Practicing" CTA still links to `/signin` and shows the waitlist

- [ ] **Step 5: Test feature flag toggle**

Change `VITE_AUTH_MODE=live` in `.env`, restart dev server.
Expected: `/signin` shows the original Apple/Google sign-in page.
Change back to `VITE_AUTH_MODE=waitlist` when done.

- [ ] **Step 6: Commit**

```bash
git add apps/web/src/routes/signin.tsx
git commit -m "feat: add WaitlistPage with feature flag toggle on /signin"
```

---

### Task 6: Integration Test + Final Verification

- [ ] **Step 1: Run the full API test suite**

```bash
cd apps/api && bash tests/run.sh
```
Expected: All existing tests (exercises) + new waitlist tests pass.

- [ ] **Step 2: Run TypeScript type check on web app**

Run: `cd apps/web && bunx tsc --noEmit`
Expected: No type errors.

- [ ] **Step 3: Build the web app**

Run: `cd apps/web && bun run build`
Expected: Build succeeds. No warnings about missing env vars.

- [ ] **Step 4: Verify route tree regeneration**

The TanStack Router auto-generates `routeTree.gen.ts`. Since we only modified an existing route (not added/removed one), the generated file should be unchanged. Verify:

Run: `git diff apps/web/src/routeTree.gen.ts`
Expected: No changes (or minor auto-generated timestamp changes only).

- [ ] **Step 5: Final commit if any cleanup needed**

Only if prior steps produced changes that need committing.

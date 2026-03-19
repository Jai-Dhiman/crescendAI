# Waitlist Gate Design

Replace the sign-in page with a beta waitlist email capture, controlled by a feature flag so real auth can be restored with one env var change.

## Decisions

- **Gate location:** Replace `/signin` content only. Landing page (`/`) unchanged; its CTAs continue pointing to `/signin`.
- **Collected data:** Email (required) + optional free-text context ("what do you play/practice?").
- **Feature flag:** Single `VITE_AUTH_MODE` env var (`"waitlist"` | `"live"`). Frontend-only; the API waitlist endpoint is always available.
- **Storage:** D1 table via new API endpoint. Fits the existing stack (same DB, same route pattern, same CORS).

## Data Model

Migration `0006_waitlist.sql`:

```sql
CREATE TABLE IF NOT EXISTS waitlist (
    email TEXT PRIMARY KEY,
    context TEXT,
    source TEXT NOT NULL DEFAULT 'web',
    created_at TEXT NOT NULL
);

CREATE INDEX idx_waitlist_created ON waitlist(created_at);
```

- `email` as PK gives natural deduplication via `INSERT OR IGNORE`.
- `context` stores the optional "what do you play" answer.
- `source` allows future differentiation (web, iOS, referral). Defaults to `"web"`.
- Index on `created_at` for admin queries (`wrangler d1 execute` to list recent signups).

## API Endpoint

`POST /api/waitlist` -- unauthenticated.

**Request:**
```json
{ "email": "user@example.com", "context": "Working through Chopin Nocturnes" }
```

**Validation:**
- Server-side email format check: regex `^[^\s@]+@[^\s@]+\.[^\s@]+$`, length 3-254 chars.
- `context` truncated to 500 chars if provided.
- Honeypot field: if request body contains non-empty `name` field, return `200 { "ok": true }` silently (bots auto-fill hidden fields).

**Response:**
- `200 { "ok": true }` -- same response whether new or duplicate (do not leak membership).
- `400 { "error": "Invalid email" }` -- malformed input.

**Rate limiting:** Cloudflare Rate Limiting rule on `POST /api/waitlist`: 10 requests/minute per IP. Configured in Cloudflare dashboard (no code needed). The existing CORS preflight handler in `server.rs` covers OPTIONS for this route.

## Feature Flag

`VITE_AUTH_MODE` in `apps/web/.env`:
- `"waitlist"` (default) -- `/signin` renders `WaitlistPage`.
- `"live"` -- `/signin` renders the existing `SignInPage` with Apple/Google auth.

The route component checks `import.meta.env.VITE_AUTH_MODE !== "live"` -- this means undefined/missing defaults to waitlist mode (safe default: gated state).

The existing sign-in code is preserved intact in `signin.tsx`.

## Route Protection & AuthProvider

In waitlist mode:
- `AuthProvider` skips the `api.auth.me()` call when `VITE_AUTH_MODE !== "live"`. This prevents Sentry noise from 401 errors on every waitlist visitor page load.
- `user` is always `null`, so `/app/*` routes are naturally inaccessible.
- Landing page CTAs point to `/signin` as before.

In live mode:
- `AuthProvider` behaves exactly as it does today.

## Waitlist Page UI

Same visual shell as the current sign-in page:
- Full-bleed background image (`Image5.jpg`)
- Radial gradient overlay
- Centered card with `bg-surface/80 backdrop-blur-xl border border-border rounded-2xl`
- Same `fade-in-up 600ms` entrance animation

**Card content (top to bottom):**
1. "crescend" wordmark (`font-display text-display-sm text-cream`)
2. "A teacher for every pianist." (`text-body-md text-text-secondary`)
3. Hook: "We're building something new. Join the beta waitlist to be first in."
4. Email input (white bg, dark text, rounded-lg, full-width)
5. Hidden honeypot field (`name=""`, `display: none`) for bot filtering
6. Optional textarea: "What do you play or practice?" (2-3 rows, placeholder: "e.g., Working through Chopin Nocturnes, intermediate level")
7. Submit button: "Join the Waitlist" (`bg-accent text-cream`, full-width)
8. Footer: "We'll only email you about beta access." (`text-body-xs text-text-tertiary`)

**States:**
- Default: form visible
- Submitting: button shows "Joining...", disabled
- Success: form replaced with "You're on the list." + "We'll reach out when your spot is ready." (fade transition). Link back to landing page.
- Error: inline message above button (`text-red-400`)

## Files Changed

| File | Change |
|------|--------|
| `apps/api/migrations/0006_waitlist.sql` | New migration: `waitlist` table + index |
| `apps/api/src/server.rs` | Add `POST /api/waitlist` route (unauthenticated) |
| `apps/api/src/services/waitlist.rs` | New handler: honeypot check, validate email, insert into D1 |
| `apps/api/src/services/mod.rs` | Add `pub mod waitlist;` |
| `apps/web/.env` | Add `VITE_AUTH_MODE=waitlist` |
| `apps/web/src/routes/signin.tsx` | Add `WaitlistPage` component; conditional render based on flag |
| `apps/web/src/lib/auth.tsx` | Skip `api.auth.me()` when `VITE_AUTH_MODE !== "live"` |
| `apps/web/src/lib/api.ts` | Add `api.waitlist.join(email, context?, name?)` method |

## Switching to Live Auth

1. Change `VITE_AUTH_MODE=live` in `.env`
2. Redeploy web app
3. Done. The API waitlist endpoint remains available (harmless). The sign-in page shows Apple/Google auth again. AuthProvider resumes checking `/api/auth/me`.

## Admin Access

Query waitlist signups via `wrangler d1 execute`:
```bash
wrangler d1 execute crescendai-db --command "SELECT * FROM waitlist ORDER BY created_at DESC LIMIT 50;"
```

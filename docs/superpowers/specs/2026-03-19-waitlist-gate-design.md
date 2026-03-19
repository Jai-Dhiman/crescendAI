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
    source TEXT,
    created_at TEXT NOT NULL
);
```

- `email` as PK gives natural deduplication via `INSERT OR IGNORE`.
- `context` stores the optional "what do you play" answer.
- `source` allows future differentiation (web, iOS, referral). Hardcoded to `"web"` for now.

## API Endpoint

`POST /api/waitlist` -- unauthenticated.

**Request:**
```json
{ "email": "user@example.com", "context": "Working through Chopin Nocturnes" }
```

**Validation:**
- Server-side email format check (contains `@`, non-empty local and domain parts).
- `context` truncated to 500 chars if provided.

**Response:**
- `200 { "ok": true }` -- same response whether new or duplicate (do not leak membership).
- `400 { "error": "Invalid email" }` -- malformed input.

## Feature Flag

`VITE_AUTH_MODE` in `apps/web/.env`:
- `"waitlist"` (default) -- `/signin` renders `WaitlistPage`.
- `"live"` -- `/signin` renders the existing `SignInPage` with Apple/Google auth.

The route component (`routes/signin.tsx`) checks `import.meta.env.VITE_AUTH_MODE` and conditionally renders. The existing sign-in code is preserved intact.

## Route Protection

No changes needed. In waitlist mode:
- `AuthProvider` always resolves to `user: null` (nobody can authenticate).
- `/app/*` routes are naturally inaccessible.
- Landing page CTAs point to `/signin` as before.

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
5. Optional textarea: "What do you play or practice?" (2-3 rows, placeholder: "e.g., Working through Chopin Nocturnes, intermediate level")
6. Submit button: "Join the Waitlist" (`bg-accent text-cream`, full-width)
7. Footer: "We'll only email you about beta access." (`text-body-xs text-text-tertiary`)

**States:**
- Default: form visible
- Submitting: button shows "Joining...", disabled
- Success: form replaced with "You're on the list." + "We'll reach out when your spot is ready." (fade transition)
- Error: inline message above button (`text-red-400`)

## Files Changed

| File | Change |
|------|--------|
| `apps/api/migrations/0006_waitlist.sql` | New migration |
| `apps/api/src/server.rs` | Add `POST /api/waitlist` route |
| `apps/api/src/services/waitlist.rs` | New handler: validate email, insert into D1 |
| `apps/api/src/services/mod.rs` | Add `pub mod waitlist;` |
| `apps/web/.env` | Add `VITE_AUTH_MODE=waitlist` |
| `apps/web/src/routes/signin.tsx` | Conditional render: WaitlistPage or SignInPage based on flag |
| `apps/web/src/lib/api.ts` | Add `api.waitlist.join(email, context?)` method |

## Switching to Live Auth

1. Change `VITE_AUTH_MODE=live` in `.env`
2. Redeploy web app
3. Done. The API waitlist endpoint remains available (harmless). The sign-in page shows Apple/Google auth again.

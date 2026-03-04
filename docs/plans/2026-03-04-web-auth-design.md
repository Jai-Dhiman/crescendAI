# Web App Authentication Design

Status: APPROVED
Date: 2026-03-04

## Summary

Production-grade Sign in with Apple authentication for the crescend.ai web app, using Apple's JS SDK with popup flow, HttpOnly cookies for JWT storage, and a provider-agnostic schema to support future auth methods (Google, passkeys).

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Auth scope | Full practice app | Web app will support real practice sessions |
| Token storage | HttpOnly cookie | XSS-safe, automatic attachment, SameSite protection |
| Apple integration | Apple JS SDK (popup) | Official SDK, handles edge cases, returns identity token directly |
| Auth state check | GET /api/auth/me | Single source of truth, works with SSR, no duplicate logic |
| Schema strategy | Provider-agnostic (student_id + auth_identities) | Build for Apple now, extensible to Google/passkeys later |

## Apple Developer Portal (DONE)

- App ID: `ai.crescend.ios` (registered, Sign in with Apple enabled)
- Services ID: `ai.crescend.web` (configured, linked to App ID)
- Team ID: `J2C869U2JJ`
- Domain: `crescend.ai` registered in Services ID config
- Return URL: `https://crescend.ai/signin`
- No domain verification file needed (that's Apple Pay, not Sign in with Apple)
- Key creation and email sources: optional, not needed for popup flow

## Auth Flow

```
User clicks "Sign in with Apple" on crescend.ai/signin
  |
  v
Apple JS SDK opens popup (appleid.apple.com)
  - User authenticates with Apple ID / Face ID / Touch ID
  - Apple returns: identity_token (JWT) + user object (name, email)
  |
  v
Web app POSTs to api.crescend.ai/api/auth/apple
  Body: { identity_token, user_id, email }
  credentials: 'include'
  |
  v
API Worker (auth/mod.rs):
  1. Parse & validate Apple token claims (issuer, audience, subject, exp)
  2. Look up auth_identities for ('apple', user_id)
  3. If new: create student (UUID) + auth_identity row
  4. If existing: update email/timestamp
  5. Issue JWT with student_id as sub (30-day expiry)
  6. Set HttpOnly cookie: token=<JWT>; Secure; SameSite=None; Path=/; Max-Age=2592000
  7. Return { student_id, email, display_name, is_new_user }
  |
  v
Web app receives 200:
  - Cookie stored automatically by browser
  - Store user info in React auth context
  - Navigate to /app
  |
  v
On every page load / route change:
  GET api.crescend.ai/api/auth/me
  - Cookie sent automatically
  - API validates JWT, returns user profile
  - If 401: redirect to /signin
```

## D1 Schema Changes

Replace the current `apple_user_id`-keyed students table:

```sql
CREATE TABLE students (
    student_id TEXT PRIMARY KEY,
    email TEXT,
    display_name TEXT,
    inferred_level TEXT,
    baseline_dynamics REAL,
    baseline_timing REAL,
    baseline_pedaling REAL,
    baseline_articulation REAL,
    baseline_phrasing REAL,
    baseline_interpretation REAL,
    baseline_session_count INTEGER DEFAULT 0,
    explicit_goals TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE auth_identities (
    provider TEXT NOT NULL,
    provider_user_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (provider, provider_user_id),
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

CREATE INDEX idx_auth_identities_student ON auth_identities(student_id);
```

- `student_id`: UUID generated server-side, used as JWT subject
- `auth_identities`: lookup table mapping provider credentials to internal student ID
- One student can have multiple auth methods (Apple now, Google/passkeys later)
- No migration needed (no production users yet)

## API Changes (Rust Worker)

### Modified: POST /api/auth/apple
- Query `auth_identities` for `('apple', user_id)` instead of `students` directly
- On new user: generate UUID, insert into `students` and `auth_identities`
- JWT `sub` claim becomes `student_id` (UUID) instead of `apple_user_id`
- Set `Set-Cookie` header with HttpOnly JWT
- Response body returns user info (no JWT in body)

### Modified: verify_auth_header (middleware)
- Check for JWT in cookie first, then fall back to Authorization header
- Supports both web (cookie) and iOS (Bearer header)

### New: GET /api/auth/me
- Read JWT from cookie or Authorization header
- Validate JWT, query students table by student_id
- Return `{ student_id, email, display_name }` on 200
- Return 401 on invalid/expired/missing token

### New: POST /api/auth/signout
- Clear auth cookie: `Set-Cookie: token=; HttpOnly; Secure; SameSite=None; Path=/; Max-Age=0`
- Return 200

### Modified: CORS (server.rs)
- Add `Access-Control-Allow-Credentials: true`
- Set `Access-Control-Allow-Origin: https://crescend.ai` (exact origin, not wildcard)
- Add `cookie` to `Access-Control-Allow-Headers`

### Audience validation update
- Accept both `ai.crescend.ios` (iOS) and `ai.crescend.web` (web) as valid audiences
- Read allowed audiences from env/config

## Web App Changes

### Rewrite: lib/auth.ts -> AuthProvider (React context)
- `AuthProvider` wraps the app, calls `GET /api/auth/me` on mount
- Exposes: `{ user, isLoading, isAuthenticated, signOut }`
- `user`: `{ studentId, email, displayName } | null`
- `signOut`: calls `POST /api/auth/signout`, clears state, navigates to `/`
- No token handling in JS -- cookies are invisible to client

### Rewrite: routes/signin.tsx
- Load Apple JS SDK (`https://appleid.cdn-apple.com/appleauth/static/jsapi/appleid/1/en_US/appleid.auth.js`)
- Initialize: `AppleID.auth.init({ clientId: 'ai.crescend.web', scope: 'name email', redirectURI: 'https://crescend.ai/signin', usePopup: true })`
- On click: `AppleID.auth.signIn()` -> POST identity token to API -> navigate to /app
- Error handling: display error message on signin page

### Update: routes/app.tsx
- Replace `getAuth()` localStorage check with auth context
- `beforeLoad` uses auth context / `/api/auth/me` for guard

### New: lib/api.ts
- Thin fetch wrapper: always sets `credentials: 'include'`, base URL `https://api.crescend.ai`
- Global 401 handler: redirect to `/signin`

## iOS Impact

- JWT `sub` changes from `apple_user_id` to `student_id` (UUID)
- iOS re-authenticates on launch, so it will get a new JWT with the UUID subject automatically
- No breaking change: Authorization header flow is preserved
- Keychain stores the same JWT format, just different subject value

## Future Extensibility

Adding a new provider (e.g., Google) requires:
1. New endpoint: `POST /api/auth/google` (validates Google token)
2. Same flow: look up `auth_identities` for `('google', google_user_id)`
3. Same JWT issuance with `student_id` as subject
4. New button on signin page

No schema changes, no JWT format changes, no middleware changes.

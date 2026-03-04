# Web App Screens Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the sign-in page, main signed-in screen (`/app`), auth guard, and clean up stale routes.

**Architecture:** TanStack Router file-based routing with a client-side auth stub (localStorage). The `/app` route uses a dedicated layout (no header/footer from root). Sign-in page is a full-bleed image with floating card. Main app screen has a thin icon sidebar + centered chat area inspired by Gemini.

**Tech Stack:** TanStack Start/Router, React 19, Tailwind CSS v4, lucide-react (icons), existing espresso/cream design tokens.

---

### Task 1: Delete `/analyze` route and update landing page links

**Files:**
- Delete: `apps/web/src/routes/analyze.tsx`
- Modify: `apps/web/src/routes/index.tsx:44-49` (hero CTA href)
- Modify: `apps/web/src/routes/index.tsx:152-154` (final CTA href)

**Step 1: Delete analyze.tsx**

```bash
rm apps/web/src/routes/analyze.tsx
```

**Step 2: Update hero CTA in index.tsx**

Change `href="/analyze"` to `href="/app"` on line 45.

**Step 3: Update final CTA in index.tsx**

Change `href="/analyze"` to `href="/app"` on line 153.

**Step 4: Verify the dev server regenerates routeTree.gen.ts**

Run: `cd apps/web && bun run dev`
Expected: No `/analyze` in the generated route tree. No build errors.

**Step 5: Commit**

```bash
git add -A apps/web/src/routes/analyze.tsx apps/web/src/routes/index.tsx
git commit -m "remove /analyze route, point CTAs to /app"
```

---

### Task 2: Create auth stub utility

**Files:**
- Create: `apps/web/src/lib/auth.ts`

**Step 1: Create the auth stub**

```typescript
// Stubbed auth utilities. Replace with real Sign in with Apple later.

const AUTH_KEY = 'crescend_auth'

export interface AuthUser {
  name: string
}

export function getAuth(): AuthUser | null {
  if (typeof window === 'undefined') return null
  const stored = localStorage.getItem(AUTH_KEY)
  if (!stored) return null
  try {
    return JSON.parse(stored) as AuthUser
  } catch {
    return null
  }
}

export function setAuth(user: AuthUser): void {
  localStorage.setItem(AUTH_KEY, JSON.stringify(user))
}

export function clearAuth(): void {
  localStorage.removeItem(AUTH_KEY)
}

export function isAuthenticated(): boolean {
  return getAuth() !== null
}
```

**Step 2: Commit**

```bash
git add apps/web/src/lib/auth.ts
git commit -m "add stubbed auth utility (localStorage)"
```

---

### Task 3: Build the sign-in page

**Files:**
- Modify: `apps/web/src/routes/signin.tsx` (replace stub entirely)

**Step 1: Implement the sign-in page**

Replace the full contents of `signin.tsx`:

```tsx
import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { setAuth } from '../lib/auth'

export const Route = createFileRoute('/signin')({ component: SignInPage })

function SignInPage() {
  const navigate = useNavigate()

  function handleSignIn() {
    // Stub: simulate successful Apple Sign In
    setAuth({ name: 'Jai' })
    navigate({ to: '/app' })
  }

  return (
    <div className="relative h-screen w-full overflow-hidden">
      {/* Full-bleed background image */}
      <img
        src="/Image4.jpg"
        alt="Hands playing piano in warm light"
        className="absolute inset-0 w-full h-full object-cover"
      />

      {/* Radial gradient overlay -- darkens edges, draws focus to center */}
      <div
        className="absolute inset-0"
        style={{
          background:
            'radial-gradient(ellipse at center, rgba(45,41,38,0.4) 0%, rgba(45,41,38,0.85) 100%)',
        }}
      />

      {/* Floating sign-in card */}
      <div className="relative z-10 flex items-center justify-center h-full px-6">
        <div
          className="w-full max-w-md bg-surface/80 backdrop-blur-xl border border-border p-10 text-center"
          style={{ animation: 'fade-in-up 600ms cubic-bezier(0.16, 1, 0.3, 1) both' }}
        >
          <h1 className="font-display text-display-sm text-cream">crescend</h1>

          <p className="mt-3 text-body-md text-text-secondary">
            A teacher for every pianist.
          </p>

          <button
            type="button"
            onClick={handleSignIn}
            className="mt-8 w-full bg-white text-black px-6 py-3 text-body-sm font-medium flex items-center justify-center gap-3 hover:bg-white/90 transition"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
              <path d="M17.05 20.28c-.98.95-2.05.88-3.08.4-1.09-.5-2.08-.48-3.24 0-1.44.62-2.2.44-3.06-.4C2.79 15.25 3.51 7.59 9.05 7.31c1.35.07 2.29.74 3.08.8 1.18-.24 2.31-.93 3.57-.84 1.51.12 2.65.72 3.4 1.8-3.12 1.87-2.38 5.98.48 7.13-.57 1.5-1.31 2.99-2.54 4.09zM12.03 7.25c-.15-2.23 1.66-4.07 3.74-4.25.29 2.58-2.34 4.5-3.74 4.25z" />
            </svg>
            Sign in with Apple
          </button>

          <p className="mt-6 text-body-xs text-text-tertiary">
            By signing in, you agree to our Terms of Service
          </p>
        </div>
      </div>
    </div>
  )
}
```

**Step 2: Run dev server and verify visually**

Run: `cd apps/web && bun run dev`
Navigate to `http://localhost:3000/signin`
Expected: Full-bleed Image4.jpg, centered floating card with backdrop blur, Apple sign-in button. Clicking it should redirect to `/app` (which won't exist yet -- that's fine).

**Step 3: Commit**

```bash
git add apps/web/src/routes/signin.tsx
git commit -m "build sign-in page with photography-forward design"
```

---

### Task 4: Conditionally hide header/footer on `/signin` and `/app`

**Files:**
- Modify: `apps/web/src/routes/__root.tsx:36-51` (RootDocument component)

**Step 1: Update RootDocument to conditionally render Header/Footer**

The root layout currently always renders Header and Footer. We need to hide them on `/signin` and `/app`. Use `useMatch` or `useRouterState` to check the current path.

```tsx
// Add to imports at top:
import { useRouterState } from '@tanstack/react-router'

// Replace RootDocument function:
function RootDocument() {
  const pathname = useRouterState({ select: (s) => s.location.pathname })
  const isAppShell = pathname === '/signin' || pathname.startsWith('/app')

  return (
    <html lang="en">
      <head>
        <HeadContent />
      </head>
      <body className="bg-espresso text-text-primary font-sans">
        {!isAppShell && <Header />}
        <main>
          <Outlet />
        </main>
        {!isAppShell && <Footer />}
        <Scripts />
      </body>
    </html>
  )
}
```

**Step 2: Verify**

Run dev server. Landing page should still have header/footer. `/signin` should NOT have header/footer.

**Step 3: Commit**

```bash
git add apps/web/src/routes/__root.tsx
git commit -m "hide header/footer on signin and app routes"
```

---

### Task 5: Build the main app screen (`/app`)

**Files:**
- Create: `apps/web/src/routes/app.tsx`

This is the largest task. The screen has three zones: thin icon sidebar (left), profile button (top-right), centered content area.

**Step 1: Create the app route with auth guard and full layout**

Create `apps/web/src/routes/app.tsx`:

```tsx
import { useState } from 'react'
import { createFileRoute, useNavigate, redirect } from '@tanstack/react-router'
import { MessageSquare, Mic, Plus } from 'lucide-react'
import { getAuth, clearAuth, isAuthenticated } from '../lib/auth'

export const Route = createFileRoute('/app')({
  beforeLoad: () => {
    if (!isAuthenticated()) {
      throw redirect({ to: '/signin' })
    }
  },
  component: AppPage,
})

function AppPage() {
  const user = getAuth()
  const navigate = useNavigate()
  const [showProfile, setShowProfile] = useState(false)

  function handleSignOut() {
    clearAuth()
    navigate({ to: '/' })
  }

  // Time-aware greeting
  const hour = new Date().getHours()
  let greeting = 'Good morning'
  if (hour >= 12 && hour < 17) greeting = 'Good afternoon'
  else if (hour >= 17) greeting = 'Good evening'

  return (
    <div className="h-screen flex overflow-hidden">
      {/* Thin icon sidebar */}
      <aside className="w-12 shrink-0 border-r border-border flex flex-col items-center py-4 gap-1">
        <SidebarButton icon={<Plus size={18} />} label="New Chat" />
        <SidebarButton icon={<MessageSquare size={18} />} label="Chats" />
        <SidebarButton icon={<MetronomeIcon />} label="Metronome" />
      </aside>

      {/* Main content area */}
      <div className="flex-1 relative flex flex-col">
        {/* Profile button -- top right */}
        <div className="absolute top-4 right-4 z-20">
          <button
            type="button"
            onClick={() => setShowProfile(!showProfile)}
            className="w-8 h-8 bg-surface border border-border flex items-center justify-center text-body-sm text-cream font-medium hover:bg-surface-2 transition"
          >
            {user?.name?.charAt(0).toUpperCase() ?? '?'}
          </button>

          {showProfile && (
            <div className="absolute right-0 top-10 bg-surface border border-border py-1 min-w-[140px]">
              <button
                type="button"
                onClick={handleSignOut}
                className="w-full text-left px-4 py-2 text-body-sm text-text-secondary hover:text-cream hover:bg-surface-2 transition"
              >
                Sign Out
              </button>
            </div>
          )}
        </div>

        {/* Centered home content */}
        <div className="flex-1 flex items-center justify-center px-6">
          <div className="w-full max-w-2xl text-center">
            <h1 className="font-display text-display-md text-cream">
              {greeting}, {user?.name ?? 'there'}.
            </h1>

            {/* Input box */}
            <div className="mt-8 bg-surface border border-border flex items-center">
              <input
                type="text"
                placeholder="What are you practicing today?"
                className="flex-1 bg-transparent px-5 py-4 text-body-md text-cream placeholder:text-text-tertiary outline-none"
              />
              <button
                type="button"
                className="px-4 py-4 text-text-secondary hover:text-cream transition"
                aria-label="Start recording"
              >
                <Mic size={20} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function SidebarButton({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <button
      type="button"
      className="w-10 h-10 flex items-center justify-center text-text-secondary hover:text-cream hover:bg-surface transition group relative"
      aria-label={label}
    >
      {icon}
      {/* Tooltip */}
      <span className="absolute left-full ml-2 px-2 py-1 bg-surface-2 text-body-xs text-cream whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity">
        {label}
      </span>
    </button>
  )
}

function MetronomeIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2v10" />
      <path d="M5 21h14" />
      <path d="M7.5 21L10 6h4l2.5 15" />
      <path d="M12 12l4-4" />
    </svg>
  )
}
```

**Step 2: Run dev server and verify**

Run: `cd apps/web && bun run dev`

Test flow:
1. Navigate to `http://localhost:3000/app` -- should redirect to `/signin` (no auth)
2. Click "Sign in with Apple" on sign-in page -- should redirect to `/app`
3. Verify: greeting with "Jai", input box with mic icon, thin sidebar with 3 icons, profile initial "J" top-right
4. Click "J" -- dropdown appears with "Sign Out"
5. Click "Sign Out" -- redirects to landing page

**Step 3: Commit**

```bash
git add apps/web/src/routes/app.tsx
git commit -m "build main app screen with sidebar, greeting, and auth guard"
```

---

### Task 6: Update landing page CTA to check auth

**Files:**
- Modify: `apps/web/src/routes/index.tsx:1` (add import)
- Modify: `apps/web/src/routes/index.tsx:44-49` (hero CTA)
- Modify: `apps/web/src/routes/index.tsx:152-154` (final CTA)

**Step 1: Make CTAs auth-aware**

The `href="/app"` links from Task 1 will work because the `/app` route's `beforeLoad` guard handles the redirect to `/signin`. No additional logic needed on the landing page -- the auth guard does the work.

Verify by testing:
1. Not signed in: click "Start Practicing" -> `/app` -> redirected to `/signin`
2. Signed in: click "Start Practicing" -> `/app` -> shows app

This task is a no-op -- the guard from Task 5 handles it. Skip to Task 7.

---

### Task 7: Final visual polish and verify full flow

**Files:**
- All routes (read-only verification)

**Step 1: Run dev server and test the complete flow**

Run: `cd apps/web && bun run dev`

Test checklist:
- [ ] Landing page loads with header + footer
- [ ] "Start Practicing" button navigates to `/signin` when not authenticated
- [ ] Sign-in page: full-bleed Image4.jpg, floating card, no header/footer
- [ ] "Sign in with Apple" button stores auth and redirects to `/app`
- [ ] App page: no header/footer, thin sidebar with 3 icons + tooltips, greeting with name, input box with mic
- [ ] Profile dropdown with "Sign Out" works
- [ ] Sign out clears auth and returns to landing page
- [ ] `/analyze` returns 404 (route deleted)
- [ ] Header "Sign In" link on landing page goes to `/signin`

**Step 2: Fix any visual issues found during testing**

Adjust spacing, colors, or sizing as needed to match the design doc.

**Step 3: Final commit**

```bash
git add -A
git commit -m "polish: complete signin and app screen implementation"
```

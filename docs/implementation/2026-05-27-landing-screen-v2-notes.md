# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Task 0: Types, manifest fixtures, and analytics helper

Implemented exactly as specified. ProofCardManifest types, analytics helper, and three JSON manifests created. All tests pass.

## Task 1: ProofCard render contract

ProofCard.tsx implemented with full implementation including ScoreCursor integration, IntersectionObserver autoplay, BarScoreChip, keyboard navigation, asset prefetch. The useProofCardTimeline.ts stub created here was intentionally incomplete to allow Task 8 tests to genuinely fail against it.

## Task 2: Scroll autoplay

Test written against existing ProofCard implementation. The autoplay test uses jsdom's default visibilityState="visible". Verified IntersectionObserver.observe works correctly.

## Task 3: Graceful degradation — missing scoreIR

Test verifies ProofCard renders diagnosis and exercise when scoreIR fetch returns non-ok. Already handled by try/catch in load() function.

## Task 4: Graceful degradation — missing audio

Audio error handling moved to addEventListener approach (not React onError prop) because React 18 does not reliably fire synthetic events for HTMLMediaElement in jsdom.

## Task 4.5: ScoreCursor instantiation and cursor movement

vi.mock at module top-level per Vitest hoisting requirements. ScoreCursor constructor called with options object {pieceId, container, ir, qstampSource}. cursorQstampSource reads audioRef.current?.currentTime directly to avoid recreating cursor on timeupdate events.

## Task 8: useProofCardTimeline bidirectional scrub sync

Full implementation was already in the worktree from prior session. Test file created to verify qstampForTime returns qstampStart floats (quaternote positions), not bar numbers. All 5 tests pass.

## Task 5: Reduced motion — autoplay disabled, play button visible

IntersectionObserver mock must be set up BEFORE vi.resetModules() so the mock function is available when the freshly-imported ProofCard module evaluates. Mock function must use function keyword (not arrow function) to be usable as constructor.

## Task 7: BarScoreChip and bar-tap behavior

BarScoreChip.tsx was already in the worktree. Component renders 3-char abbreviations visually but needed sr-only full names for screen.getByText assertions. Added aria-label + sr-only span for each dimension label.

## Task 9: Keyboard navigation

Bar buttons have tabIndex={0}. handleBarKeyDown handles Enter (open chip) and Escape (close chip).

## Task 6: Axe accessibility scan on LandingPage

vitest-axe 0.1.0 exports toHaveNoViolations from "vitest-axe/matchers" (not main entry). expect.extend({toHaveNoViolations}) (object form) required.

## Task 10: Wire up index.tsx and final CTA

index.tsx updated with ExerciseProofBlock, FinalCtaSection ("Your playing. Heard clearly."), and LandingFooter with ISMIR footnote. Old sections removed. Test uses named export {Route} not default export.

// The /vitest entrypoint registers the matchers AND augments vitest's Assertion
// interface. Importing /matchers and calling expect.extend by hand registers them
// at runtime only, so every toBeInTheDocument() was a type error.
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

afterEach(() => {
	cleanup();
});

// jsdom does not implement matchMedia — stub it globally so components that
// read window.matchMedia at render time (e.g. ProofCard reducedMotion ref) don't throw.
// Tests that need to control the return value override this in beforeEach.
Object.defineProperty(window, "matchMedia", {
	writable: true,
	value: vi.fn().mockImplementation((query: string) => ({
		matches: false,
		media: query,
		addListener: vi.fn(),
		removeListener: vi.fn(),
	})),
});

// jsdom does not implement IntersectionObserver — stub it globally with a class
// so `new IntersectionObserver(cb)` works. Tests that need to trigger callbacks
// override this in their own beforeEach.
class MockIntersectionObserver {
	observe = vi.fn();
	disconnect = vi.fn();
	unobserve = vi.fn();
	constructor(_cb: IntersectionObserverCallback) {}
}
globalThis.IntersectionObserver =
	MockIntersectionObserver as unknown as typeof IntersectionObserver;

// jsdom does not implement ResizeObserver — stub it globally so components that
// observe container resize (ScoreMarkLayer re-measures because Verovio reflows
// on width change) don't throw during render.
// No constructor: the callback argument is ignored, and declaring one only to
// type it trips noUselessConstructor. Tests that need to fire the callback
// override this in their own beforeEach, as with IntersectionObserver above.
class MockResizeObserver {
	observe = vi.fn();
	disconnect = vi.fn();
	unobserve = vi.fn();
}
globalThis.ResizeObserver =
	MockResizeObserver as unknown as typeof ResizeObserver;

// jsdom does not implement Element.scrollTo — stub it so scroll-aware components
// (e.g. ChatMessages) don't throw during render in tests.
Element.prototype.scrollTo =
	vi.fn() as unknown as typeof Element.prototype.scrollTo;

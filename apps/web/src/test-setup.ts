import * as matchers from "@testing-library/jest-dom/matchers";
import { cleanup } from "@testing-library/react";
import { afterEach, expect, vi } from "vitest";

expect.extend(matchers);
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
globalThis.IntersectionObserver = MockIntersectionObserver as unknown as typeof IntersectionObserver;

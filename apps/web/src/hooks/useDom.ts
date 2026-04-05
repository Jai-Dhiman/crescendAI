import { type RefObject, useEffect, useState } from "react";
import { useMountEffect } from "./useFoundation";

/**
 * Calls handler when a click occurs outside the referenced element.
 * No-ops when disabled.
 */
export function useClickOutside(
	ref: RefObject<HTMLElement | null>,
	handler: () => void,
	enabled: boolean,
): void {
	useEffect(() => {
		if (!enabled) return;
		function handleClick(e: MouseEvent) {
			if (ref.current && !ref.current.contains(e.target as Node)) {
				handler();
			}
		}
		document.addEventListener("mousedown", handleClick);
		return () => document.removeEventListener("mousedown", handleClick);
	}, [ref, handler, enabled]);
}

/**
 * Calls handler when Escape key is pressed. No-ops when disabled.
 */
export function useEscapeKey(handler: () => void, enabled: boolean): void {
	useEffect(() => {
		if (!enabled) return;
		function onKeyDown(e: KeyboardEvent) {
			if (e.key === "Escape") {
				handler();
			}
		}
		document.addEventListener("keydown", onKeyDown);
		return () => document.removeEventListener("keydown", onKeyDown);
	}, [handler, enabled]);
}

/**
 * Returns true when the viewport is narrower than `breakpoint` (default 768).
 * Listens for resize events.
 */
export function useIsMobile(breakpoint = 768): boolean {
	const [isMobile, setIsMobile] = useState(false);

	useMountEffect(() => {
		function check() {
			setIsMobile(window.innerWidth < breakpoint);
		}
		check();
		window.addEventListener("resize", check);
		return () => window.removeEventListener("resize", check);
	});

	return isMobile;
}

/**
 * Returns the bottom offset caused by the mobile virtual keyboard.
 * Uses the Visual Viewport API to detect keyboard presence.
 */
export function useKeyboardOffset(): number {
	const [bottomOffset, setBottomOffset] = useState(0);

	useMountEffect(() => {
		const vv = window.visualViewport;
		if (!vv) return;

		function handleResize() {
			if (!vv) return;
			const offset = window.innerHeight - vv.height - vv.offsetTop;
			setBottomOffset(Math.max(0, offset));
		}

		vv.addEventListener("resize", handleResize);
		vv.addEventListener("scroll", handleResize);
		return () => {
			vv.removeEventListener("resize", handleResize);
			vv.removeEventListener("scroll", handleResize);
		};
	});

	return bottomOffset;
}

/**
 * Returns current network connectivity status.
 * Listens for online/offline events.
 */
export function useNetworkStatus(): boolean {
	const [isOnline, setIsOnline] = useState(
		typeof navigator !== "undefined" ? navigator.onLine : true,
	);

	useMountEffect(() => {
		function handleOnline() {
			setIsOnline(true);
		}
		function handleOffline() {
			setIsOnline(false);
		}
		window.addEventListener("online", handleOnline);
		window.addEventListener("offline", handleOffline);
		return () => {
			window.removeEventListener("online", handleOnline);
			window.removeEventListener("offline", handleOffline);
		};
	});

	return isOnline;
}

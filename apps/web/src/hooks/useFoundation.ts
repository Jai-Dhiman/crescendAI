import { useEffect, useRef } from "react";

/**
 * Mount-only effect with explicit intent.
 * Replaces bare useEffect(..., []) -- makes "run once on mount" self-documenting
 * and enables lint rules against raw useEffect usage.
 */
export function useMountEffect(effect: () => void | (() => void)): void {
	// biome-ignore lint/correctness/useExhaustiveDependencies: intentional mount-only effect
	useEffect(effect, []);
}

/**
 * Synchronous ref that stays current with a value -- no useEffect needed.
 * Assigning ref.current during render is safe (no side effects) and avoids
 * the one-frame delay that useEffect-based ref sync introduces.
 *
 * Use when callbacks (RAF, setInterval, event listeners) need the latest
 * value without re-subscribing.
 */
export function useSyncRef<T>(value: T): React.MutableRefObject<T> {
	const ref = useRef(value);
	ref.current = value;
	return ref;
}

export type Theme = "light" | "dark";

const DUSK_HOUR = 19; // 19:00 device-local, dark begins
const DAWN_HOUR = 7; // 07:00 device-local, light begins

function isValidTheme(value: string | null): value is Theme {
	return value === "light" || value === "dark";
}

/**
 * Theme precedence, highest first:
 *   1. manual override (`stored`, validated — junk values are ignored)
 *   2. time of day (dark 19:00-06:59 device-local, light 07:00-18:59)
 *   3. light, when no clock is available (SSR)
 */
export function resolveTheme(input: {
	stored: string | null;
	now: Date | null;
}): Theme {
	if (isValidTheme(input.stored)) return input.stored;

	if (input.now === null) return "light";

	const hour = input.now.getHours();
	return hour >= DUSK_HOUR || hour < DAWN_HOUR ? "dark" : "light";
}

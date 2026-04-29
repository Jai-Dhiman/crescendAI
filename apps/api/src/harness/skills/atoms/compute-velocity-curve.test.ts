import { test, expect } from "vitest";
import { computeVelocityCurve } from "./compute-velocity-curve";

test("computeVelocityCurve: bars 12-14 with spec example velocities produce correct curve", async () => {
	// bar 12: velocities [60,65,70] → mean=65, p90=69 (90th pct of [60,65,70] = 65+0.9*(70-65)=69.5≈69)
	// bar 13: velocities [80,85] → mean=82.5, p90=84.5 (90th pct of [80,85])
	// bar 14: velocities [50,55,60] → mean=55, p90=59
	const notes = [
		{ onset_ms: 100, velocity: 60, bar: 12 },
		{ onset_ms: 200, velocity: 65, bar: 12 },
		{ onset_ms: 300, velocity: 70, bar: 12 },
		{ onset_ms: 400, velocity: 80, bar: 13 },
		{ onset_ms: 500, velocity: 85, bar: 13 },
		{ onset_ms: 600, velocity: 50, bar: 14 },
		{ onset_ms: 700, velocity: 55, bar: 14 },
		{ onset_ms: 800, velocity: 60, bar: 14 },
	];
	const result = (await computeVelocityCurve.invoke({
		bar_range: [12, 14],
		notes,
	})) as { bar: number; mean_velocity: number; p90_velocity: number }[];
	expect(result).toHaveLength(3);
	expect(result[0].bar).toBe(12);
	expect(result[0].mean_velocity).toBeCloseTo(65, 1);
	expect(result[1].bar).toBe(13);
	expect(result[1].mean_velocity).toBeCloseTo(82.5, 1);
	expect(result[2].bar).toBe(14);
	expect(result[2].mean_velocity).toBeCloseTo(55, 1);
});

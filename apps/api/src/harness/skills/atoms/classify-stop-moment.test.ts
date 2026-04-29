import { test, expect } from "vitest";
import { classifyStopMoment } from "./classify-stop-moment";

test("classifyStopMoment: very low pedaling with other dims at cohort mean gives stop probability > 0.7", async () => {
	// All dims at SCALER_MEAN except pedaling set to 0.15 (far below 0.4594 mean)
	// pedaling scaled = (0.15 - 0.4594) / 0.0791 ≈ -3.91 → weight -0.5483 → contribution +2.14
	// other dims at mean → scaled = 0 → zero contribution → logit ≈ 2.14 + BIAS 0.1147 ≈ 2.25
	// sigmoid(2.25) ≈ 0.905
	const scores = [0.545, 0.4848, 0.15, 0.5369, 0.5188, 0.5064];
	const result = await classifyStopMoment.invoke({ scores });
	expect(typeof result).toBe("number");
	expect(result).toBeGreaterThan(0.7);
	expect(result).toBeLessThanOrEqual(1.0);
});

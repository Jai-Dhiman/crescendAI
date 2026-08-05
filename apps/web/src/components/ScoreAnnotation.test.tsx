import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ScoreAnnotation } from "./ScoreAnnotation";

describe("ScoreAnnotation dimension color", () => {
	it("reads its swatch color from DIMENSION_COLOR_VAR", () => {
		const { getByRole } = render(
			<ScoreAnnotation
				dimension="dynamics"
				barRange={[1, 4]}
				index={0}
				isActive={false}
				style={{}}
				onClick={vi.fn()}
			/>,
		);
		const button = getByRole("button");
		expect(button).toHaveStyle({ backgroundColor: "var(--dim-dynamics)" });
	});
});

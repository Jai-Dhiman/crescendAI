import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";
import { MarkGlyph } from "./MarkGlyph";

const mark: Mark = {
	id: "m1",
	anchor: resolveAnchor({ atSeconds: 64, bars: [5, 6], alignmentQuality: 1 }),
	taxonomy: "needs_work",
	dimension: "pedaling",
	evidence: "pedal held through the bass change",
	lifecycle: "active",
};

describe("MarkGlyph", () => {
	it("names taxonomy, dimension, and location, and shows the dimension as text", () => {
		render(<MarkGlyph mark={mark} expanded={false} onToggle={vi.fn()} />);

		const button = screen.getByRole("button");
		expect(button).toHaveAccessibleName("Needs work: Pedaling, bars 5-6");
		// Colour is not the sole means of conveying the dimension (WCAG 1.4.1):
		// the dimension is present as visible text, not only as the tint dot.
		expect(button).toHaveTextContent("Pedaling");
		expect(button).toHaveAttribute("aria-expanded", "false");
	});

	it("takes lifecycle strength from the server-supplied value, not from the mark's content", () => {
		// A `strong` mark that is `improving` is not derivable from anything on
		// the client: taxonomy says the student played well, lifecycle says the
		// baseline is still moving. If the component recomputed lifecycle from
		// mark content, this combination could not survive a render.
		const undeducible: Mark = {
			...mark,
			taxonomy: "strong",
			lifecycle: "improving",
		};
		// The fade lands on the DOT, never on the button: fading the button fades
		// the mark's text too, and at `resolved` (0.4) no text colour can reach
		// 4.5:1 — pure black tops out at 2.82:1. Real-browser axe caught this;
		// jsdom cannot, because it skips the color-contrast rule entirely.
		const dot = () => screen.getByTestId("mark-lifecycle-dot");

		const { rerender } = render(
			<MarkGlyph mark={undeducible} expanded={false} onToggle={vi.fn()} />,
		);
		expect(screen.getByRole("button")).toHaveAttribute(
			"data-lifecycle",
			"improving",
		);
		expect(dot()).toHaveStyle({ opacity: "0.7" });

		rerender(
			<MarkGlyph
				mark={{ ...undeducible, lifecycle: "resolved" }}
				expanded={false}
				onToggle={vi.fn()}
			/>,
		);
		expect(screen.getByRole("button")).toHaveAttribute(
			"data-lifecycle",
			"resolved",
		);
		expect(dot()).toHaveStyle({ opacity: "0.4" });

		rerender(
			<MarkGlyph
				mark={{ ...undeducible, lifecycle: "active" }}
				expanded={false}
				onToggle={vi.fn()}
			/>,
		);
		expect(screen.getByRole("button")).toHaveAttribute(
			"data-lifecycle",
			"active",
		);
		expect(dot()).toHaveStyle({ opacity: "1" });
	});

	it("never fades the mark's text, whatever the lifecycle", () => {
		// The regression guard for the above: a future change that moves the
		// fade back onto the button would put resolved marks at 1.74:1.
		render(
			<MarkGlyph
				mark={{ ...mark, lifecycle: "resolved" }}
				expanded={false}
				onToggle={vi.fn()}
			/>,
		);
		expect(screen.getByRole("button")).not.toHaveStyle({ opacity: "0.4" });
	});
});

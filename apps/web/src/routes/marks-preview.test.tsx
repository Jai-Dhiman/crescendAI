import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MarksPreview } from "./marks-preview";

describe("marks preview route", () => {
	it("mounts both canvases against the same fixture marks", () => {
		render(<MarksPreview />);

		expect(
			screen.getByRole("heading", { name: /score overlay/i }),
		).toBeInTheDocument();
		expect(
			screen.getByRole("heading", { name: /session timeline/i }),
		).toBeInTheDocument();

		// The same mark on both canvases: pedaling bars 5-6 resolves on the
		// score canvas and also appears on the timeline.
		expect(
			screen.getAllByLabelText(/Needs work: Pedaling, bars 5-6/),
		).toHaveLength(2);
	});
});

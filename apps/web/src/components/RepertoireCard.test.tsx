import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { RepertoireEntry } from "../lib/api";
import { RepertoireCard } from "./RepertoireCard";

const NOW = new Date("2026-08-27T12:00:00.000Z");

const ENTRY: RepertoireEntry = {
	pieceId: "chopin-nocturne-op9-no2",
	composer: "Chopin",
	title: "Nocturne Op. 9 No. 2",
	sessionCount: 6,
	lastPracticedAt: "2026-08-26T18:00:00.000Z",
	openMarkCount: null,
};

describe("RepertoireCard", () => {
	it("shows the piece and its arc line", () => {
		render(<RepertoireCard entry={ENTRY} now={NOW} onOpen={vi.fn()} />);
		expect(screen.getByText("Nocturne Op. 9 No. 2")).toBeInTheDocument();
		expect(screen.getByText("Chopin")).toBeInTheDocument();
		expect(
			screen.getByText("6 sessions · last practiced yesterday"),
		).toBeInTheDocument();
	});

	it("omits the mark count when it is unknown, rather than showing zero", () => {
		render(<RepertoireCard entry={ENTRY} now={NOW} onOpen={vi.fn()} />);
		expect(screen.queryByTestId("open-mark-count")).not.toBeInTheDocument();
	});

	it("shows the mark count once one is known", () => {
		render(
			<RepertoireCard
				entry={{ ...ENTRY, openMarkCount: 3 }}
				now={NOW}
				onOpen={vi.fn()}
			/>,
		);
		expect(screen.getByTestId("open-mark-count")).toHaveTextContent(
			"3 open marks",
		);
	});

	it("reports which piece was opened", () => {
		const onOpen = vi.fn();
		render(<RepertoireCard entry={ENTRY} now={NOW} onOpen={onOpen} />);
		fireEvent.click(
			screen.getByRole("button", { name: /Nocturne Op\. 9 No\. 2/ }),
		);
		expect(onOpen).toHaveBeenCalledWith("chopin-nocturne-op9-no2");
	});
});

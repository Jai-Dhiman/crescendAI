import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SessionDisclosures } from "./SessionDisclosures";

describe("SessionDisclosures", () => {
	it("renders nothing for a clean session, and discloses gaps and no-comment zones when present", () => {
		const clean = render(
			<SessionDisclosures wsGaps={[]} noCommentZones={[]} />,
		);
		expect(clean.container).toBeEmptyDOMElement();
		clean.unmount();

		render(
			<SessionDisclosures
				wsGaps={[{ startSeconds: 0, endSeconds: 240 }]}
				noCommentZones={[
					{
						atSeconds: 58,
						reason: "transcription confidence was too low here",
					},
				]}
			/>,
		);
		expect(screen.getByText("4 minutes didn't reach me")).toBeInTheDocument();
		expect(
			screen.getByText(/No comment near 0:58 — transcription confidence/),
		).toBeInTheDocument();
	});
});

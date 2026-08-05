import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MessageContent } from "./MessageContent";

describe("MessageContent", () => {
	it("uses an opacity-modified accent for inline-code and link emphasis, not accent-lighter", () => {
		const content =
			"Use `noteVelocity` and see [the docs](https://example.com/docs).";
		const { container } = render(<MessageContent content={content} />);
		expect(container.innerHTML).not.toContain("text-accent-lighter");
		expect(container.innerHTML).toContain("text-accent/70");
	});
});

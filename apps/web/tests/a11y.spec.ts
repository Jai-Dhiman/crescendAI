import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

// __root.tsx forces data-theme="dark" on "/" and "/signin" via both a
// pre-paint flash script and a runtime effect keyed on pathname
// (resolveDocumentTheme). A manual page.evaluate that sets data-theme
// on those routes gets silently overwritten back to "dark" by that
// effect, so the light-theme case has to run on a route that is not in
// the always-dark list. "/privacy" is public, unauthenticated, and does
// no data fetching, so it renders standalone in a preview build.
const THEME_CASES = [
	{ theme: "light", path: "/privacy" },
	{ theme: "dark", path: "/signin" },
] as const;

test.describe("color contrast", () => {
	for (const { theme, path } of THEME_CASES) {
		test(`app shell has no color-contrast violations (${theme})`, async ({
			page,
		}) => {
			await page.goto(path);

			await page.evaluate((t) => {
				document.documentElement.dataset.theme = t;
			}, theme);

			// Verify the attribute actually stuck: a runtime effect in __root.tsx
			// can overwrite data-theme after mount, which would silently make
			// this test check the wrong theme.
			const appliedTheme = await page.evaluate(
				() => document.documentElement.dataset.theme,
			);
			expect(appliedTheme).toBe(theme);

			const results = await new AxeBuilder({ page })
				.withRules(["color-contrast"])
				.exclude("[data-axe-exempt]")
				.analyze();

			expect(results.violations).toEqual([]);
		});
	}
});

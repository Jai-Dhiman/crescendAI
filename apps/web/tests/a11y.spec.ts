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
	// #157: the mark canvases. A top-level route, so it renders in a preview
	// build without auth. axe's color-contrast rule needs real layout and
	// silently SKIPS in jsdom, so this is the only place mark contrast is
	// actually verified — never assert it from vitest.
	{ theme: "light", path: "/marks-preview" },
	{ theme: "dark", path: "/marks-preview" },
] as const;

test.describe("color contrast", () => {
	for (const { theme, path } of THEME_CASES) {
		// Title carries the path as well as the theme: #157 added a second
		// light case and a second dark case, and a theme-only title collides.
		test(`app shell has no color-contrast violations (${theme}, ${path})`, async ({
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

			// A bare toEqual([]) prints hundreds of lines of serialised nodes and
			// no selectors, which is unreadable and hides WHICH element failed.
			// Summarise the targets first so a failure names its elements.
			for (const v of results.violations) {
				for (const node of v.nodes) {
					console.log(`[${theme} ${path}] ${v.id} :: ${node.target.join(" ")}`);
				}
			}

			expect(results.violations).toEqual([]);
		});
	}
});

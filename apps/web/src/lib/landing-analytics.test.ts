// apps/web/src/lib/landing-analytics.test.ts
import { describe, expect, it, vi, afterEach } from "vitest";

describe("trackLandingEvent", () => {
  afterEach(() => {
    // Clean up gtag stub
    delete (window as Record<string, unknown>).gtag;
  });

  it("calls window.gtag with event name and params when gtag is present", async () => {
    const gtag = vi.fn();
    (window as Record<string, unknown>).gtag = gtag;
    const { trackLandingEvent } = await import("./landing-analytics");
    trackLandingEvent("landing_hero_cta_click", { foo: "bar" });
    expect(gtag).toHaveBeenCalledWith("event", "landing_hero_cta_click", { foo: "bar" });
  });

  it("does not throw when window.gtag is absent", async () => {
    const { trackLandingEvent } = await import("./landing-analytics");
    expect(() => trackLandingEvent("landing_hero_cta_click")).not.toThrow();
  });
});

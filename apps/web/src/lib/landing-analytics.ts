// apps/web/src/lib/landing-analytics.ts

export function trackLandingEvent(
  name: string,
  params?: Record<string, unknown>,
): void {
  const gtag = (window as Record<string, unknown>).gtag;
  if (typeof gtag === "function") {
    gtag("event", name, params);
  }
}

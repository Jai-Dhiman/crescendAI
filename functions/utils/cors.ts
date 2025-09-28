// functions/utils/cors.ts
export type AllowedOrigin = string;

export function parseAllowedOrigins(origins: string | undefined): AllowedOrigin[] {
  if (!origins) return [];
  return origins
    .split(",")
    .map((o) => o.trim())
    .filter((o) => o.length > 0);
}

export function isOriginAllowed(origin: string, allowed: AllowedOrigin[]): boolean {
  if (!origin) return true; // non-browser or same-origin fetch
  try {
    const url = new URL(origin);
    const host = url.host;
    for (const entry of allowed) {
      if (!entry) continue;
      if (entry === origin) return true; // exact match
      // wildcard support: *.domain.tld
      if (entry.startsWith("*.") && host.endsWith(entry.slice(2))) return true;
    }
  } catch {
    // If origin is malformed, do not allow
    return false;
  }
  return false;
}

export function buildCorsHeaders(origin: string, allowed: AllowedOrigin[]): HeadersInit {
  const headers: Record<string, string> = {
    "Vary": "Origin",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Max-Age": "86400",
  };
  if (origin && isOriginAllowed(origin, allowed)) {
    headers["Access-Control-Allow-Origin"] = origin;
  }
  return headers;
}

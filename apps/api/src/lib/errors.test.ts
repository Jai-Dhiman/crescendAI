import { describe, expect, it } from "vitest";
import { ConfigError, DomainError } from "./errors";

describe("ConfigError", () => {
  it("is a DomainError with name ConfigError", () => {
    const err = new ConfigError("missing binding");
    expect(err).toBeInstanceOf(DomainError);
    expect(err.name).toBe("ConfigError");
    expect(err.message).toBe("missing binding");
  });
});

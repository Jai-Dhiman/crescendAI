import { describe, expect, it, vi } from "vitest";
import { ConfigError } from "../../lib/errors";
import { runStreamingHook } from "./runStreamingHook";
import type { HookContext } from "./types";

vi.mock("./compound-registry", () => ({
  getCompoundBinding: vi.fn(),
}));

import { getCompoundBinding } from "./compound-registry";

const stubHookCtx: HookContext = {
  env: {} as never,
  studentId: "s1",
  sessionId: "",
  conversationId: null,
  digest: {},
  waitUntil: () => {},
};

describe("runStreamingHook error paths", () => {
  it("throws ConfigError when hook has no registered binding", async () => {
    vi.mocked(getCompoundBinding).mockReturnValue(undefined);
    const gen = runStreamingHook("OnStop", stubHookCtx, async () => ({} as never), [], []);
    await expect(gen.next()).rejects.toBeInstanceOf(ConfigError);
  });

  it("throws ConfigError when binding mode is not streaming", async () => {
    vi.mocked(getCompoundBinding).mockReturnValue({
      compoundName: "x",
      procedurePrompt: "",
      tools: [],
      mode: "buffered",
      phases: 2,
      artifactSchema: undefined,
      artifactToolName: undefined,
    } as never);
    const gen = runStreamingHook("OnSessionEnd", stubHookCtx, async () => ({} as never), [], []);
    await expect(gen.next()).rejects.toBeInstanceOf(ConfigError);
  });
});

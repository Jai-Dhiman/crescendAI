import { describe, expect, it } from "vitest";
import type { Bindings, Db, ServiceContext } from "../lib/types";
import { buildChatBinding } from "./teacher";
import { TOOL_REGISTRY } from "./tool-processor";

const MOCK_ENV = {
  AI_GATEWAY_TEACHER: "https://gw.example",
  ANTHROPIC_API_KEY: "test-key",
} as unknown as Bindings;

const MOCK_CTX: ServiceContext = {
  db: {} as Db,
  env: MOCK_ENV,
};

describe("buildChatBinding", () => {
  it("returns mode:'streaming' and phases:1", () => {
    const binding = buildChatBinding(MOCK_CTX, "stu_1");
    expect(binding.mode).toBe("streaming");
    expect(binding.phases).toBe(1);
  });

  it("includes all TOOL_REGISTRY tools by name", () => {
    const binding = buildChatBinding(MOCK_CTX, "stu_1");
    const expected = Object.keys(TOOL_REGISTRY).sort();
    const actual = binding.tools.map((t) => t.name).sort();
    expect(actual).toEqual(expected);
  });

  it("each tool has an object input_schema and an invoke function", () => {
    const binding = buildChatBinding(MOCK_CTX, "stu_1");
    for (const tool of binding.tools) {
      expect(typeof tool.input_schema).toBe("object");
      expect(typeof tool.invoke).toBe("function");
    }
  });
});

import { describe, expect, test, vi } from "vitest";
import { assignSegmentLoopAtom, ASSIGN_SEGMENT_LOOP_TOOL } from "./assign-segment-loop";
import { ToolPreconditionError } from "../../lib/errors";
import type { PhaseContext } from "../loop/types";
import type { Bindings } from "../../lib/types";

const mockCreateLoop = vi.fn();
vi.mock("../../services/segment-loops", () => ({
  createSegmentLoop: (...args: unknown[]) => mockCreateLoop(...args),
}));
vi.mock("../../db/client", () => ({
  createDb: () => ({}),
}));

const BASE_CTX: PhaseContext = {
  env: {} as Bindings,
  studentId: "stu-1",
  sessionId: "sess-1",
  conversationId: "conv-1",
  digest: {},
  waitUntil: () => {},
  turnCap: 5,
  trigger: "synthesis",
  pieceId: "chopin.ballades.1",
};

const VALID_INPUT = {
  piece_id: "chopin.ballades.1",
  bars_start: 12,
  bars_end: 16,
  required_correct: 3,
};

describe("assignSegmentLoopAtom", () => {
  test("synthesis trigger creates active loop", async () => {
    const mockArtifact = { kind: "segment_loop", id: "loop-1", status: "active" };
    mockCreateLoop.mockResolvedValueOnce(mockArtifact);

    const result = await assignSegmentLoopAtom(BASE_CTX, VALID_INPUT);

    expect(mockCreateLoop).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ trigger: "synthesis" }),
    );
    expect(result).toEqual(mockArtifact);
  });

  test("chat trigger creates pending loop", async () => {
    const mockArtifact = { kind: "segment_loop", id: "loop-2", status: "pending" };
    mockCreateLoop.mockResolvedValueOnce(mockArtifact);

    const chatCtx = { ...BASE_CTX, trigger: "chat" as const };
    await assignSegmentLoopAtom(chatCtx, VALID_INPUT);

    expect(mockCreateLoop).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ trigger: "chat" }),
    );
  });

  test("missing piece_id throws ToolPreconditionError", async () => {
    await expect(
      assignSegmentLoopAtom(BASE_CTX, { bars_start: 12, bars_end: 16, required_correct: 3 }),
    ).rejects.toBeInstanceOf(ToolPreconditionError);
  });

  test("bars_end < bars_start throws ValidationError", async () => {
    await expect(
      assignSegmentLoopAtom(BASE_CTX, { ...VALID_INPUT, bars_start: 20, bars_end: 10 }),
    ).rejects.toThrow();
  });

  test("ASSIGN_SEGMENT_LOOP_TOOL has correct name", () => {
    expect(ASSIGN_SEGMENT_LOOP_TOOL.name).toBe("assign_segment_loop");
  });

  test("missing trigger on ctx throws ToolPreconditionError", async () => {
    const noTriggerCtx = { ...BASE_CTX, trigger: undefined } as unknown as PhaseContext;
    await expect(
      assignSegmentLoopAtom(noTriggerCtx, VALID_INPUT),
    ).rejects.toBeInstanceOf(ToolPreconditionError);
  });
});

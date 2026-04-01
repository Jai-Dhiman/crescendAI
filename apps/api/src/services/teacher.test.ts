import { describe, it, expect, vi } from "vitest";
import { parseAnthropicStream, stripAnalysis } from "./teacher";
import type { InlineComponent } from "./tool-processor";

// ---------------------------------------------------------------------------
// Helpers for building fake SSE streams
// ---------------------------------------------------------------------------

function sseLines(...events: Array<{ event: string; data: unknown }>): string {
  return events
    .map((e) => `event: ${e.event}\ndata: ${JSON.stringify(e.data)}`)
    .join("\n\n") + "\n\n";
}

function makeStream(content: string): ReadableStream {
  const encoder = new TextEncoder();
  const bytes = encoder.encode(content);
  return new ReadableStream({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

// ---------------------------------------------------------------------------
// Test: stream parser with text only
// ---------------------------------------------------------------------------

describe("parseAnthropicStream - text only", () => {
  it("yields delta events and done with fullText", async () => {
    const sse = sseLines(
      {
        event: "message_start",
        data: { type: "message_start", message: { id: "msg_1" } },
      },
      {
        event: "content_block_start",
        data: { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
      },
      {
        event: "content_block_delta",
        data: { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hello" } },
      },
      {
        event: "content_block_delta",
        data: { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: " world" } },
      },
      {
        event: "content_block_stop",
        data: { type: "content_block_stop", index: 0 },
      },
      {
        event: "message_delta",
        data: { type: "message_delta", delta: { stop_reason: "end_turn" } },
      },
      {
        event: "message_stop",
        data: { type: "message_stop" },
      },
    );

    const stream = makeStream(sse);
    const noopProcess = vi.fn().mockResolvedValue({ name: "noop", componentsJson: [], isError: false });

    const events = [];
    for await (const event of parseAnthropicStream(stream, noopProcess)) {
      events.push(event);
    }

    // Should have two delta events and one done
    expect(events).toHaveLength(3);
    expect(events[0]).toEqual({ type: "delta", text: "Hello" });
    expect(events[1]).toEqual({ type: "delta", text: " world" });
    const doneEvent = events[2];
    expect(doneEvent.type).toBe("done");
    if (doneEvent.type === "done") {
      expect(doneEvent.fullText).toBe("Hello world");
      expect(doneEvent.allComponents).toEqual([]);
    }
    expect(noopProcess).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// Test: stream parser with text + tool_use
// ---------------------------------------------------------------------------

describe("parseAnthropicStream - text + tool_use", () => {
  it("yields delta + tool_result + done with allComponents", async () => {
    const mockComponents: InlineComponent[] = [
      { type: "keyboard_guide", config: { title: "Scale", description: "C major", hands: "right" } },
    ];
    const processToolFn = vi.fn().mockResolvedValue({
      name: "keyboard_guide",
      componentsJson: mockComponents,
      isError: false,
    });

    const sse = sseLines(
      {
        event: "message_start",
        data: { type: "message_start", message: { id: "msg_2" } },
      },
      // Text block starts
      {
        event: "content_block_start",
        data: { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
      },
      {
        event: "content_block_delta",
        data: { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Try this:" } },
      },
      {
        event: "content_block_stop",
        data: { type: "content_block_stop", index: 0 },
      },
      // Tool use block starts
      {
        event: "content_block_start",
        data: {
          type: "content_block_start",
          index: 1,
          content_block: { type: "tool_use", id: "tool_abc", name: "keyboard_guide" },
        },
      },
      {
        event: "content_block_delta",
        data: {
          type: "content_block_delta",
          index: 1,
          delta: { type: "input_json_delta", partial_json: '{"title":"Scale","description":"C major",' },
        },
      },
      {
        event: "content_block_delta",
        data: {
          type: "content_block_delta",
          index: 1,
          delta: { type: "input_json_delta", partial_json: '"hands":"right"}' },
        },
      },
      {
        event: "content_block_stop",
        data: { type: "content_block_stop", index: 1 },
      },
      {
        event: "message_stop",
        data: { type: "message_stop" },
      },
    );

    const stream = makeStream(sse);
    const events = [];
    for await (const event of parseAnthropicStream(stream, processToolFn)) {
      events.push(event);
    }

    // delta(text) + tool_result + done
    expect(events).toHaveLength(3);
    expect(events[0]).toEqual({ type: "delta", text: "Try this:" });

    const toolEvent = events[1];
    expect(toolEvent.type).toBe("tool_result");
    if (toolEvent.type === "tool_result") {
      expect(toolEvent.name).toBe("keyboard_guide");
      expect(toolEvent.componentsJson).toEqual(mockComponents);
    }

    const doneEvent = events[2];
    expect(doneEvent.type).toBe("done");
    if (doneEvent.type === "done") {
      expect(doneEvent.fullText).toBe("Try this:");
      expect(doneEvent.allComponents).toEqual(mockComponents);
    }

    // processToolFn should have been called once with parsed JSON input
    expect(processToolFn).toHaveBeenCalledOnce();
    expect(processToolFn).toHaveBeenCalledWith("keyboard_guide", {
      title: "Scale",
      description: "C major",
      hands: "right",
    });
  });
});

// ---------------------------------------------------------------------------
// Test: stream parser with failed tool (isError: true)
// ---------------------------------------------------------------------------

describe("parseAnthropicStream - failed tool", () => {
  it("does NOT yield tool_result when processToolFn returns isError: true", async () => {
    const processToolFn = vi.fn().mockResolvedValue({
      name: "create_exercise",
      componentsJson: [],
      isError: true,
    });

    const sse = sseLines(
      {
        event: "message_start",
        data: { type: "message_start", message: { id: "msg_3" } },
      },
      {
        event: "content_block_start",
        data: { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
      },
      {
        event: "content_block_delta",
        data: { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Some text" } },
      },
      {
        event: "content_block_stop",
        data: { type: "content_block_stop", index: 0 },
      },
      {
        event: "content_block_start",
        data: {
          type: "content_block_start",
          index: 1,
          content_block: { type: "tool_use", id: "tool_xyz", name: "create_exercise" },
        },
      },
      {
        event: "content_block_delta",
        data: {
          type: "content_block_delta",
          index: 1,
          delta: { type: "input_json_delta", partial_json: '{"source_passage":"bar 1","target_skill":"legato","exercises":[]}' },
        },
      },
      {
        event: "content_block_stop",
        data: { type: "content_block_stop", index: 1 },
      },
      {
        event: "message_stop",
        data: { type: "message_stop" },
      },
    );

    const stream = makeStream(sse);
    const events = [];
    for await (const event of parseAnthropicStream(stream, processToolFn)) {
      events.push(event);
    }

    // Only delta + done -- no tool_result because isError: true
    expect(events).toHaveLength(2);
    expect(events[0]).toEqual({ type: "delta", text: "Some text" });
    expect(events[1].type).toBe("done");

    const doneEvent = events[1];
    if (doneEvent.type === "done") {
      // allComponents should be empty since the tool failed
      expect(doneEvent.allComponents).toEqual([]);
    }
  });
});

// ---------------------------------------------------------------------------
// Test: split TCP chunks — SSE boundary falls across two read() calls
// ---------------------------------------------------------------------------

describe("parseAnthropicStream - split TCP chunks", () => {
  it("buffers partial SSE messages and produces correct events when boundary is split", async () => {
    const encoder = new TextEncoder();

    // Build the full SSE sequence as individual event strings
    const blockStartEvent = `event: content_block_start\ndata: ${JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } })}\n\n`;
    const deltaEvent = `event: content_block_delta\ndata: ${JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "split" } })}\n\n`;
    const blockStopEvent = `event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: 0 })}\n\n`;
    const messageStopEvent = `event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`;

    // Chunk 1: complete block_start + partial delta (cut before closing \n\n)
    const fullDelta = deltaEvent;
    const splitPoint = fullDelta.indexOf('", "index"') + 5; // cut mid-data
    const chunk1 = blockStartEvent + fullDelta.slice(0, splitPoint);
    const chunk2 = fullDelta.slice(splitPoint) + blockStopEvent + messageStopEvent;

    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(chunk1));
        controller.enqueue(encoder.encode(chunk2));
        controller.close();
      },
    });

    const noopProcess = vi.fn().mockResolvedValue({ name: "noop", componentsJson: [], isError: false });

    const events = [];
    for await (const event of parseAnthropicStream(stream, noopProcess)) {
      events.push(event);
    }

    // Should have one delta and one done, even though the delta arrived split across chunks
    const deltaEvents = events.filter((e) => e.type === "delta");
    expect(deltaEvents).toHaveLength(1);
    expect(deltaEvents[0]).toEqual({ type: "delta", text: "split" });

    const doneEvent = events.find((e) => e.type === "done");
    expect(doneEvent).toBeDefined();
    if (doneEvent && doneEvent.type === "done") {
      expect(doneEvent.fullText).toBe("split");
    }
  });
});

// ---------------------------------------------------------------------------
// Test: stripAnalysis
// ---------------------------------------------------------------------------

describe("stripAnalysis", () => {
  it("strips a single analysis block", () => {
    const input = "<analysis>This is my reasoning.</analysis>Here is my response.";
    expect(stripAnalysis(input)).toBe("Here is my response.");
  });

  it("strips multiple analysis blocks", () => {
    const input = "<analysis>First reasoning block.</analysis>Middle text.<analysis>Second reasoning block.</analysis>Final response.";
    expect(stripAnalysis(input)).toBe("Middle text.Final response.");
  });

  it("returns text unchanged when no analysis block", () => {
    const input = "Just a normal response with no analysis tags.";
    expect(stripAnalysis(input)).toBe("Just a normal response with no analysis tags.");
  });

  it("handles empty analysis block", () => {
    const input = "<analysis></analysis>The response.";
    expect(stripAnalysis(input)).toBe("The response.");
  });

  it("handles multiline analysis block", () => {
    const input = "<analysis>\nLine 1\nLine 2\n</analysis>\nThe response.";
    expect(stripAnalysis(input)).toBe("The response.");
  });

  it("trims surrounding whitespace", () => {
    const input = "  <analysis>reasoning</analysis>  response  ";
    expect(stripAnalysis(input)).toBe("response");
  });
});

import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { practiceApi } from "../lib/practice-api";
import { usePracticeSession } from "./usePracticeSession";

vi.mock("../lib/practice-api", () => ({
	practiceApi: {
		start: vi.fn(),
		uploadChunk: vi.fn(),
		connectWebSocket: vi.fn(),
	},
}));

class FakeAudioContext {
	state = "running";
	createMediaStreamSource() {
		return { connect: vi.fn() };
	}
	createAnalyser() {
		return { fftSize: 256, getByteFrequencyData: vi.fn() };
	}
	close() {
		this.state = "closed";
		return Promise.resolve();
	}
}

class FakeMediaRecorder {
	state = "inactive";
	ondataavailable: ((e: { data: Blob }) => void) | null = null;
	start() {
		this.state = "recording";
	}
	stop() {
		this.state = "inactive";
	}
}

interface FakeSocket {
	readyState: number;
	onopen: (() => void) | null;
	onmessage: ((e: MessageEvent) => void) | null;
	onerror: (() => void) | null;
	onclose: (() => void) | null;
	send: ReturnType<typeof vi.fn>;
	close: ReturnType<typeof vi.fn>;
}

function createFakeSocket(): FakeSocket {
	return {
		readyState: WebSocket.OPEN,
		onopen: null,
		onmessage: null,
		onerror: null,
		onclose: null,
		send: vi.fn(),
		close: vi.fn(),
	};
}

describe("usePracticeSession — review event (#162)", () => {
	let socket: FakeSocket;

	beforeEach(() => {
		socket = createFakeSocket();
		vi.mocked(practiceApi.start).mockResolvedValue({
			sessionId: "s1",
			conversationId: "c1",
		});
		vi.mocked(practiceApi.connectWebSocket).mockImplementation(() => {
			queueMicrotask(() => socket.onopen?.());
			return socket as unknown as WebSocket;
		});
		vi.stubGlobal("AudioContext", FakeAudioContext);
		vi.stubGlobal("MediaRecorder", FakeMediaRecorder);
		Object.defineProperty(navigator, "mediaDevices", {
			configurable: true,
			value: {
				getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [] }),
			},
		});
	});

	afterEach(() => {
		vi.unstubAllGlobals();
		vi.clearAllMocks();
	});

	it("sets verdict, carryForward, and noCommentZones from a review message", async () => {
		const { result } = renderHook(() => usePracticeSession());

		await act(async () => {
			await result.current.start();
		});
		await waitFor(() => expect(result.current.state).toBe("recording"));

		act(() => {
			socket.onmessage?.({
				data: JSON.stringify({
					type: "review",
					verdict: "Solid session, phrasing held together well.",
					carry_forward: "Watch the left-hand tempo in the coda.",
					no_comment_zones: [{ atSeconds: 45, reason: "insufficient_notes" }],
				}),
			} as MessageEvent);
		});

		expect(result.current.verdict).toBe(
			"Solid session, phrasing held together well.",
		);
		expect(result.current.carryForward).toBe(
			"Watch the left-hand tempo in the coda.",
		);
		expect(result.current.noCommentZones).toEqual([
			{ atSeconds: 45, reason: "insufficient_notes" },
		]);
	});
});

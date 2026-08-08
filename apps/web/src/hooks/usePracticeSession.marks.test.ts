import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";
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
		// getByteFrequencyData is required: useAudioActivity's rAF-driven energy
		// poll calls it every frame once the analyser is wired up, and that loop
		// keeps running (and throwing into an unhandled-rejection) past this
		// test's own assertions if the fake doesn't implement it.
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

describe("usePracticeSession marks", () => {
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

	it("appends a mark to state when a mark WS event arrives", async () => {
		const { result } = renderHook(() => usePracticeSession());

		await act(async () => {
			await result.current.start();
		});
		await waitFor(() => expect(result.current.state).toBe("recording"));

		const mark: Mark = {
			id: "m1",
			anchor: resolveAnchor({ atSeconds: 12, alignmentQuality: 0 }),
			taxonomy: "needs_work",
			dimension: "pedaling",
			evidence: "test evidence",
			lifecycle: "active",
		};

		act(() => {
			socket.onmessage?.({
				data: JSON.stringify({ type: "mark", mark }),
			} as MessageEvent);
		});

		expect(result.current.marks).toEqual([mark]);
	});
});

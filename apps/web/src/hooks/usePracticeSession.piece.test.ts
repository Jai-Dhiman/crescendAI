// apps/web/src/hooks/usePracticeSession.piece.test.ts
//
// Harness copied from usePracticeSession.sessionid.test.ts -- the established
// way to drive WS events in this codebase. Events arrive through the fake
// socket's onmessage, exactly as they do in the browser; the hook gets no
// test-only export.
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

describe("usePracticeSession -- piece identity", () => {
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

	it("surfaces piece_identified as a ConfidentGuess carrying the catalog pieceId", async () => {
		const { result } = renderHook(() => usePracticeSession());

		expect(result.current.confidentGuess).toBeNull();

		await act(async () => {
			await result.current.start();
		});
		await waitFor(() => expect(result.current.state).toBe("recording"));

		act(() => {
			socket.onmessage?.({
				data: JSON.stringify({
					type: "piece_identified",
					pieceId: "chopin-nocturne-op9-no2",
					composer: "Chopin",
					title: "Nocturne Op. 9 No. 2",
					confidence: 0.92,
					method: "chroma",
				}),
			} as MessageEvent);
		});

		await waitFor(() =>
			expect(result.current.confidentGuess).toEqual({
				pieceId: "chopin-nocturne-op9-no2",
				composer: "Chopin",
				title: "Nocturne Op. 9 No. 2",
				confidence: 0.92,
			}),
		);
	});
});

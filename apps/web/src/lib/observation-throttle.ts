import type { ObservationEvent } from "./practice-api";

/**
 * Simplified observation queue. The DO owns pacing decisions;
 * the client just queues for reconnection resilience and drains on session end.
 */
export class ObservationThrottle {
	private chunksReceived = 0;

	enqueue(obs: ObservationEvent): ObservationEvent {
		// DO controls pacing -- deliver immediately
		this.chunksReceived++;
		return obs;
	}

	onChunkProcessed(): null {
		this.chunksReceived++;
		return null;
	}

	drain(): ObservationEvent[] {
		return [];
	}

	reset(): void {
		this.chunksReceived = 0;
	}

	getChunksReceived(): number {
		return this.chunksReceived;
	}
}

import type { ObservationEvent } from "./practice-api";

/**
 * Simplified observation queue. The DO owns pacing decisions;
 * the client just queues for reconnection resilience and drains on session end.
 */
export class ObservationThrottle {
	private queued: ObservationEvent | null = null;
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
		if (this.queued) {
			const obs = this.queued;
			this.queued = null;
			return [obs];
		}
		return [];
	}

	reset(): void {
		this.queued = null;
		this.chunksReceived = 0;
	}

	getChunksReceived(): number {
		return this.chunksReceived;
	}
}

import type { ObservationEvent } from "./practice-api";

interface ObservationThrottleOptions {
	windowMs?: number;
	minChunksBeforeFirst?: number;
}

export class ObservationThrottle {
	private readonly windowMs: number;
	private readonly minChunksBeforeFirst: number;
	private lastDeliveredAt = 0;
	private queued: ObservationEvent | null = null;
	private chunksReceived = 0;

	constructor(options?: ObservationThrottleOptions) {
		this.windowMs = options?.windowMs ?? 180_000;
		this.minChunksBeforeFirst = options?.minChunksBeforeFirst ?? 4;
	}

	enqueue(obs: ObservationEvent): ObservationEvent | null {
		if (this.canDeliver()) {
			this.lastDeliveredAt = Date.now();
			return obs;
		}
		// Queue it (replace any existing queued observation)
		this.queued = obs;
		return null;
	}

	onChunkProcessed(): ObservationEvent | null {
		this.chunksReceived++;
		return this.tryRelease();
	}

	tick(): ObservationEvent | null {
		return this.tryRelease();
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
		this.lastDeliveredAt = 0;
		this.queued = null;
		this.chunksReceived = 0;
	}

	private canDeliver(): boolean {
		if (this.chunksReceived < this.minChunksBeforeFirst) return false;
		if (this.lastDeliveredAt === 0) return true;
		return Date.now() - this.lastDeliveredAt >= this.windowMs;
	}

	private tryRelease(): ObservationEvent | null {
		if (this.queued && this.canDeliver()) {
			const obs = this.queued;
			this.queued = null;
			this.lastDeliveredAt = Date.now();
			return obs;
		}
		return null;
	}
}

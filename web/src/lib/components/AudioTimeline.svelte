<script lang="ts">
import { Play, Pause, Volume2, VolumeX } from 'lucide-svelte';
import type { TemporalFeedbackItem } from '$lib/types/analysis';

let { 
	audioUrl = $bindable(''),
	temporalFeedback = [] as TemporalFeedbackItem[],
	onSeek = (time: number) => {}
} = $props<{
	audioUrl?: string;
	temporalFeedback?: TemporalFeedbackItem[];
	onSeek?: (time: number) => void;
}>();

let audioElement: HTMLAudioElement | null = $state(null);
let isPlaying = $state(false);
let currentTime = $state(0);
let duration = $state(0);
let volume = $state(1);
let isMuted = $state(false);
let isLoading = $state(true);
let hasError = $state(false);

// Parse timestamp string like "0:00-0:03" to seconds
function parseTimestamp(timestamp: string): { start: number; end: number } {
	const [startStr, endStr] = timestamp.split('-');
	const parseTime = (time: string) => {
		const parts = time.trim().split(':');
		if (parts.length === 2) {
			return parseInt(parts[0]) * 60 + parseInt(parts[1]);
		}
		return 0;
	};
	return { start: parseTime(startStr), end: parseTime(endStr) };
}

// Get segments from temporal feedback with scores (excluding zeros)
const segments = $derived(
	temporalFeedback.map((item: TemporalFeedbackItem, index: number) => {
		const { start, end } = parseTimestamp(item.timestamp);
		// Calculate a simple importance score based on insight count
		const score = item.insights.length;
		return { start, end, index, score };
	})
);

function togglePlay() {
	if (!audioElement) return;
	if (isPlaying) {
		audioElement.pause();
	} else {
		audioElement.play();
	}
}

function toggleMute() {
	if (!audioElement) return;
	isMuted = !isMuted;
	audioElement.muted = isMuted;
}

function handleTimeUpdate() {
	if (audioElement) {
		currentTime = audioElement.currentTime;
	}
}

function handleLoadedMetadata() {
	if (audioElement) {
		duration = audioElement.duration;
		isLoading = false;
	}
}

function handleError() {
	hasError = true;
	isLoading = false;
}

function handleSeek(event: MouseEvent) {
	const target = event.currentTarget as HTMLElement;
	const rect = target.getBoundingClientRect();
	const x = event.clientX - rect.left;
	const percentage = x / rect.width;
	const time = percentage * duration;
	
	if (audioElement) {
		audioElement.currentTime = time;
		currentTime = time;
	}
	onSeek(time);
}

function handleVolumeChange(event: Event) {
	const target = event.target as HTMLInputElement;
	volume = parseFloat(target.value);
	if (audioElement) {
		audioElement.volume = volume;
	}
}

function seekToSegment(segmentStart: number) {
	if (audioElement) {
		audioElement.currentTime = segmentStart;
		currentTime = segmentStart;
		if (!isPlaying) {
			audioElement.play();
		}
	}
	onSeek(segmentStart);
}

function formatTime(seconds: number): string {
	const mins = Math.floor(seconds / 60);
	const secs = Math.floor(seconds % 60);
	return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Check which segment is currently playing
function isSegmentActive(segment: { start: number; end: number }): boolean {
	return currentTime >= segment.start && currentTime <= segment.end;
}
</script>

<div class="audio-timeline-container">
	{#if hasError}
		<div class="audio-error">
			<p class="error-text">Unable to load audio file</p>
		</div>
	{:else if !audioUrl}
		<div class="audio-placeholder">
			<p class="placeholder-text">No audio file available for playback</p>
		</div>
	{:else}
		<audio
			bind:this={audioElement}
			src={audioUrl}
			onplay={() => (isPlaying = true)}
			onpause={() => (isPlaying = false)}
			ontimeupdate={handleTimeUpdate}
			onloadedmetadata={handleLoadedMetadata}
			onerror={handleError}
		></audio>
		
		<!-- Timeline Visualization -->
		<div class="timeline-visualization">
			<div class="timeline-track" role="progressbar" aria-valuemin="0" aria-valuemax={duration} aria-valuenow={currentTime}>
				<!-- Background segments -->
				{#each segments as segment}
					<button
						class="timeline-segment"
						class:active={isSegmentActive(segment)}
						class:high-importance={segment.score >= 2}
						style="left: {(segment.start / duration) * 100}%; width: {((segment.end - segment.start) / duration) * 100}%"
						onclick={() => seekToSegment(segment.start)}
						aria-label="Jump to {segment.start} seconds"
					></button>
				{/each}
				
				<!-- Progress indicator -->
				<div class="timeline-progress" style="width: {(currentTime / duration) * 100}%"></div>
				
				<!-- Clickable overlay -->
				<button 
					class="timeline-overlay" 
					onclick={handleSeek}
					aria-label="Seek to position"
				></button>
			</div>
		</div>
		
		<!-- Playback Controls -->
		<div class="audio-controls">
			<div class="control-group">
				<button 
					class="control-button play-button" 
					onclick={togglePlay}
					disabled={isLoading}
					aria-label={isPlaying ? 'Pause' : 'Play'}
				>
					{#if isPlaying}
						<Pause size={24} />
					{:else}
						<Play size={24} />
					{/if}
				</button>
				
				<div class="time-display">
					<span class="time-current">{formatTime(currentTime)}</span>
					<span class="time-separator">/</span>
					<span class="time-total">{formatTime(duration)}</span>
				</div>
			</div>
			
			<div class="control-group volume-group">
				<button 
					class="control-button volume-button" 
					onclick={toggleMute}
					aria-label={isMuted ? 'Unmute' : 'Mute'}
				>
					{#if isMuted || volume === 0}
						<VolumeX size={20} />
					{:else}
						<Volume2 size={20} />
					{/if}
				</button>
				<input
					type="range"
					class="volume-slider"
					min="0"
					max="1"
					step="0.01"
					value={volume}
					oninput={handleVolumeChange}
					aria-label="Volume"
				/>
			</div>
		</div>
	{/if}
</div>

<style>
.audio-timeline-container {
	background: rgba(255, 255, 255, 0.9);
	backdrop-filter: blur(20px);
	border: 1px solid rgba(201, 168, 118, 0.2);
	border-radius: 16px;
	padding: 1.5rem;
	box-shadow: 0 8px 32px rgba(30, 39, 73, 0.08);
}

.audio-error,
.audio-placeholder {
	padding: 2rem;
	text-align: center;
	color: var(--subtle-gray);
	font-family: var(--font-serif);
}

.timeline-visualization {
	margin-bottom: 1.5rem;
}

.timeline-track {
	position: relative;
	height: 48px;
	background: rgba(201, 168, 118, 0.1);
	border-radius: 8px;
	overflow: hidden;
	cursor: pointer;
}

.timeline-segment {
	position: absolute;
	top: 0;
	height: 100%;
	background: rgba(201, 168, 118, 0.2);
	border: none;
	cursor: pointer;
	transition: all 0.2s ease;
	padding: 0;
}

.timeline-segment:hover {
	background: rgba(201, 168, 118, 0.35);
}

.timeline-segment.active {
	background: rgba(30, 39, 73, 0.15);
}

.timeline-segment.high-importance {
	background: rgba(201, 168, 118, 0.3);
	border-top: 2px solid var(--accent-gold);
}

.timeline-segment.high-importance:hover {
	background: rgba(201, 168, 118, 0.5);
}

.timeline-progress {
	position: absolute;
	top: 0;
	left: 0;
	height: 100%;
	background: linear-gradient(90deg, var(--accent-gold), var(--warm-bronze));
	pointer-events: none;
	border-radius: 8px 0 0 8px;
	transition: width 0.1s linear;
}

.timeline-overlay {
	position: absolute;
	top: 0;
	left: 0;
	width: 100%;
	height: 100%;
	background: transparent;
	border: none;
	cursor: pointer;
	padding: 0;
}

.audio-controls {
	display: flex;
	align-items: center;
	justify-content: space-between;
	gap: 1rem;
}

.control-group {
	display: flex;
	align-items: center;
	gap: 0.75rem;
}

.control-button {
	display: flex;
	align-items: center;
	justify-content: center;
	background: var(--rich-navy);
	color: white;
	border: none;
	border-radius: 50%;
	cursor: pointer;
	transition: all 0.2s ease;
}

.play-button {
	width: 48px;
	height: 48px;
}

.play-button:hover:not(:disabled) {
	background: var(--deep-forest);
	transform: scale(1.05);
}

.play-button:disabled {
	opacity: 0.5;
	cursor: not-allowed;
}

.volume-button {
	width: 36px;
	height: 36px;
}

.volume-button:hover {
	background: var(--deep-forest);
}

.time-display {
	font-family: var(--font-mono);
	font-size: 0.875rem;
	color: var(--charcoal-text);
	display: flex;
	gap: 0.25rem;
}

.time-separator {
	color: var(--subtle-gray);
}

.volume-group {
	gap: 0.5rem;
}

.volume-slider {
	width: 80px;
	height: 4px;
	-webkit-appearance: none;
	appearance: none;
	background: rgba(201, 168, 118, 0.2);
	border-radius: 2px;
	outline: none;
}

.volume-slider::-webkit-slider-thumb {
	-webkit-appearance: none;
	appearance: none;
	width: 14px;
	height: 14px;
	background: var(--accent-gold);
	cursor: pointer;
	border-radius: 50%;
	transition: all 0.2s ease;
}

.volume-slider::-webkit-slider-thumb:hover {
	background: var(--warm-bronze);
	transform: scale(1.15);
}

.volume-slider::-moz-range-thumb {
	width: 14px;
	height: 14px;
	background: var(--accent-gold);
	cursor: pointer;
	border-radius: 50%;
	border: none;
	transition: all 0.2s ease;
}

.volume-slider::-moz-range-thumb:hover {
	background: var(--warm-bronze);
	transform: scale(1.15);
}

@media (max-width: 768px) {
	.audio-timeline-container {
		padding: 1rem;
	}
	
	.audio-controls {
		flex-direction: column;
		gap: 1rem;
	}
	
	.volume-group {
		width: 100%;
		justify-content: center;
	}
	
	.volume-slider {
		flex: 1;
		max-width: 150px;
	}
}
</style>

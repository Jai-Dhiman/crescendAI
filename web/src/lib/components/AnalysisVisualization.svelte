<script lang="ts">
	import SketchyWaveform from './SketchyWaveform.svelte';

	interface Props {
		class?: string;
	}

	let { class: className = '' }: Props = $props();

	const dimensions = [
		"Timing Precision",
		"Articulation", 
		"Tonal Quality",
		"Dynamic Range",
		"Phrasing",
		"Rhythm Accuracy",
		"Pedal Technique",
		"Musical Expression",
		"Tempo Stability"
	];
</script>

<div class="relative {className}">
	<!-- Background waveform -->
	<div class="absolute inset-0 opacity-10">
		<SketchyWaveform class="w-full h-full text-slate-300" />
	</div>
	
	<!-- Analysis overlay -->
	<div class="relative bg-white/95 backdrop-blur-sm rounded-xl border-2 border-slate-200 p-4 sm:p-6 shadow-xl">
		<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
			{#each dimensions as dimension, index}
				<div class="flex items-center space-x-2">
					<!-- Sketchy progress indicator -->
					<div class="flex-1">
						<div class="text-xs sm:text-sm text-slate-600 mb-1 truncate">{dimension}</div>
						<div class="relative h-2 bg-slate-100 rounded-full overflow-hidden">
							<div 
								class="h-full bg-slate-600 rounded-full transition-all duration-1000"
								style="width: {65 + (index * 7) % 30}%; transform: skew(-2deg) translateY({index % 2 ? '0.5px' : '-0.5px'});"
							></div>
						</div>
					</div>
					<div class="text-xs sm:text-sm text-slate-500 font-mono flex-shrink-0">
						{Math.floor(65 + (index * 7) % 30)}%
					</div>
				</div>
			{/each}
		</div>
		
		<!-- Sketchy feedback indicators -->
		<div class="mt-6 flex justify-between items-center text-sm text-slate-500">
			<div class="flex items-center space-x-1">
				<svg width="12" height="12" viewBox="0 0 12 12" class="text-green-600">
					<path
						d="M2.1 6.2c1.2 1.1 2.4 2.1 3.7 3.1c2.8-2.9 5.7-5.8 8.5-8.7"
						stroke="currentColor"
						stroke-width="1.5"
						fill="none"
						stroke-linecap="round"
						stroke-linejoin="round"
					/>
				</svg>
				<span>Strong performance areas: 6</span>
			</div>
			<div class="flex items-center space-x-1">
				<svg width="12" height="12" viewBox="0 0 12 12" class="text-amber-600">
					<circle
						cx="6"
						cy="6"
						r="4.2"
						stroke="currentColor"
						stroke-width="1.5"
						fill="none"
					/>
					<path d="M6 3v3.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" />
					<circle cx="6" cy="8.5" r="0.5" fill="currentColor" />
				</svg>
				<span>Areas for improvement: 3</span>
			</div>
		</div>
	</div>
</div>
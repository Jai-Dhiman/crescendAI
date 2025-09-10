<script lang="ts">
	import { goto } from '$app/navigation';
	import { analysisStore } from '$lib/stores/analysis';
	import { useUploadAndAnalyzeMutation } from '$lib/hooks/useAnalysis';
	import { crescendApi, type CrescendApiError } from '$lib/services/crescendApi';
	import { Upload } from 'lucide-svelte';

	let files: FileList | null = null;
	let isDragging = false;
	let uploadError = '';
	let progress = 0;
	let currentStage: 'uploading' | 'analyzing' | 'processing' | null = null;
	let comparisonMode = false;

	const ACCEPTED_AUDIO_TYPES = [
		'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/ogg', 'audio/m4a', 'audio/aac',
		'video/mp4', 'video/mov', 'video/avi'
	];

	const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB
	
	// Create the mutation for upload and analysis
	const uploadAndAnalyzeMutation = useUploadAndAnalyzeMutation();

	function validateFile(file: File): string | null {
		if (!ACCEPTED_AUDIO_TYPES.includes(file.type) && 
			!file.name.toLowerCase().match(/\.(mp3|wav|ogg|m4a|aac|mp4|mov|avi)$/)) {
			return 'Please select an audio or video file (MP3, WAV, OGG, M4A, AAC, MP4, MOV, AVI)';
		}
		
		if (file.size > MAX_FILE_SIZE) {
			return 'File size must be less than 100MB';
		}
		
		return null;
	}

	async function processFile(file: File) {
		const error = validateFile(file);
		if (error) {
			uploadError = error;
			files = null;
			return;
		}
		
		uploadError = '';
		progress = 0;
		currentStage = 'uploading';

		try {
			let result;
			
			if (comparisonMode) {
				// Use comparison workflow
				result = await crescendApi.uploadAndCompare(
					file,
					'hybrid_ast',
					'ultra_small_ast',
					(stage, progressValue) => {
						currentStage = stage;
						if (progressValue !== undefined) {
							progress = progressValue;
						} else {
							// Set progress based on stage
							switch (stage) {
								case 'uploading':
									progress = 25;
									break;
								case 'analyzing':
									progress = 50;
									break;
								case 'processing':
									progress = 75;
									break;
							}
						}
					}
				);
			} else {
				// Use regular analysis workflow
				result = await $uploadAndAnalyzeMutation.mutateAsync({
					file,
					onProgress: (stage, progressValue) => {
						currentStage = stage;
						if (progressValue !== undefined) {
							progress = progressValue;
						} else {
							// Set progress based on stage
							switch (stage) {
								case 'uploading':
									progress = 25;
									break;
								case 'analyzing':
									progress = 50;
									break;
								case 'processing':
									progress = 75;
									break;
							}
						}
					}
				});
			}

			// Analysis complete, save result and navigate
			progress = 100;
			analysisStore.set(result);
			
			// Navigate to appropriate results page
			if (comparisonMode) {
				goto(`/results/comparison?id=${result.id}`);
			} else {
				goto('/results');
			}
			
		} catch (error) {
			console.error('Analysis failed:', error);
			currentStage = null;
			progress = 0;
			
			if (error instanceof Error) {
				uploadError = error.message;
			} else {
				uploadError = 'Analysis failed. Please try again.';
			}
		}
	}

	function handleFileSelect(event: Event) {
		const input = event.target as HTMLInputElement;
		const selectedFiles = input.files;
		if (selectedFiles && selectedFiles.length > 0) {
			files = selectedFiles;
			processFile(selectedFiles[0]);
		}
	}

	function handleDrop(event: DragEvent) {
		event.preventDefault();
		isDragging = false;
		const droppedFiles = event.dataTransfer?.files;
		if (droppedFiles && droppedFiles.length > 0) {
			files = droppedFiles;
			processFile(droppedFiles[0]);
		}
	}

	function handleDragOver(event: DragEvent) {
		event.preventDefault();
		isDragging = true;
	}

	function handleDragLeave(event: DragEvent) {
		// Only stop dragging if we're actually leaving the drop area
		if (!(event.currentTarget as HTMLElement).contains(event.relatedTarget as Node)) {
			isDragging = false;
		}
	}

	function resetUpload() {
		files = null;
		uploadError = '';
		progress = 0;
		const input = document.getElementById('audioFile') as HTMLInputElement;
		if (input) input.value = '';
	}

</script>

<svelte:head>
	<title>CrescendAI - Analyze Your Piano Playing</title>
</svelte:head>

<main class="min-h-screen bg-cream relative overflow-hidden px-6 py-8">
	<!-- Background elements -->
	<div class="absolute inset-0 opacity-5">
		<!-- Staff lines -->
		<div class="absolute top-1/4 left-0 right-0 h-px bg-charcoal transform rotate-1"></div>
		<div class="absolute top-1/3 left-0 right-0 h-px bg-charcoal transform -rotate-1"></div>
		<div class="absolute top-1/2 left-0 right-0 h-px bg-charcoal transform rotate-0.5"></div>
		<div class="absolute top-2/3 left-0 right-0 h-px bg-charcoal transform -rotate-0.5"></div>
		<div class="absolute top-3/4 left-0 right-0 h-px bg-charcoal transform rotate-1"></div>
	</div>


	<div class="max-w-4xl mx-auto min-h-screen flex flex-col justify-center items-center text-center relative z-10 mobile-padding">
		<!-- Brand Section -->
		<div class="space-y-6 md:space-y-8 mb-8 md:mb-12 mobile-spacing">
			<!-- Icon with artistic placement -->
			<div class="relative">
				<img src="/crescendai.png" alt="CrescendAI" class="sketchy-hero-icon sketchy-border sketchy-rounded w-24 h-24 md:w-32 md:h-32 hero-icon-mobile mx-auto mb-4 md:mb-6" />
			</div>

			<!-- Title with creative typography -->
			<div class="space-y-3 md:space-y-4">
				<h1 class="sketchy-title text-4xl md:text-7xl lg:text-8xl">CrescendAI</h1>
				<p class="handwritten text-lg md:text-xl lg:text-2xl text-sage max-w-lg mx-auto leading-relaxed px-4">
					~ where your piano meets AI brilliance ~
				</p>
			</div>
		</div>

		<!-- Mode Selection -->
		<div class="mb-6 space-y-4">
			<div class="flex items-center justify-center gap-6 p-4 sketchy-card bg-white/50">
				<!-- Single Analysis Option -->
				<label class="flex items-center gap-3 cursor-pointer">
					<input
						type="radio"
						name="analysisMode"
						value="single"
						checked={!comparisonMode}
						class="sketchy-radio"
						on:change={() => comparisonMode = false}
					/>
					<div class="text-center">
						<div class="sketchy-text font-medium">Quick Analysis</div>
						<div class="text-xs text-warm-gray">Single model analysis</div>
					</div>
				</label>

				<!-- Comparison Option -->
				<label class="flex items-center gap-3 cursor-pointer">
					<input
						type="radio"
						name="analysisMode"
						value="compare"
						checked={comparisonMode}
						class="sketchy-radio"
						on:change={() => comparisonMode = true}
					/>
					<div class="text-center">
						<div class="sketchy-text font-medium">A/B Comparison</div>
						<div class="text-xs text-warm-gray">Two models side-by-side</div>
					</div>
				</label>
			</div>

			{#if comparisonMode}
				<div class="text-center p-3 bg-sage/10 sketchy-border sketchy-rounded">
					<p class="handwritten text-sm text-sage">
						✨ Compare Hybrid AST vs Ultra-Small AST models
					</p>
				</div>
			{/if}
		</div>

		<!-- Upload Section -->
		<div class="max-w-2xl mx-auto">
			<input
				type="file"
				id="audioFile"
				accept="audio/*,video/*"
				class="hidden"
				on:change={handleFileSelect}
			/>
			
			<!-- Main upload area with creative design -->
			<div 
				class="sketchy-upload-area {isDragging ? 'drag-over' : ''} cursor-pointer relative"
				on:drop={handleDrop}
				on:dragover={handleDragOver}
				on:dragleave={handleDragLeave}
				on:click={() => document.getElementById('audioFile')?.click()}
				on:keydown={(e) => {
					if (e.key === 'Enter' || e.key === ' ') {
						e.preventDefault();
						document.getElementById('audioFile')?.click();
					}
				}}
				tabindex="0"
				role="button"
				aria-label="Upload audio file"
			>

				<div class="space-y-6 py-4">
					<!-- Central illustration -->
					<div class="relative mx-auto w-24 h-24">
						<Upload class="sketchy-icon w-full h-full" aria-hidden="true" />
					</div>

					<div class="space-y-3">
						<h2 class="sketchy-text text-xl md:text-2xl font-medium">
							Drop Your Performance Here
						</h2>
						<p class="sketchy-text text-warm-gray leading-relaxed px-2 md:px-4 text-sm md:text-base">
							Drag & drop your recording or 
							<span class="marker-highlight handwritten text-sage font-medium">click anywhere</span> to browse
						</p>
						<p class="text-xs md:text-sm text-warm-gray opacity-75 px-2">
							Supports MP3, WAV, M4A, MP4 • Max 100MB
						</p>
					</div>

					{#if progress > 0 && progress < 100}
						<div class="mt-6 space-y-4">
						<p class="sketchy-text text-sm text-charcoal text-center">
							{#if currentStage === 'uploading'}
								Uploading your masterpiece...
							{:else if currentStage === 'analyzing'}
								AI is listening...
							{:else if currentStage === 'processing'}
								Creating your analysis...
							{/if}
						</p>
							<div class="sketchy-progress max-w-sm mx-auto">
								<div style={`width: ${progress}%`}></div>
							</div>
							<p class="handwritten text-sm text-warm-gray text-center">{progress}% complete</p>
						</div>
					{/if}
				</div>
			</div>

			<!-- Error Display -->
			{#if uploadError}
				<div class="card p-4 border-red-400 bg-red-50 mt-6 max-w-lg mx-auto">
					<div class="flex items-center justify-between gap-4">
						<p class="sketchy-text text-red-800 flex-1">{uploadError}</p>
						<button on:click={resetUpload} class="btn-primary bg-red-500 border-red-500 hover:bg-red-600 hover:border-red-600 text-sm px-3 py-1">
							try again
						</button>
					</div>
				</div>
			{/if}
		</div>

		<!-- Footer -->
		<div class="absolute bottom-4 md:bottom-8 left-1/2 transform -translate-x-1/2 text-center px-4">
			<div class="sketchy-text text-xs text-warm-gray opacity-50 mt-1">
				CrescendAI • 2025
			</div>
		</div>
	</div>
</main>
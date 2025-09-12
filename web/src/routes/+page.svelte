<script lang="ts">
import { goto } from "$app/navigation";
import { browser } from "$app/environment";
import { analysisStore } from "$lib/stores/analysis";
import { useUploadAndAnalyzeMutation } from "$lib/hooks/useAnalysis";
import { crescendApi, type CrescendApiError } from "$lib/services/crescendApi";
import { Upload } from "lucide-svelte";

let files: FileList | null = $state(null);
let isDragging = $state(false);
let uploadError = $state("");
let progress = $state(0);
let currentStage: "uploading" | "analyzing" | "processing" | null =
	$state(null);
let comparisonMode = $state(false);

const ACCEPTED_AUDIO_TYPES = [
	"audio/mpeg",
	"audio/mp3",
	"audio/wav",
	"audio/ogg",
	"audio/m4a",
	"audio/aac",
	"video/mp4",
	"video/mov",
	"video/avi",
];

const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB

// Create the mutation for upload and analysis (lazily initialized)
let uploadAndAnalyzeMutation: ReturnType<
	typeof useUploadAndAnalyzeMutation
> | null = $state(null);

// Initialize mutation when needed and context is available
function getUploadMutation() {
	if (!browser) return null;
	if (!uploadAndAnalyzeMutation) {
		try {
			uploadAndAnalyzeMutation = useUploadAndAnalyzeMutation();
		} catch (error) {
			console.error("Failed to initialize upload mutation:", error);
			return null;
		}
	}
	return uploadAndAnalyzeMutation;
}

function validateFile(file: File): string | null {
	if (
		!ACCEPTED_AUDIO_TYPES.includes(file.type) &&
		!file.name.toLowerCase().match(/\.(mp3|wav|ogg|m4a|aac|mp4|mov|avi)$/)
	) {
		return "Please select an audio or video file (MP3, WAV, OGG, M4A, AAC, MP4, MOV, AVI)";
	}

	if (file.size > MAX_FILE_SIZE) {
		return "File size must be less than 100MB";
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

	uploadError = "";
	progress = 0;
	currentStage = "uploading";

	try {
		let result;

		if (comparisonMode) {
			// Use comparison workflow
			result = await crescendApi.uploadAndCompare(
				file,
				"hybrid_ast",
				"ultra_small_ast",
				(stage, progressValue) => {
					currentStage = stage;
					if (progressValue !== undefined) {
						progress = progressValue;
					} else {
						// Set progress based on stage
						switch (stage) {
							case "uploading":
								progress = 25;
								break;
							case "analyzing":
								progress = 50;
								break;
							case "processing":
								progress = 75;
								break;
						}
					}
				},
			);
		} else {
			// Use regular analysis workflow - try mutation first, fallback to direct API
			const mutation = getUploadMutation();
			if (mutation) {
				result = await mutation.mutateAsync({
					file,
					onProgress: (stage, progressValue) => {
						currentStage = stage;
						if (progressValue !== undefined) {
							progress = progressValue;
						} else {
							// Set progress based on stage
							switch (stage) {
								case "uploading":
									progress = 25;
									break;
								case "analyzing":
									progress = 50;
									break;
								case "processing":
									progress = 75;
									break;
							}
						}
					},
				});
			} else {
				// Fallback to direct API call
				result = await crescendApi.uploadAndAnalyze(
					file,
					(stage, progressValue) => {
						currentStage = stage;
						if (progressValue !== undefined) {
							progress = progressValue;
						} else {
							// Set progress based on stage
							switch (stage) {
								case "uploading":
									progress = 25;
									break;
								case "analyzing":
									progress = 50;
									break;
								case "processing":
									progress = 75;
									break;
							}
						}
					},
				);
			}
		}

		// Analysis complete, save result and navigate
		progress = 100;
		analysisStore.set(result);

		// Navigate to appropriate results page
		if (comparisonMode) {
			goto(`/results/comparison?id=${result.id}`);
		} else {
			goto("/results");
		}
	} catch (error) {
		console.error("Analysis failed:", error);
		currentStage = null;
		progress = 0;

		if (error instanceof Error) {
			uploadError = error.message;
		} else {
			uploadError = "Analysis failed. Please try again.";
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
	if (
		!(event.currentTarget as HTMLElement).contains(event.relatedTarget as Node)
	) {
		isDragging = false;
	}
}

function resetUpload() {
	files = null;
	uploadError = "";
	progress = 0;
	const input = document.getElementById("audioFile") as HTMLInputElement;
	if (input) input.value = "";
}
</script>

<svelte:head>
	<title>CrescendAI - Academic Piano Performance Analysis</title>
</svelte:head>

<main class="min-h-screen relative overflow-hidden px-6 py-6">
	<!-- Subtle academic border elements -->
	<div class="absolute inset-0 pointer-events-none">
		<div class="absolute top-8 left-8 w-20 h-20 border border-accent-gold opacity-20 rounded-full"></div>
		<div class="absolute top-16 right-12 w-12 h-12 border border-accent-gold opacity-15 rotate-45"></div>
		<div class="absolute bottom-20 left-16 w-16 h-16 border border-muted-sage opacity-20 rounded-full"></div>
	</div>


	<div class="max-w-5xl mx-auto flex flex-col items-center justify-center text-center relative z-10 py-4 min-h-screen">
		<!-- Header Section -->
		<header class="relative space-y-3 mb-6">
			<!-- Header staff lines -->
			<div class="absolute inset-0 opacity-[0.06] pointer-events-none">
				<div class="staff-line" style="top: 20%"></div>
				<div class="staff-line" style="top: 32%"></div>
				<div class="staff-line" style="top: 44%"></div>
				<div class="staff-line" style="top: 56%"></div>
				<div class="staff-line" style="top: 68%"></div>
			</div>
			<!-- Logo and Institution -->
			<div class="relative mb-2 z-10">
				<img src="/crescendai.png" alt="CrescendAI" class="academic-logo w-16 h-16 mx-auto object-contain" />
			</div>

			<!-- Academic Title Block -->
			<div class="space-y-3 relative z-10">
				<div class="title-veil mx-auto inline-block">
					<h1 class="academic-title text-rich-navy">CrescendAI</h1>
				</div>
				
				<div class="max-w-2xl mx-auto">
					<p class="academic-subtitle text-lg md:text-xl text-subtle-gray leading-relaxed">
						Advanced Piano Performance Analysis using Audio Spectrogram Transformers
					</p>
				</div>
			</div>
		</header>

		<!-- Analysis Method Selection -->
		<section class="mb-8 mx-auto" style="margin-bottom: 1rem; max-width: 24rem;">
			<div class="card" style="padding: 0.5rem;">
				<h3 class="text-xl font-semibold text-charcoal-text mb-3 text-center" style="padding-top: 0.25rem;">Analysis Method</h3>
				
				<div class="grid grid-cols-2 gap-2">
					<!-- Standard Analysis -->
					<label class="analysis-option group cursor-pointer">
						<input
							type="radio"
							name="analysisMode"
							value="single"
							checked={!comparisonMode}
							class="sr-only"
							onchange={() => comparisonMode = false}
						/>
						<div class="analysis-card" style="padding: 0.5rem;">
							<div class="flex items-center justify-between mb-2">
								<h4 class="font-medium text-sm text-charcoal-text">Standard Analysis</h4>
								<div class="radio-indicator" style="width: 18px; height: 18px;"></div>
							</div>
							<p class="text-xs text-subtle-gray" style="line-height: 1.3;">
								Comprehensive single-model evaluation
							</p>
						</div>
					</label>

					<!-- Comparative Analysis -->
					<label class="analysis-option group cursor-pointer">
						<input
							type="radio"
							name="analysisMode"
							value="compare"
							checked={comparisonMode}
							class="sr-only"
							onchange={() => comparisonMode = true}
						/>
						<div class="analysis-card" style="padding: 0.5rem;">
							<div class="flex items-center justify-between mb-2">
								<h4 class="font-medium text-sm text-charcoal-text">Comparative Study</h4>
								<div class="radio-indicator" style="width: 18px; height: 18px;"></div>
							</div>
							<p class="text-xs text-subtle-gray" style="line-height: 1.3;">
								Side-by-side model comparison
							</p>
						</div>
					</label>
				</div>

				<!-- {#if comparisonMode}
					<div class="mt-6 p-4 bg-accent-gold/10 rounded-lg border border-accent-gold/20">
						<p class="text-sm text-warm-bronze text-center">
							<strong>Hybrid AST</strong> vs <strong>Ultra-Small AST</strong> · Comparative Performance Metrics
						</p>
					</div>
				{/if} -->
			</div>
		</section>

		<!-- Upload Section -->
		<section class="max-w-3xl mx-auto">
			<input
				type="file"
				id="audioFile"
				accept="audio/*,video/*"
				class="hidden"
				onchange={handleFileSelect}
			/>
			
			<!-- Academic Upload Interface -->
			<div 
				class="upload-zone {isDragging ? 'drag-active' : ''} cursor-pointer"
				ondrop={handleDrop}
				ondragover={handleDragOver}
				ondragleave={handleDragLeave}
				onclick={() => document.getElementById('audioFile')?.click()}
				onkeydown={(e) => {
					if (e.key === 'Enter' || e.key === ' ') {
						e.preventDefault();
						document.getElementById('audioFile')?.click();
					}
				}}
				tabindex="0"
				role="button"
				aria-label="Upload recording for analysis"
			>
				<div class="upload-content">
					<!-- Upload Icon -->
					<div class="upload-icon-wrapper">
						<Upload class="upload-icon" aria-hidden="true" />
						<div class="icon-accent"></div>
					</div>

					<!-- Upload Text -->
					<div class="upload-text">
						<h2 class="upload-title">
							Submit Recording for Analysis
						</h2>
						<p class="upload-description">
							Drag and drop your performance recording, or 
							<span class="upload-highlight">click to browse files</span>
						</p>
						<div class="upload-specs">
							<span class="spec-item">MP3, WAV, M4A, MP4</span>
							<span class="spec-divider">•</span>
							<span class="spec-item">Max 100MB</span>
							<span class="spec-divider">•</span>
							<span class="spec-item">High Quality Preferred</span>
						</div>
					</div>

					<!-- Processing State -->
					<!-- {#if progress > 0 && progress < 100}
						<div class="processing-state">
							<div class="processing-info">
								<p class="processing-label">
									{#if currentStage === 'uploading'}
										Uploading Recording
									{:else if currentStage === 'analyzing'}
										Analyzing Performance
									{:else if currentStage === 'processing'}
										Generating Report
									{/if}
								</p>
								<span class="processing-percentage">{progress}%</span>
							</div>
							<div class="progress-bar">
								<div class="progress-fill" style={`width: ${progress}%`}></div>
							</div>
						</div>
					{/if} -->
				</div>
			</div>

			<!-- Error Display -->
			<!-- {#if uploadError}
				<div class="error-card mt-6 max-w-lg mx-auto">
					<div class="flex items-start gap-4">
						<div class="error-icon">⚠</div>
						<div class="flex-1">
							<h4 class="error-title">Upload Error</h4>
							<p class="error-message">{uploadError}</p>
							<button onclick={resetUpload} class="error-retry">
								Try Again
							</button>
						</div>
					</div>
				</div>
			{/if} -->
		</section>

		<!-- Footer -->
		<footer class="mt-8 w-full flex justify-center">
			<div class="academic-footer text-center">
				<div class="footer-divider"></div>
				<p class="footer-text">
					<strong>CrescendAI</strong> · Advanced Performance Analysis · 2025
				</p>
				<p class="footer-subtitle">
					Empowering Musicians Through AI Research
				</p>
			</div>
		</footer>
	</div>
</main>
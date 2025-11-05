<script lang="ts">
import { goto } from "$app/navigation";
import { browser } from "$app/environment";
import { analysisStore } from "$lib/stores/analysis";
import { crescendApi, type CrescendApiError } from "$lib/services/crescendApi";
import { Upload, AlertTriangle } from "lucide-svelte";
import BeginnerChat from "$lib/components/chat/BeginnerChat.svelte";

let files: FileList | null = $state(null);
let isDragging = $state(false);
let uploadError = $state("");
let progress = $state(0);
let currentStage: "uploading" | "analyzing" | "processing" | null =
	$state(null);
// Comparison mode temporarily disabled (single-model only)
// let comparisonMode = $state(false);

const ACCEPTED_AUDIO_TYPES = [
	"audio/mpeg",
	"audio/mp3",
	"audio/wav",
	"audio/x-wav", // Support for .wav files that show as audio/x-wav
	"audio/ogg",
	"audio/m4a",
	"audio/aac",
	"audio/flac", // Add FLAC support as mentioned in error
	"audio/mp4", // Add audio/mp4 as mentioned in error
	"video/mp4",
	"video/mov",
	"video/avi",
];

const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB

// Use direct API calls instead of mutations to avoid context issues

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

		// Comparison workflow temporarily disabled — always use regular analysis
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

		// Analysis complete, save result and navigate
		progress = 100;
		
		// Create blob URL for audio playback and add to result
		const audioUrl = URL.createObjectURL(file);
		analysisStore.set({
			...result,
			audioUrl,
			fileName: file.name
		});

		// Navigate to results page (comparison disabled)
		goto("/results");
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

  {#if false}
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
				
{#if comparisonMode}
					<div class="mt-6 p-4 bg-accent-gold/10 rounded-lg border border-accent-gold/20">
						<p class="text-sm text-warm-bronze text-center">
							<strong>Hybrid AST</strong> vs <strong>Ultra-Small AST</strong> · Comparative Performance Metrics
						</p>
					</div>
{/if}
		</section>
  {/if}

		<!-- Beginner Chat Section -->
		<section class="max-w-4xl mx-auto mb-8">
			<BeginnerChat />
		</section>

		<!-- OR Divider -->
		<div class="divider-container">
			<div class="divider-line"></div>
			<span class="divider-text">OR</span>
			<div class="divider-line"></div>
		</div>

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
					{#if progress > 0 && progress < 100}
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
					{/if}
				</div>
			</div>

			<!-- Error Display -->
			{#if uploadError}
				<div 
					class="error-card" 
					style="
						margin-top: 1.5rem; 
						max-width: 28rem; 
						margin-left: auto; 
						margin-right: auto;
						background: #fff5f5; 
						border: 2px solid #feb2b2; 
						border-radius: 12px; 
						padding: 1.25rem; 
						box-shadow: 0 4px 12px rgba(220, 53, 69, 0.15); 
						border-left: 4px solid #dc3545;
					"
				>
					<div style="display: flex; align-items: flex-start; gap: 1rem;">
						<div 
							class="error-icon"
							style="
								width: 2.5rem; 
								height: 2.5rem; 
								display: flex; 
								align-items: center; 
								justify-content: center; 
								background: #fee2e2; 
								border-radius: 50%; 
								color: #dc3545; 
								flex-shrink: 0; 
								border: 2px solid #fecaca;
							"
						>
							<AlertTriangle size={20} />
						</div>
						<div style="flex: 1;">
							<h4 
								class="error-title"
								style="
									font-family: var(--font-sans); 
									font-size: 1.1rem; 
									font-weight: 600; 
									color: #dc3545; 
									margin: 0 0 0.5rem 0;
								"
							>
								Upload Error
							</h4>
							<p 
								class="error-message"
								style="
									font-family: var(--font-serif); 
									color: #991b1b; 
									line-height: 1.5; 
									margin: 0 0 1rem 0; 
									font-size: 0.95rem;
								"
							>
								{uploadError}
							</p>
							<button 
								onclick={resetUpload} 
								class="error-retry"
								style="
									background: #dc3545; 
									color: white; 
									padding: 0.6rem 1.2rem; 
									border-radius: 8px; 
									font-size: 0.875rem; 
									font-weight: 600; 
									font-family: var(--font-sans); 
									transition: all 0.2s ease; 
									border: none; 
									cursor: pointer; 
									text-transform: uppercase; 
									letter-spacing: 0.05em;
								"
							>
								Try Again
							</button>
						</div>
					</div>
				</div>
			{/if}
		</section>

		<!-- Footer -->
		<footer class="mt-8 w-full flex justify-center">
		<div class="academic-footer text-center">
			<div class="footer-divider"></div>
			<p class="footer-text">
				<strong>CrescendAI</strong> · Advanced Performance Analysis · 2025
			</p>
		</div>
		</footer>
	</div>
</main>

<style>
.divider-container {
	display: flex;
	align-items: center;
	justify-content: center;
	max-width: 500px;
	margin: 3rem auto;
	gap: 1.5rem;
}

.divider-line {
	flex: 1;
	height: 1px;
	background: linear-gradient(to right, transparent, #c4a777, transparent);
	opacity: 0.3;
}

.divider-text {
	font-family: var(--font-sans, sans-serif);
	font-size: 0.875rem;
	font-weight: 600;
	letter-spacing: 0.2em;
	color: #c4a777;
	padding: 0 1rem;
	background: white;
	border: 2px solid #c4a777;
	border-radius: 2rem;
	min-width: 60px;
	text-align: center;
	padding: 0.5rem 1.5rem;
	opacity: 0.7;
}
</style>

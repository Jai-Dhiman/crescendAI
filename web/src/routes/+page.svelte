<script lang="ts">
import Button from '$lib/components/Button.svelte';
	import Card from '$lib/components/Card.svelte';
	import SketchyPianoKeys from '$lib/components/SketchyPianoKeys.svelte';
	import SketchyWaveform from '$lib/components/SketchyWaveform.svelte';
	import SketchySheetMusic from '$lib/components/SketchySheetMusic.svelte';
	import AnalysisVisualization from '$lib/components/AnalysisVisualization.svelte';
	import FeatureCard from '$lib/components/FeatureCard.svelte';

	const features = [
		{
			title: "Multi-Dimensional Analysis",
			description: "Our AI evaluates 19 distinct performance dimensions including timing precision, articulation clarity, tonal quality, and musical expression."
		},
		{
			title: "Score-Referenced Feedback",
			description: "Compare your performance directly against the written score with measure-by-measure analysis and specific improvement suggestions."
		},
		{
			title: "Conservatory-Grade Standards",
			description: "Built with input from piano faculty at leading conservatories, ensuring feedback meets the highest academic and artistic standards."
		},
		{
			title: "Progress Tracking",
			description: "Monitor your development over time with detailed performance analytics and personalized practice recommendations."
		}
	];

	const analysisCategories = [
		"Rhythm & Timing",
		"Articulation & Touch",
		"Dynamics & Expression",
		"Pedaling Technique",
		"Musical Interpretation"
	];

	function handleGetStarted() {
		window.location.href = '/auth';
	}

	function handleWatchDemo() {
		console.log("Demo functionality to be implemented");
	}
</script>

<svelte:head>
	<title>CrescendAI - Perfect Your Piano Performance with AI</title>
</svelte:head>

<div class="min-h-screen bg-slate-50">
	<!-- Hero Section -->
	<section class="relative overflow-hidden bg-gradient-to-b from-slate-50 to-white">
		<div class="absolute inset-0 opacity-5">
			<SketchyPianoKeys class="absolute top-0 left-0 w-full h-32 text-slate-400" />
			<SketchyPianoKeys class="absolute bottom-0 right-0 w-full h-32 text-slate-400 transform rotate-180" />
		</div>
		
		<div class="relative max-w-6xl mx-auto px-6 py-20">
			<div class="grid lg:grid-cols-2 gap-12 items-center">
				<div class="space-y-8">
					<div class="space-y-4">
						<div class="inline-block">
							<div class="px-4 py-2 bg-slate-100 border border-slate-300 rounded-full text-sm text-slate-700 transform -rotate-1">
								AI-Powered Piano Analysis
							</div>
						</div>
						
						<h1 class="text-4xl lg:text-5xl text-slate-900 leading-tight">
							Perfect Your Piano Performance with{" "}
							<span class="relative">
								<span class="relative z-10">19-Dimensional</span>
								<svg class="absolute -bottom-2 left-0 right-0 h-3" viewBox="0 0 200 12" fill="none">
									<path d="M2 8.5c65.3-1.2 130.7-2.1 196 0.5" stroke="#64748b" stroke-width="2" opacity="0.6" stroke-linecap="round" />
								</svg>
							</span>
							Feedback
						</h1>
						
						<p class="text-lg text-slate-600 leading-relaxed">
							Advanced AI analysis designed for conservatory and university piano students. 
							Get detailed, actionable feedback on every aspect of your performance—from technical precision to musical expression.
						</p>
					</div>
					
					<div class="flex flex-col sm:flex-row gap-4">
						<Button 
							onclick={handleGetStarted}
							class="bg-slate-800 hover:bg-slate-700 text-white px-8 py-3 rounded-lg border-2 border-slate-800 transform hover:-translate-y-1 transition-all"
						>
							Start Free Analysis
						</Button>
						<Button 
							variant="outline" 
							onclick={handleWatchDemo}
							class="border-2 border-slate-300 text-slate-700 hover:bg-slate-50 px-8 py-3 rounded-lg transform hover:-translate-y-1 transition-all"
						>
							Watch Demo
						</Button>
					</div>
					
					<div class="flex items-center space-x-6 text-sm text-slate-500">
						<div class="flex items-center space-x-2">
							<svg class="w-4 h-4" viewBox="0 0 16 16" fill="none">
								<path d="M2.1 8.2c1.2 1.1 2.4 2.1 3.7 3.1c2.8-2.9 5.7-5.8 8.5-8.7" 
									  stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
							</svg>
							<span>No signup required</span>
						</div>
						<div class="flex items-center space-x-2">
							<svg class="w-4 h-4" viewBox="0 0 16 16" fill="none">
								<path d="M2.1 8.2c1.2 1.1 2.4 2.1 3.7 3.1c2.8-2.9 5.7-5.8 8.5-8.7" 
									  stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
							</svg>
							<span>Results in minutes</span>
						</div>
					</div>
				</div>
				
				<div class="relative">
					<div class="relative z-10">
						<img
							src="https://images.unsplash.com/photo-1597066453679-331bb8175ddb?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwaWFubyUyMGtleXMlMjBzaGVldCUyMG11c2ljJTIwY29uc2VydmF0b3J5fGVufDF8fHx8MTc1NzM2ODk3OHww&ixlib=rb-4.1.0&q=80&w=1080"
							alt="Piano keys and sheet music"
							class="w-full h-96 object-cover rounded-lg border-4 border-white shadow-lg transform rotate-2"
						/>
					</div>
					<div class="absolute -top-4 -right-4 z-20">
						<AnalysisVisualization class="w-80 h-48 transform -rotate-3" />
					</div>
				</div>
			</div>
		</div>
	</section>

	<!-- Features Section -->
	<section class="py-20 bg-white">
		<div class="max-w-6xl mx-auto px-6">
			<div class="text-center mb-16">
				<div class="inline-block mb-4">
					<SketchySheetMusic class="w-20 h-16 text-slate-400 mx-auto" />
				</div>
				<h2 class="text-3xl lg:text-4xl text-slate-900 mb-4">
					Comprehensive Performance Analysis
				</h2>
				<p class="text-lg text-slate-600 max-w-2xl mx-auto">
					Our AI doesn't just listen—it understands music theory, performance practice, and pedagogical principles to provide meaningful feedback.
				</p>
			</div>
			
			<div class="grid md:grid-cols-2 gap-8 mb-16">
				{#each features as feature, index}
					<FeatureCard
						title={feature.title}
						description={feature.description}
						class="transform hover:scale-105 transition-transform"
					>
						{#snippet icon()}
							{#if index === 0}
								<SketchyWaveform class="w-8 h-8 text-slate-600" />
							{:else if index === 1}
								<SketchySheetMusic class="w-8 h-8 text-slate-600" />
							{:else if index === 2}
								<svg class="w-8 h-8 text-slate-600" viewBox="0 0 32 32" fill="none">
									<path d="M8.2 16.1c2.1-2.2 4.3-4.3 6.6-6.2c3.8 3.9 7.7 7.9 11.6 11.8c1.2-1.3 2.5-2.5 3.8-3.7" 
										  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
								</svg>
							{:else}
								<svg class="w-8 h-8 text-slate-600" viewBox="0 0 32 32" fill="none">
									<circle cx="16" cy="12" r="4.2" stroke="currentColor" stroke-width="2" fill="none" />
									<path d="M10.1 20.8c1.8-1.2 3.8-1.9 5.9-1.9s4.1 0.7 5.9 1.9" 
										  stroke="currentColor" stroke-width="2" stroke-linecap="round" />
								</svg>
							{/if}
						{/snippet}
					</FeatureCard>
				{/each}
			</div>
		</div>
	</section>

	<!-- Analysis Categories -->
	<section class="py-20 bg-slate-50">
		<div class="max-w-6xl mx-auto px-6">
			<div class="text-center mb-16">
				<h2 class="text-3xl lg:text-4xl text-slate-900 mb-4">
					19 Dimensions of Analysis
				</h2>
				<p class="text-lg text-slate-600 max-w-2xl mx-auto">
					Every recording is analyzed across multiple performance categories, providing detailed insights into both technical execution and musical interpretation.
				</p>
			</div>
			
			<div class="grid lg:grid-cols-5 gap-8">
				{#each analysisCategories as category, index}
					<Card class="p-6 text-center border-2 border-slate-200 bg-white transform hover:-translate-y-2 transition-all hover:shadow-lg">
						<div class="mb-4">
							<div class="w-12 h-12 bg-slate-100 rounded-full mx-auto flex items-center justify-center border-2 border-slate-200">
								<span class="text-slate-600 font-mono text-lg">{index + 1}</span>
							</div>
						</div>
						<h3 class="text-slate-800 mb-2">{category}</h3>
						<div class="text-sm text-slate-500">
							{Math.floor(3 + index * 0.8)} metrics
						</div>
					</Card>
				{/each}
			</div>
		</div>
	</section>

	<!-- How It Works -->
	<section class="py-20 bg-white">
		<div class="max-w-6xl mx-auto px-6">
			<div class="text-center mb-16">
				<h2 class="text-3xl lg:text-4xl text-slate-900 mb-4">
					Simple Process, Detailed Results
				</h2>
				<p class="text-lg text-slate-600">
					Upload your recording and receive comprehensive feedback in minutes
				</p>
			</div>
			
			<div class="grid md:grid-cols-3 gap-8">
				{#each [
					{
						step: "1",
						title: "Upload Recording",
						description: "Upload your piano performance—audio or video files accepted. No special equipment needed."
					},
					{
						step: "2", 
						title: "AI Analysis",
						description: "Our specialized AI analyzes your performance across 19 dimensions, comparing against conservatory standards."
					},
					{
						step: "3",
						title: "Detailed Feedback",
						description: "Receive actionable insights with specific practice recommendations and progress tracking."
					}
				] as item, index}
					<div class="text-center relative">
						{#if index < 2}
							<div class="hidden md:block absolute top-8 left-full w-full">
								<svg class="w-full h-4" viewBox="0 0 100 16" fill="none">
									<path d="M2 8c32.7-1.2 65.3-0.8 98 0" stroke="#cbd5e1" stroke-width="2" stroke-linecap="round" opacity="0.6" />
									<path d="M88 4l8 4-8 4" stroke="#cbd5e1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" opacity="0.6" />
								</svg>
							</div>
						{/if}
						<div class="w-16 h-16 bg-slate-800 text-white rounded-full mx-auto flex items-center justify-center mb-4 border-4 border-slate-200">
							<span class="text-xl font-mono">{item.step}</span>
						</div>
						<h3 class="text-slate-800 mb-2">{item.title}</h3>
						<p class="text-slate-600 text-sm">{item.description}</p>
					</div>
				{/each}
			</div>
		</div>
	</section>

	<!-- CTA Section -->
	<section class="py-20 bg-slate-900 relative overflow-hidden">
		<div class="absolute inset-0 opacity-10">
			<SketchyWaveform class="absolute top-0 left-0 w-full h-full text-white" />
		</div>
		
		<div class="relative max-w-4xl mx-auto px-6 text-center">
			<h2 class="text-3xl lg:text-4xl text-white mb-4">
				Ready to Elevate Your Piano Performance?
			</h2>
			<p class="text-lg text-slate-300 mb-8 max-w-2xl mx-auto">
				Join conservatory students worldwide who are using AI-powered analysis to accelerate their musical development.
			</p>
			
			<div class="flex flex-col sm:flex-row gap-4 justify-center">
				<Button 
					onclick={handleGetStarted}
					class="bg-white text-slate-900 hover:bg-slate-100 px-8 py-3 rounded-lg border-2 border-white transform hover:-translate-y-1 transition-all"
				>
					Start Free Analysis
				</Button>
				<Button 
					variant="outline" 
					class="border-2 border-slate-400 text-white hover:bg-slate-800 px-8 py-3 rounded-lg transform hover:-translate-y-1 transition-all"
				>
					Learn More
				</Button>
			</div>
			
			<div class="mt-8 text-sm text-slate-400">
				Free for students • No credit card required • Results in minutes
			</div>
		</div>
	</section>

	<!-- Footer -->
	<footer class="bg-slate-100 py-8 border-t border-slate-200">
		<div class="max-w-6xl mx-auto px-6">
			<div class="flex flex-col md:flex-row justify-between items-center">
				<div class="flex items-center space-x-2 mb-4 md:mb-0">
					<SketchyPianoKeys class="w-8 h-6 text-slate-600" />
					<span class="text-slate-700">CrescendAI</span>
				</div>
				<div class="flex space-x-6 text-sm text-slate-500">
					<a href="#" class="hover:text-slate-700 transition-colors">Privacy</a>
					<a href="#" class="hover:text-slate-700 transition-colors">Terms</a>
					<a href="#" class="hover:text-slate-700 transition-colors">Support</a>
					<a href="#" class="hover:text-slate-700 transition-colors">About</a>
				</div>
			</div>
			<div class="mt-4 pt-4 border-t border-slate-200 text-center text-sm text-slate-500">
				© 2025 CrescendAI. Designed for serious musicians.
			</div>
		</div>
	</footer>
</div>

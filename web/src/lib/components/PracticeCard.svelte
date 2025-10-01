<script lang="ts">
import type { ImmediatePriority, LongTermDevelopment } from '$lib/types/analysis';

let { 
	type = 'immediate',
	priority,
	development,
	index
} = $props<{
	type: 'immediate' | 'longterm';
	priority?: ImmediatePriority;
	development?: LongTermDevelopment;
	index?: number;
}>();

// Format skill area or musical aspect by taking only the main concept words
// and capitalizing properly. E.g., "articulation soft hard" -> "Articulation"
function formatHeading(text: string): string {
	if (!text) return '';
	
	// Split by spaces and filter out descriptor words
	const descriptorWords = ['soft', 'hard', 'fast', 'slow', 'high', 'low', 'loud', 'quiet'];
	const words = text.split(' ').filter(word => 
		!descriptorWords.includes(word.toLowerCase())
	);
	
	// Take the main words and capitalize each
	const formattedWords = words.map(word => 
		word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
	);
	
	return formattedWords.join(' ');
}
</script>

{#if type === 'immediate' && priority}
	<div class="practice-card immediate-card">
		<div class="card-number">{index !== undefined ? index + 1 : ''}</div>
		<div class="card-content">
			<h3 class="card-heading">{formatHeading(priority.skill_area)}</h3>
			<div class="card-section">
				<h4 class="section-label">Practice Exercise</h4>
				<p class="section-text">{priority.specific_exercise}</p>
			</div>
			<div class="card-section outcome-section">
				<h4 class="section-label">Expected Improvement</h4>
				<p class="section-text outcome-text">{priority.expected_outcome}</p>
			</div>
		</div>
	</div>
{:else if type === 'longterm' && development}
	<div class="practice-card longterm-card">
		<div class="card-content">
			<h3 class="card-heading">{formatHeading(development.musical_aspect)}</h3>
			<div class="card-section">
				<h4 class="section-label">Development Approach</h4>
				<p class="section-text">{development.development_approach}</p>
			</div>
			<div class="card-section repertoire-section">
				<h4 class="section-label">Suggested Repertoire</h4>
				<p class="section-text repertoire-text">{development.repertoire_suggestions}</p>
			</div>
		</div>
	</div>
{/if}

<style>
.practice-card {
	background: rgba(255, 255, 255, 0.85);
	backdrop-filter: blur(10px);
	border: 1px solid rgba(201, 168, 118, 0.2);
	border-radius: 12px;
	padding: 1.5rem;
	box-shadow: 0 4px 16px rgba(30, 39, 73, 0.06);
	transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
	position: relative;
	overflow: hidden;
}

.practice-card::before {
	content: '';
	position: absolute;
	top: 0;
	left: 0;
	width: 100%;
	height: 3px;
	background: linear-gradient(90deg, var(--accent-gold), var(--warm-bronze));
	transform: scaleX(0);
	transform-origin: left;
	transition: transform 0.3s ease;
}

.practice-card:hover {
	box-shadow: 0 8px 24px rgba(30, 39, 73, 0.1);
	border-color: rgba(201, 168, 118, 0.4);
	transform: translateY(-2px);
}

.practice-card:hover::before {
	transform: scaleX(1);
}

.immediate-card {
	border-left: 3px solid var(--accent-gold);
}

.longterm-card {
	border-left: 3px solid var(--muted-sage);
}

.card-number {
	position: absolute;
	top: 1rem;
	right: 1rem;
	width: 36px;
	height: 36px;
	display: flex;
	align-items: center;
	justify-content: center;
	background: linear-gradient(135deg, var(--rich-navy), var(--deep-forest));
	color: white;
	font-family: var(--font-display);
	font-size: 1.125rem;
	font-weight: 700;
	border-radius: 50%;
	box-shadow: 0 2px 8px rgba(30, 39, 73, 0.2);
}

.card-content {
	display: flex;
	flex-direction: column;
	gap: 1rem;
}

.immediate-card .card-content {
	padding-right: 2.5rem;
}

.card-heading {
	font-family: var(--font-display);
	font-size: 1.375rem;
	font-weight: 600;
	color: var(--rich-navy);
	margin: 0;
	line-height: 1.3;
}

.card-section {
	display: flex;
	flex-direction: column;
	gap: 0.5rem;
}

.section-label {
	font-family: var(--font-sans);
	font-size: 0.75rem;
	font-weight: 600;
	text-transform: uppercase;
	letter-spacing: 0.05em;
	color: var(--muted-sage);
	margin: 0;
}

.section-text {
	font-family: var(--font-serif);
	font-size: 0.975rem;
	line-height: 1.6;
	color: var(--charcoal-text);
	margin: 0;
}

.outcome-section {
	background: linear-gradient(135deg, rgba(201, 168, 118, 0.05), rgba(201, 168, 118, 0.08));
	padding: 0.875rem;
	border-radius: 8px;
	border-left: 2px solid var(--accent-gold);
}

.outcome-text {
	font-style: italic;
	color: var(--warm-bronze);
	font-weight: 500;
}

.repertoire-section {
	background: linear-gradient(135deg, rgba(122, 132, 113, 0.05), rgba(122, 132, 113, 0.08));
	padding: 0.875rem;
	border-radius: 8px;
	border-left: 2px solid var(--muted-sage);
}

.repertoire-text {
	color: var(--deep-forest);
	font-weight: 500;
}

@media (max-width: 768px) {
	.practice-card {
		padding: 1.25rem;
	}
	
	.immediate-card .card-content {
		padding-right: 0;
		padding-top: 2rem;
	}
	
	.card-number {
		top: 0.75rem;
		right: 0.75rem;
		width: 32px;
		height: 32px;
		font-size: 1rem;
	}
	
	.card-heading {
		font-size: 1.25rem;
	}
	
	.section-text {
		font-size: 0.925rem;
	}
	
	.outcome-section,
	.repertoire-section {
		padding: 0.75rem;
	}
}
</style>

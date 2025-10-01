<script lang="ts">
import { ChevronDown, ChevronUp } from 'lucide-svelte';
import type { TemporalFeedbackItem } from '$lib/types/analysis';

let { 
	feedback,
	isActive = false
} = $props<{
	feedback: TemporalFeedbackItem;
	isActive?: boolean;
}>();

let isExpanded = $state(false);

function toggleExpanded() {
	isExpanded = !isExpanded;
}

// Determine if this segment has particularly strong or weak moments
const hasHighPoints = $derived(
	feedback.insights.some((insight: any) => 
		insight.observation.toLowerCase().includes('excellent') ||
		insight.observation.toLowerCase().includes('strong') ||
		insight.observation.toLowerCase().includes('impressive')
	)
);

const hasLowPoints = $derived(
	feedback.insights.some((insight: any) => 
		insight.observation.toLowerCase().includes('needs') ||
		insight.observation.toLowerCase().includes('weak') ||
		insight.observation.toLowerCase().includes('inconsisten')
	)
);
</script>

<div 
	class="temporal-card" 
	class:active={isActive}
	class:has-high={hasHighPoints}
	class:has-low={hasLowPoints}
>
	<button 
		class="temporal-header" 
		onclick={toggleExpanded}
		aria-expanded={isExpanded}
	>
		<div class="header-content">
			<div class="timestamp-badge">{feedback.timestamp}</div>
			<div class="focus-preview">
				{#if !isExpanded}
					<span class="focus-text">{feedback.practice_focus}</span>
				{/if}
			</div>
		</div>
		<div class="expand-icon">
			{#if isExpanded}
				<ChevronUp size={20} />
			{:else}
				<ChevronDown size={20} />
			{/if}
		</div>
	</button>
	
	{#if isExpanded}
		<div class="temporal-content">
			<!-- Practice Focus -->
			<div class="practice-focus-section">
				<h4 class="focus-heading">Key Focus</h4>
				<p class="focus-description">{feedback.practice_focus}</p>
			</div>
			
			<!-- Insights -->
			<div class="insights-section">
				{#each feedback.insights as insight}
					<div class="insight-card">
						<div class="insight-header">
							<span class="insight-category">{insight.category}</span>
						</div>
						
						<div class="insight-body">
							<div class="observation-section">
								<h5 class="section-label">What I Noticed</h5>
								<p class="observation-text">{insight.observation}</p>
							</div>
							
							<div class="advice-section">
								<h5 class="section-label">How to Improve</h5>
								<p class="advice-text">{insight.actionable_advice}</p>
							</div>
						</div>
					</div>
				{/each}
			</div>
		</div>
	{/if}
</div>

<style>
.temporal-card {
	background: rgba(255, 255, 255, 0.85);
	backdrop-filter: blur(10px);
	border: 1px solid rgba(201, 168, 118, 0.2);
	border-radius: 12px;
	overflow: hidden;
	transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
	box-shadow: 0 4px 16px rgba(30, 39, 73, 0.06);
}

.temporal-card:hover {
	box-shadow: 0 6px 24px rgba(30, 39, 73, 0.1);
	border-color: rgba(201, 168, 118, 0.35);
}

.temporal-card.active {
	border-color: var(--accent-gold);
	box-shadow: 0 6px 24px rgba(201, 168, 118, 0.2);
}

.temporal-card.has-high {
	border-left: 3px solid rgba(201, 168, 118, 0.6);
}

.temporal-card.has-low {
	border-left: 3px solid rgba(122, 132, 113, 0.5);
}

.temporal-header {
	width: 100%;
	display: flex;
	align-items: center;
	justify-content: space-between;
	padding: 1rem 1.25rem;
	background: transparent;
	border: none;
	cursor: pointer;
	transition: background 0.2s ease;
	text-align: left;
}

.temporal-header:hover {
	background: rgba(201, 168, 118, 0.05);
}

.header-content {
	flex: 1;
	display: flex;
	align-items: center;
	gap: 1rem;
}

.timestamp-badge {
	font-family: var(--font-mono);
	font-size: 0.875rem;
	font-weight: 600;
	color: var(--rich-navy);
	background: rgba(201, 168, 118, 0.15);
	padding: 0.35rem 0.75rem;
	border-radius: 6px;
	white-space: nowrap;
}

.focus-preview {
	flex: 1;
	overflow: hidden;
}

.focus-text {
	font-family: var(--font-serif);
	font-size: 0.95rem;
	color: var(--subtle-gray);
	display: block;
	overflow: hidden;
	text-overflow: ellipsis;
	white-space: nowrap;
}

.expand-icon {
	color: var(--accent-gold);
	display: flex;
	align-items: center;
	transition: transform 0.2s ease;
}

.temporal-content {
	padding: 0 1.25rem 1.25rem;
	animation: slideDown 0.3s ease;
}

@keyframes slideDown {
	from {
		opacity: 0;
		transform: translateY(-10px);
	}
	to {
		opacity: 1;
		transform: translateY(0);
	}
}

.practice-focus-section {
	background: linear-gradient(135deg, rgba(201, 168, 118, 0.08), rgba(201, 168, 118, 0.12));
	border-left: 3px solid var(--accent-gold);
	padding: 1rem;
	border-radius: 8px;
	margin-bottom: 1.25rem;
}

.focus-heading {
	font-family: var(--font-sans);
	font-size: 0.75rem;
	font-weight: 600;
	text-transform: uppercase;
	letter-spacing: 0.05em;
	color: var(--warm-bronze);
	margin: 0 0 0.5rem 0;
}

.focus-description {
	font-family: var(--font-serif);
	font-size: 1rem;
	line-height: 1.6;
	color: var(--charcoal-text);
	margin: 0;
}

.insights-section {
	display: flex;
	flex-direction: column;
	gap: 1rem;
}

.insight-card {
	border: 1px solid rgba(201, 168, 118, 0.15);
	border-radius: 8px;
	overflow: hidden;
	transition: all 0.2s ease;
}

.insight-card:hover {
	border-color: rgba(201, 168, 118, 0.3);
	box-shadow: 0 2px 8px rgba(30, 39, 73, 0.05);
}

.insight-header {
	background: rgba(30, 39, 73, 0.04);
	padding: 0.5rem 0.875rem;
	border-bottom: 1px solid rgba(201, 168, 118, 0.1);
}

.insight-category {
	font-family: var(--font-sans);
	font-size: 0.75rem;
	font-weight: 600;
	text-transform: uppercase;
	letter-spacing: 0.05em;
	color: var(--rich-navy);
}

.insight-body {
	padding: 0.875rem;
	display: flex;
	flex-direction: column;
	gap: 0.875rem;
}

.observation-section,
.advice-section {
	display: flex;
	flex-direction: column;
	gap: 0.375rem;
}

.section-label {
	font-family: var(--font-sans);
	font-size: 0.8rem;
	font-weight: 600;
	color: var(--muted-sage);
	margin: 0;
}

.observation-text,
.advice-text {
	font-family: var(--font-serif);
	font-size: 0.95rem;
	line-height: 1.6;
	color: var(--charcoal-text);
	margin: 0;
}

.advice-text {
	font-style: italic;
	color: var(--subtle-gray);
}

@media (max-width: 768px) {
	.temporal-header {
		padding: 0.875rem 1rem;
	}
	
	.header-content {
		flex-direction: column;
		align-items: flex-start;
		gap: 0.5rem;
	}
	
	.timestamp-badge {
		font-size: 0.8rem;
	}
	
	.focus-text {
		font-size: 0.875rem;
		white-space: normal;
	}
	
	.temporal-content {
		padding: 0 1rem 1rem;
	}
	
	.practice-focus-section {
		padding: 0.875rem;
	}
	
	.insight-body {
		padding: 0.75rem;
	}
}
</style>

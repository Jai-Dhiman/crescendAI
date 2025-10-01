<script lang="ts">
	interface Props {
		icon: any;
		title: string;
		description: string;
		class?: string;
	}

	let { icon, title, description, class: className = '' }: Props = $props();
</script>

<div class="group relative {className}">
	<!-- Sketchy border effect -->
	<div
		class="absolute inset-0 bg-white border-2 border-slate-300 rounded-lg transform rotate-1 transition-transform group-hover:rotate-0"
	></div>
	<div
		class="relative bg-white border-2 border-slate-400 rounded-lg p-6 transform -rotate-0.5 transition-transform group-hover:rotate-1"
	>
		<div class="flex items-start space-x-4">
			<div class="flex-shrink-0 p-2 bg-slate-50 rounded-lg border border-slate-200">
				{@render icon()}
			</div>
			<div class="flex-1">
				<h3 class="text-slate-800 mb-2">{title}</h3>
				<p class="text-slate-600 text-sm leading-relaxed">{description}</p>
			</div>
		</div>
	</div>
</div>
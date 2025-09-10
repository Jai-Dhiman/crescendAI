<script lang="ts">
	import { QueryClientProvider } from '@tanstack/svelte-query';
	import { queryClient } from '$lib/query';
	import '../app.css';

	let { children } = $props();
</script>

<svelte:head>
	<title>CrescendAI - Piano Performance Analyzer</title>
	<meta name="description" content="AI-powered piano performance analysis with 19-dimensional feedback" />
	<link rel="icon" href="/crescendai.png" />
</svelte:head>

<QueryClientProvider client={queryClient}>
	{@render children?.()}
</QueryClientProvider>

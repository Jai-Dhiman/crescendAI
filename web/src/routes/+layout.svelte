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
	
	<!-- Google Fonts -->
	<link rel="preconnect" href="https://fonts.googleapis.com">
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
	<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Crimson+Text:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
</svelte:head>

<QueryClientProvider client={queryClient}>
	{@render children?.()}
</QueryClientProvider>

<script lang="ts">
	import { QueryClientProvider } from '@tanstack/svelte-query';
	import { queryClient } from '$lib/query';
	import '../app.css';
	import RunOnGPUToggle from '$lib/components/RunOnGPUToggle.svelte';
	import { runOnGPU, loadRunOnGPUFromStorage } from '$lib/stores/settings';
	import { onMount } from 'svelte';

	let { children } = $props();

	// Attach X-Run-GPU header to all client-side fetch calls
	onMount(() => {
		loadRunOnGPUFromStorage();
		const originalFetch = window.fetch.bind(window);
		window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
			// Snapshot current toggle value
			let run = false;
			const unsub = runOnGPU.subscribe((v) => (run = v));
			unsub();

			const headers = new Headers((init && init.headers) || (input instanceof Request ? input.headers : undefined));
			headers.set('X-Run-GPU', run ? 'true' : 'false');

			if (input instanceof Request) {
				const req = new Request(input, { headers });
				return originalFetch(req);
			} else {
				return originalFetch(input, { ...(init || {}), headers });
			}
		};
	});
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

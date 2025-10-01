<script lang="ts">
  import { runOnGPU, loadRunOnGPUFromStorage } from '$lib/stores/settings';
  import { onMount } from 'svelte';

  onMount(() => {
    loadRunOnGPUFromStorage();
  });

  let checked = false;
  const unsub = runOnGPU.subscribe((v) => (checked = v));
  $: runOnGPU.set(checked);
</script>

<label style="display:inline-flex;align-items:center;gap:10px;user-select:none;">
  <span class="sketchy-text text-sm text-charcoal-text">Run on GPU</span>
  <!-- hidden checkbox keeps state accessible and in sync with store -->
  <input type="checkbox" bind:checked class="sr-only" aria-label="Run on GPU" />

  <!-- gold switch (inline styles to avoid CSS conflicts) -->
  <span
    role="switch"
    aria-checked={checked}
    tabindex="0"
    on:click={() => (checked = !checked)}
    on:keydown={(e) => { if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); checked = !checked; } }}
    style="position:relative;display:inline-block;vertical-align:middle;width:44px;height:24px;border-radius:9999px;cursor:pointer;"
  >
    <!-- Track -->
    <span
      aria-hidden="true"
      style={`position:absolute;inset:0;border-radius:9999px;transition:background 0.2s ease,border-color 0.2s ease;${checked ? 'background: var(--accent-gold); border:2px solid var(--accent-gold);' : 'background: transparent; border:2px solid var(--accent-gold);'}`}
    ></span>
    <!-- Knob -->
    <span
      aria-hidden="true"
      style={`position:absolute;width:20px;height:20px;top:2px;left:${checked ? '22px' : '2px'};border-radius:50%;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,0.2);transition:left 0.2s ease;`}
    ></span>
  </span>
</label>

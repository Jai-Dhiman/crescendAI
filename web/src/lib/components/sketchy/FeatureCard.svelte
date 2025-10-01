<script lang="ts">
  import Card from '../ui/Card.svelte';
  import Typography from '../ui/Typography.svelte';
  
  type Props = {
    icon: any;
    title: string;
    description: string;
    class?: string;
  };

  let {
    icon,
    title,
    description,
    class: className = '',
    ...rest
  } = $props();
</script>

<div class="relative mb-4 {className}" {...rest}>
  <!-- Sketchy border effect - background card -->
<div 
    class="absolute inset-0 bg-white border-2 border-slate-300 rounded-lg transform rotate-1 z-0"
  ></div>
  
  <!-- Main content card -->
  <Card 
    variant="elevated" 
    padding="lg"
    class="relative bg-white border-2 border-slate-400 transform -rotate-0.5 z-10"
  >
    {#snippet children()}
      <div class="flex items-start gap-4">
        <div class="flex-shrink-0">
          <div class="p-2 bg-slate-50 rounded-lg border border-slate-200">
            {@render icon()}
          </div>
        </div>
        <div class="flex-1">
          <Typography 
            variant="h4" 
            class="mb-2 font-semibold text-slate-800"
          >
            {#snippet children()}{title}{/snippet}
          </Typography>
          <Typography 
            variant="body" 
            class="text-slate-600 text-sm leading-5"
          >
            {#snippet children()}{description}{/snippet}
          </Typography>
        </div>
      </div>
    {/snippet}
  </Card>
</div>

<style>
  .transform {
    transition: transform 0.2s ease;
  }
  
  .transform:hover {
    transform: rotate(-0.25deg) translateY(-2px);
  }
</style>

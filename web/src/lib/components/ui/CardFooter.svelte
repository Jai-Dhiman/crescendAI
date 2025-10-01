<script lang="ts">
  interface Props {
    children: any;
    class?: string;
  }

  let {
    children,
    class: className = '',
    ...rest
  }: Props = $props();
</script>

<div class="mt-4 flex items-center justify-end gap-2 {className}" {...rest}>
  {@render children()}
</div>

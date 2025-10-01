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

<div class="mb-4 {className}" {...rest}>
  {@render children()}
</div>

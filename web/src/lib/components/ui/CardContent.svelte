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

<div class="flex-1 {className}" {...rest}>
  {@render children()}
</div>

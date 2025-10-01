<script lang="ts">
  interface Props {
    variant?: 'h1' | 'h2' | 'h3' | 'h4' | 'body' | 'bodyLarge' | 'bodySmall' | 'label' | 'caption';
    color?: string;
    children: any;
    class?: string;
    element?: 'h1' | 'h2' | 'h3' | 'h4' | 'p' | 'span' | 'div' | 'label';
  }

  let {
    variant = 'body',
    color,
    children,
    class: className = '',
    element,
    ...rest
  }: Props = $props();

  const variantClasses = {
    h1: 'text-4xl font-bold leading-tight',
    h2: 'text-3xl font-semibold leading-tight',
    h3: 'text-2xl font-semibold leading-tight',
    h4: 'text-xl font-medium leading-snug',
    body: 'text-base leading-relaxed',
    bodyLarge: 'text-lg leading-relaxed',
    bodySmall: 'text-sm leading-relaxed',
    label: 'text-sm font-medium',
    caption: 'text-xs text-gray-600'
  };

  const defaultElements = {
    h1: 'h1',
    h2: 'h2',
    h3: 'h3',
    h4: 'h4',
    body: 'p',
    bodyLarge: 'p',
    bodySmall: 'p',
    label: 'label',
    caption: 'span'
  };

  const elementTag = element || defaultElements[variant];
  const colorStyle = color ? `color: ${color};` : '';
</script>

<svelte:element
  this={elementTag}
  class="{variantClasses[variant]} {className}"
  style={colorStyle}
  {...rest}
>
  {@render children()}
</svelte:element>

<script lang="ts">
  interface Props {
    variant?: 'default' | 'outlined' | 'elevated';
    padding?: 'sm' | 'md' | 'lg' | 'xl';
    children: any;
    class?: string;
  }

  let {
    variant = 'default',
    padding = 'md',
    children,
    class: className = '',
    ...rest
  }: Props = $props();

  const variantClasses = {
    default: 'bg-white border-0',
    outlined: 'bg-white border border-gray-200',
    elevated: 'bg-white shadow-md border-0'
  };

  const paddingClasses = {
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-6',
    xl: 'p-8'
  };
</script>

<div
  class="rounded-lg transition-all duration-200 hover:shadow-sm {variantClasses[variant]} {paddingClasses[padding]} {className}"
  {...rest}
>
  {@render children()}
</div>

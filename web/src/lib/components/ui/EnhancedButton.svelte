<script lang="ts">
  interface Props {
    variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'destructive';
    size?: 'sm' | 'md' | 'lg';
    disabled?: boolean;
    loading?: boolean;
    leftIcon?: any;
    rightIcon?: any;
    children: any;
    class?: string;
    onclick?: (event: MouseEvent) => void;
    type?: 'button' | 'submit' | 'reset';
  }

  let {
    variant = 'primary',
    size = 'md',
    disabled = false,
    loading = false,
    leftIcon,
    rightIcon,
    children,
    class: className = '',
    onclick,
    type = 'button',
    ...rest
  }: Props = $props();

  const variantClasses = {
    primary: 'bg-slate-800 text-white border-slate-800 hover:bg-slate-900',
    secondary: 'bg-slate-100 text-slate-800 border-slate-100 hover:bg-slate-200',
    outline: 'bg-transparent text-slate-800 border-slate-300 hover:bg-slate-50',
    ghost: 'bg-transparent text-slate-800 border-transparent hover:bg-slate-100',
    destructive: 'bg-red-600 text-white border-red-600 hover:bg-red-700'
  };

  const sizeClasses = {
    sm: 'px-3 py-2 text-sm rounded min-h-8',
    md: 'px-4 py-3 text-base rounded-md min-h-10', 
    lg: 'px-6 py-3 text-lg rounded-lg min-h-12'
  };

  const isDisabled = disabled || loading;
</script>

<button
  {type}
  disabled={isDisabled}
  onclick={onclick}
  class="inline-flex items-center justify-center gap-2 font-medium cursor-pointer transition-all duration-200 outline-none relative border hover:opacity-90 hover:-translate-y-1 active:translate-y-0 disabled:opacity-60 disabled:cursor-not-allowed disabled:transform-none focus-visible:outline-2 focus-visible:outline-blue-500 focus-visible:outline-offset-2 {variantClasses[variant]} {sizeClasses[size]} {className}"
  {...rest}
>
  {#if loading}
    <div class="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
  {:else if leftIcon}
    <span class="icon-left">
      {@render leftIcon()}
    </span>
  {/if}
  
  {#if !loading}
    {@render children()}
  {/if}
  
  {#if !loading && rightIcon}
    <span class="icon-right">
      {@render rightIcon()}
    </span>
  {/if}
</button>

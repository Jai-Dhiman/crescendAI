<script lang="ts">
	interface Props {
		variant?: 'elevated' | 'outlined' | 'filled';
		padding?: 'none' | 'sm' | 'md' | 'lg';
		children: any;
		class?: string;
	}

	let {
		variant = 'elevated',
		padding = 'md',
		children,
		class: className = '',
		...rest
	}: Props = $props();

	const variantClasses = {
		elevated: 'bg-white border border-slate-200 shadow-md',
		outlined: 'bg-white border border-slate-200',
		filled: 'bg-slate-100 border border-transparent'
	};

	const paddingClasses = {
		none: '',
		sm: 'p-3',
		md: 'p-4',
		lg: 'p-6'
	};
</script>

<div
	class="card rounded-xl transition-all duration-200 hover:-translate-y-1 {variantClasses[variant]} {paddingClasses[padding]} {className}"
	{...rest}
>
	{@render children()}
</div>
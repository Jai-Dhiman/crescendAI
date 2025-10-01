<script lang="ts">
	interface Props {
		class?: string;
	}

	let { class: className = '' }: Props = $props();
</script>

<svg
	viewBox="0 0 160 120"
	class={className}
	fill="none"
	xmlns="http://www.w3.org/2000/svg"
>
	<!-- Staff lines with slight imperfections -->
	<path
		d="M10.2 20.1c49.8-0.2 99.6-0.1 149.3 0.1"
		stroke="currentColor"
		stroke-width="1.2"
		opacity="0.6"
	/>
	<path
		d="M10.1 30.2c49.9 0.1 99.7 0 149.4-0.2"
		stroke="currentColor"
		stroke-width="1.2"
		opacity="0.6"
	/>
	<path
		d="M10.3 40.1c49.7 0.1 99.5 0.2 149.2-0.1"
		stroke="currentColor"
		stroke-width="1.2"
		opacity="0.6"
	/>
	<path
		d="M10.2 50.2c49.8-0.1 99.6 0.1 149.3 0"
		stroke="currentColor"
		stroke-width="1.2"
		opacity="0.6"
	/>
	<path
		d="M10.1 60.1c49.9 0.2 99.7-0.1 149.4 0.1"
		stroke="currentColor"
		stroke-width="1.2"
		opacity="0.6"
	/>

	<!-- Second staff -->
	<path
		d="M10.3 80.2c49.7-0.1 99.5 0.1 149.2 0"
		stroke="currentColor"
		stroke-width="1.2"
		opacity="0.6"
	/>
	<path
		d="M10.2 90.1c49.8 0.1 99.6-0.2 149.3 0.1"
		stroke="currentColor"
		stroke-width="1.2"
		opacity="0.6"
	/>
	<path
		d="M10.1 100.2c49.9-0.1 99.7 0.1 149.4-0.1"
		stroke="currentColor"
		stroke-width="1.2"
		opacity="0.6"
	/>
	<path
		d="M10.3 110.1c49.7 0.2 99.5-0.1 149.2 0"
		stroke="currentColor"
		stroke-width="1.2"
		opacity="0.6"
	/>

	<!-- Treble clef - sketchy -->
	<path
		d="M25.1 25.8c-2.1 0.3-3.8 2.1-3.9 4.2c-0.1 2.1 1.5 4.1 3.6 4.3c2.1 0.2 4.2-1.4 4.4-3.5c0.2-2.1-1.3-4.2-3.4-4.5c-2.1-0.3-4.3 1.2-4.6 3.3c-0.3 2.1 1.1 4.3 3.2 4.7c2.1 0.4 4.4-1.1 4.8-3.2"
		stroke="currentColor"
		stroke-width="1.5"
		stroke-linecap="round"
		opacity="0.8"
	/>
	<path
		d="M26.2 28.1c0.1-6.8 0.3-13.6 0.8-20.3c0.5-6.7 1.3-13.4 2.4-20"
		stroke="currentColor"
		stroke-width="1.5"
		stroke-linecap="round"
		opacity="0.8"
	/>
	<path
		d="M29.1 45.2c0.2-5.7 0.1-11.4-0.2-17.1"
		stroke="currentColor"
		stroke-width="1.5"
		stroke-linecap="round"
		opacity="0.8"
	/>

	<!-- Notes - imperfect circles -->
	<ellipse
		cx="45.2"
		cy="40.8"
		rx="2.8"
		ry="2.1"
		fill="currentColor"
		opacity="0.9"
		transform="rotate(-12 45.2 40.8)"
	/>
	<ellipse
		cx="55.8"
		cy="35.2"
		rx="2.9"
		ry="2.0"
		fill="currentColor"
		opacity="0.9"
		transform="rotate(-15 55.8 35.2)"
	/>
	<ellipse
		cx="66.1"
		cy="30.9"
		rx="2.7"
		ry="2.2"
		fill="currentColor"
		opacity="0.9"
		transform="rotate(-10 66.1 30.9)"
	/>
	<ellipse
		cx="76.4"
		cy="25.1"
		rx="2.8"
		ry="2.1"
		fill="currentColor"
		opacity="0.9"
		transform="rotate(-13 76.4 25.1)"
	/>

	<!-- Stems with slight wobble -->
	<path
		d="M47.8 40.2c0.1-7.8 0.3-15.6 0.6-23.4"
		stroke="currentColor"
		stroke-width="1.3"
		opacity="0.9"
	/>
	<path
		d="M58.4 35.1c0.2-7.9 0.4-15.8 0.7-23.7"
		stroke="currentColor"
		stroke-width="1.3"
		opacity="0.9"
	/>
	<path
		d="M68.7 30.8c0.1-8.1 0.3-16.2 0.5-24.3"
		stroke="currentColor"
		stroke-width="1.3"
		opacity="0.9"
	/>
	<path
		d="M79.1 25.0c0.2-8.0 0.4-16.0 0.7-24.0"
		stroke="currentColor"
		stroke-width="1.3"
		opacity="0.9"
	/>
</svg>
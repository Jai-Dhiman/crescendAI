import { useLayoutEffect, useRef } from "react";

export function ClipSvg({ svg }: { svg: string }) {
	const ref = useRef<HTMLDivElement>(null);
	useLayoutEffect(() => {
		if (!ref.current) return;
		ref.current.textContent = "";
		// biome-ignore lint/security/noDomManipulation: controlled SVG from Verovio WASM, not user input
		ref.current.insertAdjacentHTML("afterbegin", svg);
		const svgEl = ref.current.querySelector("svg");
		if (svgEl) {
			svgEl.setAttribute("width", "100%");
			svgEl.removeAttribute("height");
			(svgEl as SVGElement).style.display = "block";
		}
	}, [svg]);
	return <div ref={ref} className="[&>svg]:w-full [&>svg]:block" />;
}

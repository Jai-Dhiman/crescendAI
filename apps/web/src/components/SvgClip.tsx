import { useLayoutEffect, useRef } from "react";

interface SvgClipProps {
	svgMarkup: string;
	startMeasureId: string | null;
	endMeasureId: string | null;
}

// Crop the SVG viewBox to show only the system rows containing the start/end measures.
// Uses getBoundingClientRect for screen-space positions and converts back to viewBox
// coordinates, so it works regardless of the SVG's internal coordinate units.
function cropSvgToMeasureRange(
	svgEl: SVGSVGElement,
	startMeasureId: string,
	endMeasureId: string,
): void {
	const startMeasure = svgEl.querySelector(`[id="${startMeasureId}"]`);
	if (!startMeasure) return;

	const startSystem = startMeasure.closest(".system") as SVGGElement | null;
	if (!startSystem) return;

	const endMeasure = svgEl.querySelector(`[id="${endMeasureId}"]`);
	const endSystem =
		(endMeasure?.closest(".system") as SVGGElement | null) ?? startSystem;

	const svgRect = svgEl.getBoundingClientRect();
	if (svgRect.width === 0 || svgRect.height === 0) return;

	const startRect = startSystem.getBoundingClientRect();
	const endRect = endSystem.getBoundingClientRect();

	const vb = svgEl.viewBox.baseVal;
	if (!vb) return;

	const scaleY = vb.height / svgRect.height;

	// 12px of visual padding converted to viewBox units
	const padVb = 12 * scaleY;

	const minY = Math.max(
		vb.y,
		(startRect.top - svgRect.top) * scaleY + vb.y - padVb,
	);
	const maxY = Math.min(
		vb.y + vb.height,
		(endRect.bottom - svgRect.top) * scaleY + vb.y + padVb,
	);

	if (maxY <= minY) return;

	svgEl.setAttribute("viewBox", `${vb.x} ${minY} ${vb.width} ${maxY - minY}`);
}

export function SvgClip({
	svgMarkup,
	startMeasureId,
	endMeasureId,
}: SvgClipProps) {
	const ref = useRef<HTMLDivElement>(null);

	useLayoutEffect(() => {
		if (!ref.current) return;
		ref.current.textContent = "";
		// biome-ignore lint/security/noDomManipulation: controlled SVG from Verovio WASM, not user input
		ref.current.insertAdjacentHTML("afterbegin", svgMarkup);
		const svgEl = ref.current.querySelector("svg") as SVGSVGElement | null;
		if (!svgEl) return;

		// Verovio may omit viewBox when adjustPageHeight is enabled. Synthesize it
		// from width/height before removing height, since height becomes 0 after removal.
		if (!svgEl.getAttribute("viewBox")) {
			const w = parseFloat(svgEl.getAttribute("width") ?? "0");
			const h = parseFloat(svgEl.getAttribute("height") ?? "0");
			if (w > 0 && h > 0) {
				svgEl.setAttribute("viewBox", `0 0 ${w} ${h}`);
			}
		}

		svgEl.setAttribute("width", "100%");
		// Remove explicit height so the browser sizes from the viewBox aspect ratio.
		// Must happen after the viewBox is set (above) or after any crop modifies it.
		svgEl.removeAttribute("height");
		svgEl.style.display = "block";

		if (startMeasureId && endMeasureId) {
			cropSvgToMeasureRange(svgEl, startMeasureId, endMeasureId);
		}
	}, [svgMarkup, startMeasureId, endMeasureId]);

	return <div ref={ref} className="[&>svg]:w-full [&>svg]:block" />;
}

import { useLayoutEffect, useRef } from "react";

interface SvgClipBBoxProps {
	svgMarkup: string;
	startMeasureId: string | null;
	endMeasureId: string | null;
}

// Convert an element's local getBBox() coordinates to SVG viewport coordinates
// by applying getCTM() (the cumulative transform matrix to the SVG viewport).
// getBBox() alone returns LOCAL coordinates and ignores ancestor transforms —
// which breaks for Verovio systems that use translate() to position each row.
function toViewportBBox(
	el: SVGGraphicsElement,
	svgEl: SVGSVGElement,
): { y: number; bottom: number } | null {
	const bbox = el.getBBox();
	const ctm = el.getCTM();
	if (!ctm) return null;

	const p1 = svgEl.createSVGPoint();
	p1.x = bbox.x;
	p1.y = bbox.y;

	const p2 = svgEl.createSVGPoint();
	p2.x = bbox.x + bbox.width;
	p2.y = bbox.y + bbox.height;

	const tp1 = p1.matrixTransform(ctm);
	const tp2 = p2.matrixTransform(ctm);

	return { y: Math.min(tp1.y, tp2.y), bottom: Math.max(tp1.y, tp2.y) };
}

function cropSvgToMeasureRangeBBox(
	svgEl: SVGSVGElement,
	startMeasureId: string,
	endMeasureId: string,
): void {
	const startMeasure = svgEl.querySelector(
		`[id="${startMeasureId}"]`,
	) as SVGGraphicsElement | null;
	if (!startMeasure) return;

	const startSystem = startMeasure.closest(
		".system",
	) as SVGGraphicsElement | null;
	if (!startSystem) return;

	const endMeasure = svgEl.querySelector(
		`[id="${endMeasureId}"]`,
	) as SVGGraphicsElement | null;
	const endSystem =
		(endMeasure?.closest(".system") as SVGGraphicsElement | null) ??
		startSystem;

	const vb = svgEl.viewBox.baseVal;
	if (!vb || vb.width === 0 || vb.height === 0) return;

	const startBox = toViewportBBox(startSystem, svgEl);
	const endBox = toViewportBBox(endSystem, svgEl);
	if (!startBox || !endBox) return;

	// 120 SVG units ≈ small visual margin (Verovio staff height at scale 40 ≈ 900 units)
	const pad = 120;
	const minY = Math.max(vb.y, startBox.y - pad);
	const maxY = Math.min(vb.y + vb.height, endBox.bottom + pad);

	if (maxY <= minY) return;

	svgEl.setAttribute("viewBox", `${vb.x} ${minY} ${vb.width} ${maxY - minY}`);
}

export function SvgClipBBox({
	svgMarkup,
	startMeasureId,
	endMeasureId,
}: SvgClipBBoxProps) {
	const ref = useRef<HTMLDivElement>(null);

	useLayoutEffect(() => {
		if (!ref.current) return;
		ref.current.textContent = "";
		// biome-ignore lint/security/noDomManipulation: controlled SVG from Verovio WASM, not user input
		ref.current.insertAdjacentHTML("afterbegin", svgMarkup);
		const svgEl = ref.current.querySelector("svg") as SVGSVGElement | null;
		if (!svgEl) return;

		if (!svgEl.getAttribute("viewBox")) {
			const w = parseFloat(svgEl.getAttribute("width") ?? "0");
			const h = parseFloat(svgEl.getAttribute("height") ?? "0");
			if (w > 0 && h > 0) {
				svgEl.setAttribute("viewBox", `0 0 ${w} ${h}`);
			}
		}

		// Crop while SVG still has its original width/height so getCTM() has valid context,
		// then switch to responsive sizing.
		if (startMeasureId && endMeasureId) {
			cropSvgToMeasureRangeBBox(svgEl, startMeasureId, endMeasureId);
		}

		svgEl.setAttribute("width", "100%");
		svgEl.removeAttribute("height");
		svgEl.style.display = "block";
	}, [svgMarkup, startMeasureId, endMeasureId]);

	return <div ref={ref} className="[&>svg]:w-full [&>svg]:block" />;
}

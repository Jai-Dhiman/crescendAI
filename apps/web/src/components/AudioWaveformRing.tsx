import { useEffect, useRef } from "react";
import { useSyncRef } from "../hooks/useFoundation";

interface AudioWaveformRingProps {
	analyserNode: AnalyserNode | null;
	isPlaying: boolean;
	active: boolean;
}

// Sage green
const SAGE_R = 122;
const SAGE_G = 154;
const SAGE_B = 130;

// Number of points around the circle
const NUM_POINTS = 128;
// Breathing animation period (ms)
const BREATH_PERIOD = 4000;
// Lerp coefficient (frame-rate-independent base)
const LERP_BASE = 0.15;
// Rotation speed (deg/frame at 60fps)
const ROTATION_SPEED = 0.5;
// Idle frame throttle (ms) for ~15fps
const IDLE_FRAME_MIN_MS = 66;
// Max displacement as fraction of ring radius
const MAX_DISPLACEMENT = 0.35;
// Base ring stroke width
const STROKE_WIDTH = 1.5;

/**
 * Build a log-scale mapping table: for each of NUM_POINTS positions around
 * the circle, which frequency bin index to read. Concentrates resolution in
 * low frequencies where piano fundamentals live.
 */
function buildLogBinMap(binCount: number): number[] {
	const map: number[] = [];
	for (let i = 0; i < NUM_POINTS; i++) {
		const t = i / NUM_POINTS;
		// Quadratic mapping concentrates low frequencies
		const binIndex = Math.floor(t ** 2 * (binCount - 1));
		map.push(Math.min(binIndex, binCount - 1));
	}
	return map;
}

export function AudioWaveformRing({
	analyserNode,
	isPlaying,
	active,
}: AudioWaveformRingProps) {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const rafRef = useRef<number>(0);
	const lastFrameRef = useRef(0);
	const rotationRef = useRef(0);
	const displacementsRef = useRef<Float32Array | null>(null);
	const dataArrayRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
	const logBinMapRef = useRef<number[] | null>(null);
	const crossfadeRef = useRef(0); // 0 = full breathing, 1 = full frequency
	const isPlayingRef = useSyncRef(isPlaying);
	const sizeRef = useRef({ w: 0, h: 0 });

	// Persistent RAF animation loop -- re-initializes when analyserNode or active changes.
	// biome-ignore lint/correctness/useExhaustiveDependencies: only re-run when analyserNode/active change
	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;

		// Initialize displacements
		if (!displacementsRef.current) {
			displacementsRef.current = new Float32Array(NUM_POINTS);
		}

		// Set up analyser data structures
		if (analyserNode && !logBinMapRef.current) {
			const binCount = analyserNode.frequencyBinCount;
			logBinMapRef.current = buildLogBinMap(binCount);
			dataArrayRef.current = new Uint8Array(
				binCount,
			) as Uint8Array<ArrayBuffer>;
		}

		// ResizeObserver for canvas sizing
		const observer = new ResizeObserver((entries) => {
			for (const entry of entries) {
				const { width, height } = entry.contentRect;
				const dpr = window.devicePixelRatio || 1;
				canvas.width = width * dpr;
				canvas.height = height * dpr;
				sizeRef.current = { w: width, h: height };
			}
		});
		observer.observe(canvas);

		const ctx = canvas.getContext("2d")!;
		if (!ctx) return;

		function draw(timestamp: number) {
			const dt = lastFrameRef.current
				? timestamp - lastFrameRef.current
				: 16.67;
			lastFrameRef.current = timestamp;

			// Throttle in idle mode
			if (!active && dt < IDLE_FRAME_MIN_MS) {
				rafRef.current = requestAnimationFrame(draw);
				return;
			}

			// Self-throttle on slow devices
			if (dt > 33 && active) {
				rafRef.current = requestAnimationFrame(draw);
				lastFrameRef.current = timestamp;
				return;
			}

			const { w, h } = sizeRef.current;
			if (w === 0 || h === 0) {
				rafRef.current = requestAnimationFrame(draw);
				return;
			}

			const dpr = window.devicePixelRatio || 1;
			ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
			ctx.clearRect(0, 0, w, h);

			const cx = w / 2;
			const cy = h / 2;
			const radius = Math.min(w, h) * 0.38;
			const displacements = displacementsRef.current!;

			// Update crossfade (0 = breathing, 1 = frequency)
			const crossfadeTarget = isPlayingRef.current && analyserNode ? 1 : 0;
			const crossfadeAlpha = 1 - (1 - 0.005) ** dt;
			crossfadeRef.current +=
				(crossfadeTarget - crossfadeRef.current) * crossfadeAlpha;

			// Frame-rate-independent lerp alpha
			const lerpAlpha = 1 - (1 - LERP_BASE) ** (dt / 16.67);

			// Calculate target displacements
			let totalEnergy = 0;

			// Read frequency data once per frame
			if (analyserNode && dataArrayRef.current) {
				analyserNode.getByteFrequencyData(dataArrayRef.current);
			}

			for (let i = 0; i < NUM_POINTS; i++) {
				let freqTarget = 0;

				// Frequency-reactive displacement
				if (dataArrayRef.current && logBinMapRef.current) {
					const binIdx = logBinMapRef.current[i];
					const value = dataArrayRef.current[binIdx] / 255;
					freqTarget = value * MAX_DISPLACEMENT * radius;
					totalEnergy += value;
				}

				// Breathing displacement (sinusoidal)
				const breathPhase = (timestamp % BREATH_PERIOD) / BREATH_PERIOD;
				const breathTarget =
					Math.sin(breathPhase * Math.PI * 2) * radius * 0.03;

				// Blend breathing and frequency based on crossfade
				const cf = crossfadeRef.current;
				const target = freqTarget * cf + breathTarget * (1 - cf);

				// Lerp toward target
				displacements[i] += (target - displacements[i]) * lerpAlpha;
			}

			// Compute opacity from energy (0.6 to 1.0)
			const avgEnergy = analyserNode ? totalEnergy / NUM_POINTS : 0;
			const opacity = 0.6 + avgEnergy * 0.4;

			// Update rotation (frame-rate-normalized)
			rotationRef.current += ROTATION_SPEED * (dt / 16.67);
			const rotRad = (rotationRef.current * Math.PI) / 180;

			// Draw the ring
			ctx.beginPath();
			for (let i = 0; i <= NUM_POINTS; i++) {
				const idx = i % NUM_POINTS;
				const angle = (idx / NUM_POINTS) * Math.PI * 2 - Math.PI / 2 + rotRad;
				const r = radius + displacements[idx];
				const x = cx + Math.cos(angle) * r;
				const y = cy + Math.sin(angle) * r;

				if (i === 0) {
					ctx.moveTo(x, y);
				} else {
					// Smooth curve through points using quadratic bezier
					const prevIdx = (i - 1) % NUM_POINTS;
					const prevAngle =
						(prevIdx / NUM_POINTS) * Math.PI * 2 - Math.PI / 2 + rotRad;
					const prevR = radius + displacements[prevIdx];
					const prevX = cx + Math.cos(prevAngle) * prevR;
					const prevY = cy + Math.sin(prevAngle) * prevR;
					const cpX = (prevX + x) / 2;
					const cpY = (prevY + y) / 2;
					ctx.quadraticCurveTo(prevX, prevY, cpX, cpY);
				}
			}
			ctx.closePath();

			ctx.strokeStyle = `rgba(${SAGE_R}, ${SAGE_G}, ${SAGE_B}, ${opacity})`;
			ctx.lineWidth = STROKE_WIDTH;
			ctx.stroke();

			rafRef.current = requestAnimationFrame(draw);
		}

		rafRef.current = requestAnimationFrame(draw);

		return () => {
			cancelAnimationFrame(rafRef.current);
			observer.disconnect();
		};
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [analyserNode, active]);

	return (
		<canvas ref={canvasRef} className="w-full h-full" aria-hidden="true" />
	);
}

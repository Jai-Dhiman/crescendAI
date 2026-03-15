import { useEffect, useRef } from "react";
import { createLogger } from "../lib/logger";

const log = createLogger("Ripples");

interface ResonanceRipplesProps {
	energy: number;
	isPlaying: boolean;
	active: boolean;
}

// Sage green RGB
const SAGE_R = 122;
const SAGE_G = 154;
const SAGE_B = 130;

// Idle ripple interval (ms)
const IDLE_RIPPLE_INTERVAL = 3000;
// Min interval between active ripples at peak energy (ms)
const MIN_RIPPLE_INTERVAL = 400;
// Max interval between active ripples at low energy (ms)
const MAX_RIPPLE_INTERVAL = 1500;
// Ripple expansion speed (px/sec)
const EXPANSION_SPEED = 80;
// Max ripple radius as fraction of canvas min dimension
const MAX_RADIUS_FRACTION = 0.45;
// Wobble noise amplitude as fraction of radius
const WOBBLE_AMOUNT = 0.04;
// Idle throttle: skip frames when delta < this (ms) for ~15fps
const IDLE_FRAME_MIN_MS = 66;

interface Ripple {
	birthTime: number;
	maxRadius: number;
	wobblePhase1: number;
	wobblePhase2: number;
	baseOpacity: number;
}

export function ResonanceRipples({
	energy,
	isPlaying,
	active,
}: ResonanceRipplesProps) {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const rafRef = useRef<number>(0);
	const ripplesRef = useRef<Ripple[]>([]);
	const lastRippleTimeRef = useRef(0);
	const lastFrameTimeRef = useRef(0);
	const frameCountRef = useRef(0);

	// Store props in refs so the rAF loop reads fresh values without
	// restarting the effect on every energy change (~60 times/sec).
	const energyRef = useRef(energy);
	const isPlayingRef = useRef(isPlaying);
	energyRef.current = energy;
	isPlayingRef.current = isPlaying;

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;

		const ctx = canvas.getContext("2d");
		if (!ctx) return;

		function draw(timestamp: number) {
			rafRef.current = requestAnimationFrame(draw);

			const dt = lastFrameTimeRef.current
				? timestamp - lastFrameTimeRef.current
				: 16;

			const currentIsPlaying = isPlayingRef.current;
			const currentEnergy = energyRef.current;

			// Throttle to ~15fps when idle
			if (!currentIsPlaying && dt < IDLE_FRAME_MIN_MS) return;

			lastFrameTimeRef.current = timestamp;

			// Resize canvas for DPR
			const rect = canvas!.getBoundingClientRect();
			const dpr = window.devicePixelRatio || 1;
			const w = rect.width * dpr;
			const h = rect.height * dpr;

			if (canvas!.width !== w || canvas!.height !== h) {
				canvas!.width = w;
				canvas!.height = h;
			}

			ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
			ctx!.clearRect(0, 0, rect.width, rect.height);

			const centerX = rect.width / 2;
			const centerY = rect.height / 2;
			const maxRadius =
				Math.min(rect.width, rect.height) * MAX_RADIUS_FRACTION;

			// Spawn new ripples
			const timeSinceLastRipple = timestamp - lastRippleTimeRef.current;

			if (currentIsPlaying) {
				// Active: ripple frequency scales with energy
				const interval =
					MAX_RIPPLE_INTERVAL -
					(MAX_RIPPLE_INTERVAL - MIN_RIPPLE_INTERVAL) *
						Math.min(currentEnergy * 5, 1); // energy ~0.04-0.2 maps to 0-1
				if (timeSinceLastRipple > interval) {
					ripplesRef.current.push({
						birthTime: timestamp,
						maxRadius: maxRadius * (0.6 + currentEnergy * 4 * 0.4), // bigger at higher energy
						wobblePhase1: Math.random() * Math.PI * 2,
						wobblePhase2: Math.random() * Math.PI * 2,
						baseOpacity: Math.min(0.15 + currentEnergy * 3, 0.55), // brighter at higher energy
					});
					lastRippleTimeRef.current = timestamp;
				}
			} else {
				// Idle: one faint ripple every ~3s
				if (timeSinceLastRipple > IDLE_RIPPLE_INTERVAL) {
					ripplesRef.current.push({
						birthTime: timestamp,
						maxRadius: maxRadius * 0.5,
						wobblePhase1: Math.random() * Math.PI * 2,
						wobblePhase2: Math.random() * Math.PI * 2,
						baseOpacity: 0.25,
					});
					lastRippleTimeRef.current = timestamp;
				}
			}

			// Draw and cull ripples
			const alive: Ripple[] = [];

			for (const ripple of ripplesRef.current) {
				const age = (timestamp - ripple.birthTime) / 1000; // seconds
				const radius = age * EXPANSION_SPEED;

				if (radius > ripple.maxRadius) continue; // expired

				alive.push(ripple);

				// Opacity fades linearly with radius
				const progress = radius / ripple.maxRadius;
				const opacity = ripple.baseOpacity * (1 - progress);

				if (opacity < 0.005) continue; // invisible

				// Draw organic ring
				ctx!.beginPath();
				ctx!.strokeStyle = `rgba(${SAGE_R}, ${SAGE_G}, ${SAGE_B}, ${opacity})`;
				ctx!.lineWidth = 1.8;

				const steps = 64;
				const wobbleNoise = radius * WOBBLE_AMOUNT;

				for (let i = 0; i <= steps; i++) {
					const theta = (i / steps) * Math.PI * 2;
					const wobble =
						wobbleNoise *
						(Math.sin(theta * 3 + ripple.wobblePhase1) * 0.6 +
							Math.sin(theta * 5 + ripple.wobblePhase2) * 0.4);
					const r = radius + wobble;
					const x = centerX + Math.cos(theta) * r;
					const y = centerY + Math.sin(theta) * r;

					if (i === 0) {
						ctx!.moveTo(x, y);
					} else {
						ctx!.lineTo(x, y);
					}
				}

				ctx!.closePath();
				ctx!.stroke();
			}

			ripplesRef.current = alive;

			// Dev logging (throttled)
			frameCountRef.current++;
			if (frameCountRef.current % 60 === 0) {
				log.log(
					`Active ripples: ${alive.length}, energy: ${currentEnergy.toFixed(3)}, isPlaying: ${currentIsPlaying}`,
				);
			}
		}

		if (active) {
			rafRef.current = requestAnimationFrame(draw);
		}

		return () => {
			cancelAnimationFrame(rafRef.current);
			lastFrameTimeRef.current = 0;
		};
	}, [active]); // Only depends on active -- energy/isPlaying read from refs

	return (
		<canvas
			ref={canvasRef}
			className="block w-full h-full"
			style={{ width: "100%", height: "100%" }}
		/>
	);
}

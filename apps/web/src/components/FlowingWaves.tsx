import { useEffect, useRef } from "react";

interface FlowingWavesProps {
	analyserNode: AnalyserNode | null;
	active: boolean;
}

const WAVE_LAYERS = [
	{ frequency: 0.008, speed: 0.6, amplitude: 0.7, opacity: 0.55, width: 2 },
	{ frequency: 0.012, speed: -0.8, amplitude: 0.5, opacity: 0.35, width: 1.8 },
	{ frequency: 0.006, speed: 0.4, amplitude: 0.9, opacity: 0.2, width: 1.5 },
	{ frequency: 0.015, speed: -1.1, amplitude: 0.35, opacity: 0.15, width: 1.2 },
];

// Sage color in RGB for compositing
const SAGE_R = 122;
const SAGE_G = 154;
const SAGE_B = 130;

export function FlowingWaves({ analyserNode, active }: FlowingWavesProps) {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const animFrameRef = useRef<number>(0);
	const smoothedEnergyRef = useRef(0);
	const phaseRef = useRef(WAVE_LAYERS.map(() => Math.random() * Math.PI * 2));
	const lastTimeRef = useRef(0);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;

		const ctx = canvas.getContext("2d");
		if (!ctx) return;

		let dataArray: Uint8Array<ArrayBuffer> | null = null;
		if (analyserNode) {
			dataArray = new Uint8Array(
				analyserNode.frequencyBinCount,
			) as Uint8Array<ArrayBuffer>;
		}

		function draw(timestamp: number) {
			animFrameRef.current = requestAnimationFrame(draw);

			const dt = lastTimeRef.current
				? (timestamp - lastTimeRef.current) / 1000
				: 0.016;
			lastTimeRef.current = timestamp;

			// Resize canvas to match CSS size (handle DPR)
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

			// Compute audio energy
			let energy = 0;
			if (analyserNode && dataArray && active) {
				analyserNode.getByteFrequencyData(dataArray);
				let sum = 0;
				for (let i = 0; i < dataArray.length; i++) {
					sum += dataArray[i];
				}
				energy = sum / (dataArray.length * 255);
			}

			// Smooth energy with lerp
			const lerpFactor = active ? 0.08 : 0.03;
			const targetEnergy = active ? energy : 0.05;
			smoothedEnergyRef.current +=
				(targetEnergy - smoothedEnergyRef.current) * lerpFactor;
			const smoothed = smoothedEnergyRef.current;

			const centerY = rect.height / 2;

			// Draw each wave layer (back to front)
			for (let l = WAVE_LAYERS.length - 1; l >= 0; l--) {
				const layer = WAVE_LAYERS[l];
				phaseRef.current[l] += layer.speed * dt;

				const phase = phaseRef.current[l];
				const amp = centerY * layer.amplitude * (0.15 + smoothed * 0.85);

				ctx!.beginPath();
				ctx!.strokeStyle = `rgba(${SAGE_R}, ${SAGE_G}, ${SAGE_B}, ${layer.opacity})`;
				ctx!.lineWidth = layer.width;
				ctx!.lineCap = "round";
				ctx!.lineJoin = "round";

				const step = 3;

				for (let x = 0; x <= rect.width; x += step) {
					// Combine two sine frequencies for organic movement
					const y =
						centerY +
						Math.sin(x * layer.frequency + phase) * amp * 0.6 +
						Math.sin(x * layer.frequency * 1.7 + phase * 0.7) * amp * 0.4;

					if (x === 0) {
						ctx!.moveTo(x, y);
					} else {
						// Use quadratic curve for smoothness
						const prevX = x - step;
						const prevY =
							centerY +
							Math.sin(prevX * layer.frequency + phase) * amp * 0.6 +
							Math.sin(prevX * layer.frequency * 1.7 + phase * 0.7) * amp * 0.4;
						const cpX = (prevX + x) / 2;
						const cpY = (prevY + y) / 2;
						ctx!.quadraticCurveTo(prevX, prevY, cpX, cpY);
					}
				}

				ctx!.stroke();
			}
		}

		animFrameRef.current = requestAnimationFrame(draw);

		return () => {
			cancelAnimationFrame(animFrameRef.current);
			lastTimeRef.current = 0;
		};
	}, [analyserNode, active]);

	return (
		<canvas
			ref={canvasRef}
			className="block w-full h-full"
			style={{ width: "100%", height: "100%" }}
		/>
	);
}

import { useRef, useEffect } from 'react'

interface WaveformVisualizerProps {
  analyserNode: AnalyserNode | null
  width?: number
  height?: number
}

export function WaveformVisualizer({ analyserNode, width = 280, height = 80 }: WaveformVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animFrameRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !analyserNode) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const bufferLength = analyserNode.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)

    function draw() {
      animFrameRef.current = requestAnimationFrame(draw)
      analyserNode!.getByteFrequencyData(dataArray)

      ctx!.clearRect(0, 0, width, height)

      const barCount = 32
      const barWidth = (width / barCount) * 0.6
      const gap = (width / barCount) * 0.4
      const step = Math.floor(bufferLength / barCount)

      for (let i = 0; i < barCount; i++) {
        const value = dataArray[i * step] / 255
        const barHeight = Math.max(2, value * height * 0.85)

        const x = i * (barWidth + gap)
        const y = (height - barHeight) / 2

        ctx!.fillStyle = '#7A9A82' // sage/accent
        ctx!.beginPath()
        ctx!.roundRect(x, y, barWidth, barHeight, 2)
        ctx!.fill()
      }
    }

    draw()

    return () => cancelAnimationFrame(animFrameRef.current)
  }, [analyserNode, width, height])

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="block"
    />
  )
}

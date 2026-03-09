import { useState, useRef, useCallback } from 'react'
import { practiceApi } from '../lib/practice-api'
import type { PracticeWsEvent, ObservationEvent, DimScores } from '../lib/practice-api'

export type PracticeState =
  | 'idle'
  | 'requesting-mic'
  | 'connecting'
  | 'recording'
  | 'summarizing'
  | 'error'

export interface UsePracticeSessionReturn {
  state: PracticeState
  elapsedSeconds: number
  observations: ObservationEvent[]
  latestScores: DimScores | null
  summary: string | null
  error: string | null
  analyserNode: AnalyserNode | null
  start: () => Promise<void>
  stop: () => void
}

export function usePracticeSession(): UsePracticeSessionReturn {
  const [state, setState] = useState<PracticeState>('idle')
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [observations, setObservations] = useState<ObservationEvent[]>([])
  const [latestScores, setLatestScores] = useState<DimScores | null>(null)
  const [summary, setSummary] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null)

  const sessionIdRef = useRef<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const chunkIndexRef = useRef(0)

  const cleanup = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current)
    if (mediaRecorderRef.current?.state === 'recording') mediaRecorderRef.current.stop()
    if (audioContextRef.current?.state !== 'closed') audioContextRef.current?.close()
    streamRef.current?.getTracks().forEach((t) => t.stop())
    wsRef.current?.close()

    timerRef.current = null
    mediaRecorderRef.current = null
    audioContextRef.current = null
    streamRef.current = null
    wsRef.current = null
    sessionIdRef.current = null
    chunkIndexRef.current = 0
  }, [])

  const start = useCallback(async () => {
    setState('requesting-mic')
    setElapsedSeconds(0)
    setObservations([])
    setLatestScores(null)
    setSummary(null)
    setError(null)

    // 1. Request mic
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
    } catch (e) {
      setState('error')
      setError('Microphone access denied. Please allow mic access and try again.')
      return
    }

    // 2. Set up AudioContext + AnalyserNode for waveform visualization
    const audioCtx = new AudioContext()
    audioContextRef.current = audioCtx
    const source = audioCtx.createMediaStreamSource(stream)
    const analyser = audioCtx.createAnalyser()
    analyser.fftSize = 256
    source.connect(analyser)
    setAnalyserNode(analyser)

    // 3. Start session on server
    setState('connecting')
    let sessionId: string
    try {
      const { sessionId: sid } = await practiceApi.start()
      sessionId = sid
      sessionIdRef.current = sid
    } catch (e) {
      cleanup()
      setState('error')
      setError('Failed to start practice session. Please try again.')
      return
    }

    // 4. Connect WebSocket
    const ws = practiceApi.connectWebSocket(sessionId)
    wsRef.current = ws

    ws.onmessage = (event) => {
      const data: PracticeWsEvent = JSON.parse(event.data)
      switch (data.type) {
        case 'chunk_processed':
          setLatestScores(data.scores)
          break
        case 'observation':
          setObservations((prev) => [...prev, {
            text: data.text,
            dimension: data.dimension,
            framing: data.framing,
          }])
          break
        case 'session_summary':
          setSummary(data.summary)
          setObservations(data.observations)
          setState('idle')
          cleanup()
          break
        case 'error':
          setError(data.message)
          break
      }
    }

    ws.onerror = () => {
      setError('WebSocket connection lost.')
      setState('error')
      cleanup()
    }

    await new Promise<void>((resolve, reject) => {
      ws.onopen = () => resolve()
      ws.onerror = () => reject(new Error('WebSocket failed to connect'))
    })

    // 5. Start MediaRecorder with 15s chunks
    const recorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus',
    })
    mediaRecorderRef.current = recorder
    chunkIndexRef.current = 0

    recorder.ondataavailable = async (event) => {
      if (event.data.size === 0) return
      const idx = chunkIndexRef.current++
      try {
        const { r2Key } = await practiceApi.uploadChunk(sessionId, idx, event.data)
        ws.send(JSON.stringify({ type: 'chunk_ready', index: idx, r2Key }))
      } catch (e) {
        console.error('Chunk upload failed:', e)
      }
    }

    recorder.start(15000) // 15-second timeslice
    setState('recording')

    // 6. Start elapsed timer
    const startTime = Date.now()
    timerRef.current = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000))
    }, 1000)
  }, [cleanup])

  const stop = useCallback(() => {
    if (state !== 'recording') return
    setState('summarizing')

    // Stop recording (triggers final ondataavailable)
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop()
    }

    // Tell DO to end session
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'end_session' }))
    }

    // Timer cleanup (WS stays open until summary arrives)
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [state])

  return {
    state,
    elapsedSeconds,
    observations,
    latestScores,
    summary,
    error,
    analyserNode,
    start,
    stop,
  }
}

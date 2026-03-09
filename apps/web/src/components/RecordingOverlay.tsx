import { useState, useCallback } from 'react'
import { Stop, CircleNotch } from '@phosphor-icons/react'
import { WaveformVisualizer } from './WaveformVisualizer'
import { ObservationToast } from './ObservationToast'
import type { PracticeState } from '../hooks/usePracticeSession'
import type { ObservationEvent } from '../lib/practice-api'

interface RecordingOverlayProps {
  state: PracticeState
  elapsedSeconds: number
  observations: ObservationEvent[]
  analyserNode: AnalyserNode | null
  error: string | null
  onStop: () => void
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

export function RecordingOverlay({
  state,
  elapsedSeconds,
  observations,
  analyserNode,
  error,
  onStop,
}: RecordingOverlayProps) {
  const [dismissedIds, setDismissedIds] = useState<Set<number>>(new Set())

  const handleDismiss = useCallback((idx: number) => {
    setDismissedIds((prev) => new Set(prev).add(idx))
  }, [])

  const visibleObservations = observations
    .map((obs, idx) => ({ ...obs, idx }))
    .filter(({ idx }) => !dismissedIds.has(idx))
    .slice(-3) // Show max 3 toasts at once

  const isConnecting = state === 'requesting-mic' || state === 'connecting'
  const isSummarizing = state === 'summarizing'

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center pb-32 pointer-events-none">
      {/* Overlay card */}
      <div className="pointer-events-auto bg-espresso/95 backdrop-blur-md border border-border rounded-3xl px-8 py-6 shadow-card animate-overlay-in flex flex-col items-center gap-4 min-w-[340px]">
        {/* Status text */}
        {isConnecting && (
          <div className="flex items-center gap-2 text-text-secondary text-body-sm">
            <CircleNotch size={16} className="animate-spin" />
            <span>Connecting...</span>
          </div>
        )}

        {isSummarizing && (
          <div className="flex items-center gap-2 text-text-secondary text-body-sm">
            <CircleNotch size={16} className="animate-spin" />
            <span>Generating summary...</span>
          </div>
        )}

        {/* Waveform */}
        {state === 'recording' && (
          <>
            <WaveformVisualizer analyserNode={analyserNode} />

            {/* Timer */}
            <span className="font-display text-display-sm text-cream tabular-nums">
              {formatTime(elapsedSeconds)}
            </span>

            {/* Stop button */}
            <button
              type="button"
              onClick={onStop}
              className="w-14 h-14 flex items-center justify-center rounded-full bg-red-600 hover:bg-red-500 text-cream transition"
              aria-label="Stop recording"
            >
              <Stop size={24} weight="fill" />
            </button>
          </>
        )}

        {/* Error */}
        {error && (
          <p className="text-body-sm text-red-400 text-center max-w-xs">{error}</p>
        )}
      </div>

      {/* Toast stack (positioned to the right) */}
      <div className="pointer-events-auto fixed right-6 bottom-32 flex flex-col gap-3">
        {visibleObservations.map(({ idx, text, dimension }) => (
          <ObservationToast
            key={idx}
            text={text}
            dimension={dimension}
            onDismiss={() => handleDismiss(idx)}
          />
        ))}
      </div>
    </div>
  )
}

import { useEffect, useState } from 'react'

interface ObservationToastProps {
  text: string
  dimension: string
  onDismiss: () => void
  autoHideMs?: number
}

export function ObservationToast({ text, dimension, onDismiss, autoHideMs = 8000 }: ObservationToastProps) {
  const [exiting, setExiting] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => {
      setExiting(true)
      setTimeout(onDismiss, 300) // Wait for exit animation
    }, autoHideMs)
    return () => clearTimeout(timer)
  }, [autoHideMs, onDismiss])

  return (
    <div
      className={`max-w-sm bg-surface-card border border-border rounded-xl px-4 py-3 shadow-card ${
        exiting ? 'animate-slide-out-right' : 'animate-slide-in-right'
      }`}
    >
      <span className="block text-body-xs text-accent font-medium uppercase tracking-wide mb-1">
        {dimension}
      </span>
      <p className="text-body-sm text-cream leading-relaxed">
        {text}
      </p>
    </div>
  )
}

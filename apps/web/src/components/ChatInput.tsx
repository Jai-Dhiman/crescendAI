import { useState, useRef, useEffect } from 'react'
import { PaperPlaneTilt, Waveform } from '@phosphor-icons/react'

interface ChatInputProps {
  onSend: (message: string) => void
  onRecord?: () => void
  disabled: boolean
  placeholder?: string
  centered?: boolean
}

export function ChatInput({ onSend, onRecord, disabled, placeholder, centered }: ChatInputProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const hasText = value.trim().length > 0

  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [value])

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  function handleSend() {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
  }

  return (
    <div className={centered ? 'w-full max-w-2xl' : 'px-6 py-4 pb-6 animate-input-settle'}>
      <div className={`flex items-center gap-3 ${centered ? '' : 'max-w-2xl mx-auto'}`}>
        <div className="flex-1 bg-surface-card border border-border rounded-2xl shadow-card flex items-end px-4 py-2">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder ?? 'Message your teacher...'}
            disabled={disabled}
            rows={3}
            className="flex-1 bg-transparent py-1.5 text-body-md text-cream placeholder:text-text-tertiary outline-none resize-none min-h-[4.5rem]"
          />

          {hasText && (
            <button
              type="button"
              onClick={handleSend}
              disabled={disabled}
              className="shrink-0 w-8 h-8 flex items-center justify-center rounded-full bg-accent text-cream hover:brightness-110 transition animate-pop-in"
            >
              <PaperPlaneTilt size={16} weight="fill" />
            </button>
          )}
        </div>

        {!hasText && (
          <button
            type="button"
            onClick={onRecord}
            className="shrink-0 w-16 h-16 flex items-center justify-center rounded-full bg-accent text-cream hover:brightness-110 transition animate-pop-in"
            aria-label="Record audio"
          >
            <Waveform size={24} />
          </button>
        )}
      </div>
    </div>
  )
}

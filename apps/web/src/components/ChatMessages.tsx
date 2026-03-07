import { useEffect, useRef } from 'react'
import type { MessageRow } from '../lib/api'

interface ChatMessagesProps {
  messages: MessageRow[]
  streamingContent: string | null
}

export function ChatMessages({ messages, streamingContent }: ChatMessagesProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingContent])

  if (messages.length === 0 && !streamingContent) {
    return null
  }

  return (
    <div className="flex-1 overflow-y-auto px-6 py-8">
      <div className="max-w-2xl mx-auto space-y-6">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} role={msg.role} content={msg.content} />
        ))}
        {streamingContent !== null && (
          <MessageBubble role="assistant" content={streamingContent} />
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

function MessageBubble({ role, content }: { role: string; content: string }) {
  if (role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="bg-surface border border-border rounded-2xl px-5 py-3 max-w-[80%]">
          <p className="text-body-md text-cream whitespace-pre-wrap">{content}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%]">
        <p className="text-body-md text-cream whitespace-pre-wrap">{content}</p>
      </div>
    </div>
  )
}

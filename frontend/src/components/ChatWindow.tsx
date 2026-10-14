import { useEffect, useRef, useState } from 'react'
import { askQuestion, getChatHistory } from '../api'
import type { ChatMessage } from '../types'
import MessageBubble from './MessageBubble'

let nextId = 1

export default function ChatWindow() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [loadingHistory, setLoadingHistory] = useState(true)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    getChatHistory()
      .then((history) => {
        const loaded: ChatMessage[] = history.map((m) => ({
          id: m.id,
          role: m.role,
          content: m.content,
          // History only stores chunk IDs, not full snippets/scores, so a
          // past answer's "Show sources" panel can't be rebuilt - only the
          // answer text, triad scores, and timings survive a reload.
          triad_scores: m.triad_scores,
          timings: m.timings,
          graph_facts: m.graph_facts,
        }))
        setMessages(loaded)
        nextId = (history.at(-1)?.id ?? 0) + 1
      })
      .catch(() => {
        // Transient network hiccup on load - start with an empty chat rather than blocking the UI.
      })
      .finally(() => setLoadingHistory(false))
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function handleSend() {
    const question = input.trim()
    if (!question || sending) return

    setInput('')
    setSending(true)
    const userMsg: ChatMessage = { id: nextId++, role: 'user', content: question }
    const pendingMsg: ChatMessage = { id: nextId++, role: 'assistant', content: '', pending: true }
    setMessages((prev) => [...prev, userMsg, pendingMsg])

    try {
      const res = await askQuestion(question)
      setMessages((prev) =>
        prev.map((m) =>
          m.id === pendingMsg.id
            ? {
                ...m,
                pending: false,
                content: res.answer,
                sources: res.sources,
                triad_scores: res.triad_scores,
                timings: res.timings,
                graph_facts: res.graph_facts,
              }
            : m,
        ),
      )
    } catch (e) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === pendingMsg.id
            ? { ...m, pending: false, content: `Error: ${e instanceof Error ? e.message : String(e)}` }
            : m,
        ),
      )
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="flex h-full flex-1 flex-col bg-white dark:bg-neutral-900">
      <div className="flex-1 space-y-4 overflow-y-auto p-6">
        {!loadingHistory && messages.length === 0 && (
          <p className="mt-10 text-center text-sm text-neutral-400">
            Ask a question about your uploaded documents.
          </p>
        )}
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-neutral-200 p-4 dark:border-neutral-800">
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
            placeholder="Hỏi điều gì đó về tài liệu của bạn…"
            rows={2}
            className="flex-1 resize-none rounded-lg border border-neutral-300 p-2 text-sm outline-none focus:border-blue-400 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-100"
          />
          <button
            onClick={handleSend}
            disabled={sending || !input.trim()}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  )
}

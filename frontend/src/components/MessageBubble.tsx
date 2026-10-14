import { useState } from 'react'
import type { ChatMessage, Timings } from '../types'

function formatMs(ms: number | null): string | null {
  if (ms == null) return null
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`
}

function TimingLine({ timings }: { timings: Timings }) {
  const parts: Array<[string, number | null]> = [
    ['retrieve', timings.retrieval_ms],
    ['rerank', timings.rerank_ms],
    ['graph lookup', timings.graph_lookup_ms],
    ['generate', timings.generation_ms],
    ['eval', timings.triad_eval_ms],
  ]
  const breakdown = parts
    .filter(([, ms]) => ms != null)
    .map(([label, ms]) => `${label} ${formatMs(ms)}`)
    .join(' · ')

  return (
    <div className="mt-3 border-t border-neutral-300/50 pt-2 font-mono text-xs text-neutral-400 dark:border-neutral-700 dark:text-neutral-500">
      ⏱ {formatMs(timings.total_ms) ?? '—'} total{breakdown ? ` · ${breakdown}` : ''}
    </div>
  )
}

function TriadBar({ label, value }: { label: string; value: number | null }) {
  const pct = value != null ? Math.round(Math.max(0, Math.min(1, value)) * 100) : 0
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-32 shrink-0 text-neutral-500 dark:text-neutral-400">{label}</span>
      <div className="h-2 flex-1 rounded-full bg-neutral-200 dark:bg-neutral-800">
        {value != null && <div className="h-2 rounded-full bg-blue-500" style={{ width: `${pct}%` }} />}
      </div>
      <span className="w-10 shrink-0 text-right text-neutral-500 dark:text-neutral-400">
        {value != null ? value.toFixed(2) : 'N/A'}
      </span>
    </div>
  )
}

export default function MessageBubble({ message }: { message: ChatMessage }) {
  const [showSources, setShowSources] = useState(false)
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-2xl rounded-2xl px-4 py-3 text-sm ${
          isUser
            ? 'bg-blue-600 text-white'
            : 'bg-neutral-100 text-neutral-900 dark:bg-neutral-800 dark:text-neutral-100'
        }`}
      >
        {message.pending ? (
          <span className="animate-pulse text-neutral-400">Thinking…</span>
        ) : (
          <p className="whitespace-pre-wrap">{message.content}</p>
        )}

        {!isUser && !message.pending && !!message.sources?.length && (
          <div className="mt-3 border-t border-neutral-300/50 pt-2 dark:border-neutral-700">
            <button
              onClick={() => setShowSources((v) => !v)}
              className="text-xs font-medium text-blue-600 hover:underline dark:text-blue-400"
            >
              {showSources ? 'Hide sources' : `Show sources (${message.sources.length})`}
            </button>
            {showSources && (
              <div className="mt-2 space-y-2">
                {message.sources?.map((s) => (
                  <div
                    key={s.chunk_id}
                    className="rounded bg-white/60 p-2 text-xs text-neutral-600 dark:bg-black/20 dark:text-neutral-300"
                  >
                    <div className="mb-1 flex justify-between font-medium text-neutral-500 dark:text-neutral-400">
                      <span>{s.source_document ?? s.chunk_id}</span>
                      <span className="flex gap-2">
                        {s.rrf_rank != null && <span title="Position after RRF fusion, before reranking">RRF #{s.rrf_rank}</span>}
                        {s.rerank_score != null && <span title="Local cross-encoder relevance score">rerank {s.rerank_score.toFixed(2)}</span>}
                      </span>
                    </div>
                    {s.snippet}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {!isUser && !message.pending && message.graph_facts && message.graph_facts.length > 0 && (
          <div className="mt-3 space-y-1 border-t border-neutral-300/50 pt-2 dark:border-neutral-700">
            <div className="text-xs font-medium text-neutral-500 dark:text-neutral-400">🕸 Graph facts</div>
            {message.graph_facts.map((fact) => (
              <div key={fact} className="text-xs text-neutral-600 dark:text-neutral-300">
                {fact}
              </div>
            ))}
          </div>
        )}

        {!isUser && !message.pending && message.triad_scores && (
          <div className="mt-3 space-y-1 border-t border-neutral-300/50 pt-2 dark:border-neutral-700">
            <TriadBar label="Context Relevance" value={message.triad_scores.context_relevance} />
            <TriadBar label="Groundedness" value={message.triad_scores.groundedness} />
            <TriadBar label="Answer Relevance" value={message.triad_scores.answer_relevance} />
            <TriadBar label="Overall Score" value={message.triad_scores.composite} />
          </div>
        )}

        {!isUser && !message.pending && message.timings && <TimingLine timings={message.timings} />}
      </div>
    </div>
  )
}

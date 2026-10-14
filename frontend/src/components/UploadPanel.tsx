import { useCallback, useEffect, useRef, useState } from 'react'
import { deleteDocument, listDocuments, uploadDocument } from '../api'
import type { DocumentOut } from '../types'

const ACCEPT = '.pdf,.docx,.html,.htm,.txt,.md'

const STATUS_STYLE: Record<DocumentOut['status'], string> = {
  processing: 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300',
  indexed: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300',
  failed: 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300',
}

export default function UploadPanel({ onIndexChanged }: { onIndexChanged: () => void }) {
  const [documents, setDocuments] = useState<DocumentOut[]>([])
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const refresh = useCallback(async () => {
    try {
      setDocuments(await listDocuments())
    } catch {
      // transient network hiccup - keep the previous list rather than clearing it
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  async function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return
    setUploading(true)
    setUploadError(null)
    for (const file of Array.from(files)) {
      try {
        await uploadDocument(file)
      } catch (e) {
        setUploadError(e instanceof Error ? e.message : String(e))
      }
    }
    setUploading(false)
    await refresh()
    onIndexChanged()
  }

  async function handleDelete(id: number) {
    await deleteDocument(id)
    await refresh()
    onIndexChanged()
  }

  return (
    <aside className="flex h-full w-80 shrink-0 flex-col border-r border-neutral-200 bg-neutral-50 dark:border-neutral-800 dark:bg-neutral-950">
      <div className="border-b border-neutral-200 p-4 dark:border-neutral-800">
        <h1 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">Vietnamese RAG</h1>
        <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
          Upload PDF, DOCX, HTML, or text files, then ask questions below.
        </p>
      </div>

      <div className="p-4">
        <div
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            handleFiles(e.dataTransfer.files)
          }}
          onClick={() => inputRef.current?.click()}
          className={`cursor-pointer rounded-lg border-2 border-dashed p-6 text-center text-sm transition-colors ${
            dragOver
              ? 'border-blue-400 bg-blue-50 dark:bg-blue-950/30'
              : 'border-neutral-300 text-neutral-500 hover:border-neutral-400 dark:border-neutral-700 dark:text-neutral-400'
          }`}
        >
          {uploading ? 'Uploading…' : 'Drag files here or click to browse'}
          <input
            ref={inputRef}
            type="file"
            multiple
            accept={ACCEPT}
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>
        {uploadError && <p className="mt-2 text-sm text-red-600 dark:text-red-400">{uploadError}</p>}
      </div>

      <div className="flex-1 overflow-y-auto px-4 pb-4">
        {documents.length === 0 && (
          <p className="text-sm text-neutral-400 dark:text-neutral-500">No documents uploaded yet.</p>
        )}
        <ul className="space-y-2">
          {documents.map((doc) => (
            <li
              key={doc.id}
              className="rounded-md border border-neutral-200 bg-white p-3 text-sm dark:border-neutral-800 dark:bg-neutral-900"
            >
              <div className="flex items-start justify-between gap-2">
                <span className="truncate font-medium text-neutral-800 dark:text-neutral-200" title={doc.filename}>
                  {doc.filename}
                </span>
                <button
                  onClick={() => handleDelete(doc.id)}
                  className="text-neutral-400 hover:text-red-500"
                  title="Delete"
                >
                  ✕
                </button>
              </div>
              <div className="mt-2 flex items-center gap-2">
                <span className={`rounded px-1.5 py-0.5 text-xs font-medium ${STATUS_STYLE[doc.status]}`}>
                  {doc.status}
                </span>
                <span className="text-xs text-neutral-500 dark:text-neutral-400">
                  {doc.num_chunks} chunks
                </span>
              </div>
              {doc.error && <p className="mt-1 text-xs text-red-500">{doc.error}</p>}
            </li>
          ))}
        </ul>
      </div>
    </aside>
  )
}

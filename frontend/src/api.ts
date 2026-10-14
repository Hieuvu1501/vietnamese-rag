import type { ChatResponse, DocumentOut, GraphData, MessageOut } from './types'

const BASE = '/api'

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed: ${res.status}`)
  }
  return res.json() as Promise<T>
}

export async function listDocuments(): Promise<DocumentOut[]> {
  const res = await fetch(`${BASE}/documents`)
  return handle(res)
}

export async function uploadDocument(file: File): Promise<DocumentOut> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/documents`, { method: 'POST', body: form })
  return handle(res)
}

export async function deleteDocument(id: number): Promise<void> {
  const res = await fetch(`${BASE}/documents/${id}`, { method: 'DELETE' })
  await handle(res)
}

export async function getChatHistory(): Promise<MessageOut[]> {
  const res = await fetch(`${BASE}/chat/history`)
  return handle(res)
}

export async function getGraph(): Promise<GraphData> {
  const res = await fetch(`${BASE}/graph`)
  return handle(res)
}

export async function askQuestion(question: string): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  return handle(res)
}

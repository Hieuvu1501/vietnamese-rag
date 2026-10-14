import { useState } from 'react'
import ChatWindow from './components/ChatWindow'
import KnowledgeMap from './components/KnowledgeMap'
import UploadPanel from './components/UploadPanel'

type Tab = 'chat' | 'map'

const TABS: { id: Tab; label: string }[] = [
  { id: 'chat', label: 'Chat' },
  { id: 'map', label: 'Knowledge map' },
]

function App() {
  const [tab, setTab] = useState<Tab>('chat')
  // Bumped on upload/delete so an open map remounts and refetches.
  const [indexVersion, setIndexVersion] = useState(0)

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <UploadPanel onIndexChanged={() => setIndexVersion((v) => v + 1)} />
      <div className="flex min-w-0 flex-1 flex-col">
        <nav className="flex gap-1 border-b border-neutral-200 bg-white px-4 pt-2 dark:border-neutral-800 dark:bg-neutral-900">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`-mb-px border-b-2 px-3 py-2 text-sm font-medium ${
                tab === t.id
                  ? 'border-blue-600 text-blue-600 dark:border-blue-400 dark:text-blue-400'
                  : 'border-transparent text-neutral-500 hover:text-neutral-800 dark:text-neutral-400 dark:hover:text-neutral-200'
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
        {/* Chat stays mounted while hidden so an in-flight answer isn't lost when switching tabs. */}
        <div className={tab === 'chat' ? 'flex min-h-0 flex-1' : 'hidden'}>
          <ChatWindow />
        </div>
        {tab === 'map' && <KnowledgeMap key={indexVersion} />}
      </div>
    </div>
  )
}

export default App

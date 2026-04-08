import { useState } from 'react'
import { StatusPanel } from './components/StatusPanel'
import { ImageUpload } from './components/ImageUpload'
import { SegmentPanel } from './components/SegmentPanel'
import { AutoSegmentPanel } from './components/AutoSegmentPanel'
import { CvToolsPanel } from './components/CvToolsPanel'
import { SessionHistory } from './components/SessionHistory'
import { useSessionHistory } from './hooks/useSessionHistory'
import type { ImageMeta } from './types'
import './App.css'

type ActiveTab = 'segment' | 'auto' | 'cv'

const TABS: { id: ActiveTab; label: string }[] = [
  { id: 'segment', label: 'Prompt Segment' },
  { id: 'auto', label: 'Auto Segment' },
  { id: 'cv', label: 'CV Tools' },
]

export default function App() {
  const [image, setImage] = useState<ImageMeta | null>(null)
  const [activeTab, setActiveTab] = useState<ActiveTab>('segment')
  const { entries, addEntry, clearHistory } = useSessionHistory()

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-header__title">CV SAM – Local Test UI</h1>
        <p className="app-header__subtitle">
          Direct operator access · no control-plane dependency
        </p>
      </header>

      <main className="app-main">
        <StatusPanel />
        <ImageUpload image={image} onImage={setImage} />

        <nav className="tab-nav" aria-label="Workflow tabs">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              className={`tab-btn${activeTab === tab.id ? ' tab-btn--active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {activeTab === 'segment' && <SegmentPanel image={image} onResult={addEntry} />}
        {activeTab === 'auto' && <AutoSegmentPanel image={image} onResult={addEntry} />}
        {activeTab === 'cv' && <CvToolsPanel image={image} onResult={addEntry} />}

        <SessionHistory entries={entries} onClear={clearHistory} />
      </main>
    </div>
  )
}

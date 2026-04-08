import { useState } from 'react'
import { StatusPanel } from './components/StatusPanel'
import { ImageUpload } from './components/ImageUpload'
import type { ImageMeta } from './types'
import './App.css'

export default function App() {
  const [image, setImage] = useState<ImageMeta | null>(null)

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
      </main>
    </div>
  )
}

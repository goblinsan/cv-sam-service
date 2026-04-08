import { useCallback, useRef, useState } from 'react'
import type { ImageMeta } from '../types'

const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/webp']
const ACCEPTED_EXT = '.png,.jpg,.jpeg,.webp'

function formatLabel(mime: string): string {
  return mime.replace('image/', '').toUpperCase()
}

function readImageMeta(file: File): Promise<ImageMeta> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => {
      resolve({
        file,
        width: img.naturalWidth,
        height: img.naturalHeight,
        format: formatLabel(file.type),
        previewUrl: url,
      })
    }
    img.onerror = () => {
      URL.revokeObjectURL(url)
      reject(new Error('Failed to decode image'))
    }
    img.src = url
  })
}

interface Props {
  onImage: (meta: ImageMeta | null) => void
  image: ImageMeta | null
}

export function ImageUpload({ onImage, image }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFile = useCallback(
    async (file: File | null) => {
      setError(null)
      if (!file) return
      if (!ACCEPTED_TYPES.includes(file.type)) {
        setError(`Unsupported format: ${file.type || 'unknown'}. Use PNG, JPEG, or WebP.`)
        return
      }
      try {
        const meta = await readImageMeta(file)
        onImage(meta)
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Could not load image')
      }
    },
    [onImage],
  )

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      setDragging(false)
      const file = e.dataTransfer.files[0] ?? null
      void handleFile(file)
    },
    [handleFile],
  )

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      void handleFile(e.target.files?.[0] ?? null)
      // Reset the input so the same file can be re-selected after a clear
      e.target.value = ''
    },
    [handleFile],
  )

  const handleClear = useCallback(() => {
    if (image) URL.revokeObjectURL(image.previewUrl)
    onImage(null)
    setError(null)
  }, [image, onImage])

  return (
    <section className="panel upload-panel">
      <header className="panel__header">
        <h2 className="panel__title">Image</h2>
        {image && (
          <div className="panel__actions">
            <button className="btn btn--sm" onClick={() => inputRef.current?.click()}>
              Replace
            </button>
            <button className="btn btn--sm btn--danger" onClick={handleClear}>
              Clear
            </button>
          </div>
        )}
      </header>

      {error && <p className="error-msg">{error}</p>}

      {image ? (
        <div className="image-preview">
          <img
            src={image.previewUrl}
            alt="Uploaded preview"
            className="image-preview__thumb"
          />
          <dl className="image-preview__meta">
            <dt>Dimensions</dt>
            <dd>
              {image.width} × {image.height} px
            </dd>
            <dt>Format</dt>
            <dd>{image.format}</dd>
            <dt>File size</dt>
            <dd>{(image.file.size / 1024).toFixed(1)} KB</dd>
            <dt>File name</dt>
            <dd className="image-preview__filename">{image.file.name}</dd>
          </dl>
        </div>
      ) : (
        <div
          className={`drop-zone${dragging ? ' drop-zone--active' : ''}`}
          role="button"
          aria-label="Drop an image here or click to browse"
          tabIndex={0}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
          }}
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <span className="drop-zone__icon" aria-hidden="true">🖼</span>
          <p className="drop-zone__text">
            Drag &amp; drop an image here, or <span className="drop-zone__link">browse</span>
          </p>
          <p className="drop-zone__hint">PNG · JPEG · WebP</p>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_EXT}
        className="visually-hidden"
        onChange={handleInputChange}
        aria-hidden="true"
      />
    </section>
  )
}

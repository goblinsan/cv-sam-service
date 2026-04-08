import { useRef, useState } from 'react'
import { segmentAuto } from '../api/client'
import type { AutoSegment, AutoSegmentResponse, ImageMeta } from '../types'

const SEG_COLORS = [
  '#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4',
  '#d946ef', '#84cc16', '#f97316', '#3b82f6', '#e11d48',
]

interface Props {
  image: ImageMeta | null
}

function colorForIndex(i: number): string {
  return SEG_COLORS[i % SEG_COLORS.length]
}

export function AutoSegmentPanel({ image }: Props) {
  const imgRef = useRef<HTMLImageElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const [maxMasks, setMaxMasks] = useState(50)
  const [outputFormat, setOutputFormat] = useState<'masks' | 'polygons' | 'both'>('masks')
  const [result, setResult] = useState<AutoSegmentResponse | null>(null)
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleRun = async () => {
    if (!image) return
    setLoading(true)
    setError(null)
    setResult(null)
    setSelectedIdx(null)
    try {
      const res = await segmentAuto({
        image: image.file,
        maxMasks,
        outputFormat,
      })
      setResult(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  // Draw the selected mask onto the canvas overlay
  const handleSelectMask = (idx: number) => {
    setSelectedIdx(idx)
    drawMaskOverlay(idx)
  }

  const drawMaskOverlay = (idx: number | null) => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img || !result) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

    if (idx === null) return
    const seg = result.segments[idx]
    if (!seg.mask) return

    const maskImg = new Image()
    maskImg.onload = () => {
      const offscreen = document.createElement('canvas')
      offscreen.width = canvas.width
      offscreen.height = canvas.height
      const octx = offscreen.getContext('2d')!
      octx.drawImage(maskImg, 0, 0, canvas.width, canvas.height)
      const pixel = octx.getImageData(0, 0, canvas.width, canvas.height)

      const color = colorForIndex(idx)
      const r = parseInt(color.slice(1, 3), 16)
      const g = parseInt(color.slice(3, 5), 16)
      const b = parseInt(color.slice(5, 7), 16)

      const overlay = ctx.createImageData(canvas.width, canvas.height)
      for (let i = 0; i < pixel.data.length; i += 4) {
        if (pixel.data[i] > 128) {
          overlay.data[i] = r
          overlay.data[i + 1] = g
          overlay.data[i + 2] = b
          overlay.data[i + 3] = 120
        }
      }
      ctx.putImageData(overlay, 0, 0)

      // Draw bounding box
      const [bx, by, bw, bh] = seg.bbox
      const scaleX = canvas.width / img.naturalWidth
      const scaleY = canvas.height / img.naturalHeight
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.strokeRect(bx * scaleX, by * scaleY, bw * scaleX, bh * scaleY)
    }
    maskImg.src = `data:image/png;base64,${seg.mask}`
  }

  const canvasWidth = 640
  const canvasHeight = image ? Math.round((image.height / image.width) * canvasWidth) : 360

  const selected: AutoSegment | null = result && selectedIdx !== null ? result.segments[selectedIdx] : null

  return (
    <section className="panel">
      <header className="panel__header">
        <h2 className="panel__title">Auto Segmentation</h2>
        <button
          className="btn btn--sm btn--primary"
          onClick={() => void handleRun()}
          disabled={loading || !image}
        >
          {loading ? 'Running…' : 'Run /api/segment/auto'}
        </button>
      </header>

      {!image && <p className="hint-msg">Load an image above to begin.</p>}
      {error && <p className="error-msg">{error}</p>}

      {image && (
        <>
          <div className="seg-controls">
            <div className="seg-controls__group">
              <label className="seg-controls__label" htmlFor="auto-max-masks">
                Max masks: <strong>{maxMasks}</strong>
              </label>
              <input
                id="auto-max-masks"
                type="range"
                min={1}
                max={200}
                value={maxMasks}
                onChange={(e) => setMaxMasks(Number(e.target.value))}
                className="range-input"
              />
            </div>
            <div className="seg-controls__group">
              <span className="seg-controls__label">Output</span>
              <select
                className="select-sm"
                value={outputFormat}
                onChange={(e) => setOutputFormat(e.target.value as typeof outputFormat)}
              >
                <option value="masks">Masks</option>
                <option value="polygons">Polygons</option>
                <option value="both">Both</option>
              </select>
            </div>
          </div>

          {/* Canvas */}
          <div className="canvas-wrap">
            <img
              ref={imgRef}
              src={image.previewUrl}
              alt=""
              style={{ display: 'none' }}
              onLoad={() => drawMaskOverlay(selectedIdx)}
            />
            <canvas
              ref={canvasRef}
              width={canvasWidth}
              height={canvasHeight}
              className="seg-canvas"
            />
          </div>
        </>
      )}

      {result && (
        <div className="auto-results">
          <p className="auto-results__summary">
            {result.count} segment{result.count !== 1 ? 's' : ''} · {result.processing_time_ms.toFixed(0)} ms
          </p>

          <div className="seg-table-wrap">
            <table className="seg-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Score</th>
                  <th>Stability</th>
                  <th>Area (px)</th>
                  <th>BBox</th>
                </tr>
              </thead>
              <tbody>
                {result.segments.map((seg, i) => (
                  <tr
                    key={i}
                    className={`seg-table__row${selectedIdx === i ? ' seg-table__row--selected' : ''}`}
                    onClick={() => handleSelectMask(i)}
                    style={{ '--row-color': colorForIndex(i) } as React.CSSProperties}
                  >
                    <td>
                      <span className="seg-dot" style={{ background: colorForIndex(i) }} />
                      {i + 1}
                    </td>
                    <td>{(seg.score * 100).toFixed(1)}%</td>
                    <td>{(seg.stability_score * 100).toFixed(1)}%</td>
                    <td>{seg.area.toLocaleString()}</td>
                    <td className="seg-table__bbox">
                      {seg.bbox.map((v) => Math.round(v)).join(', ')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {selected && (
            <div className="seg-detail">
              <h3 className="seg-detail__title">Segment {selectedIdx! + 1} detail</h3>
              <dl className="seg-detail__dl">
                <dt>Score</dt><dd>{(selected.score * 100).toFixed(2)}%</dd>
                <dt>Stability</dt><dd>{(selected.stability_score * 100).toFixed(2)}%</dd>
                <dt>Area</dt><dd>{selected.area.toLocaleString()} px</dd>
                <dt>BBox</dt><dd>[{selected.bbox.map((v) => Math.round(v)).join(', ')}]</dd>
                {selected.polygon && (
                  <>
                    <dt>Polygon pts</dt><dd>{selected.polygon.length}</dd>
                  </>
                )}
              </dl>
            </div>
          )}
        </div>
      )}
    </section>
  )
}

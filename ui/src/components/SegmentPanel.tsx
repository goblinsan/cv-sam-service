import { useCallback, useEffect, useRef, useState } from 'react'
import { segment } from '../api/client'
import type { ActionType, ImageMeta, SegmentResponse } from '../types'
import { downloadCanvas, downloadCsv, downloadJson } from '../utils/exportUtils'

// Colors for overlaying up to 3 masks
const MASK_COLORS = [
  'rgba(99, 102, 241, 0.45)',
  'rgba(34, 197, 94, 0.45)',
  'rgba(245, 158, 11, 0.45)',
]

interface ClickPoint {
  x: number
  y: number
  label: 1 | 0
}

interface Props {
  image: ImageMeta | null
  onResult: (entry: { imageName: string; action: ActionType; summary: string }) => void
}

function drawOverlay(
  canvas: HTMLCanvasElement,
  img: HTMLImageElement,
  points: ClickPoint[],
  box: [number, number, number, number] | null,
  result: SegmentResponse | null,
  activeMask: number | null,
) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  const { width: cw, height: ch } = canvas
  ctx.clearRect(0, 0, cw, ch)

  // Draw source image
  ctx.drawImage(img, 0, 0, cw, ch)

  const scaleX = cw / img.naturalWidth
  const scaleY = ch / img.naturalHeight

  // Draw masks
  if (result?.masks) {
    const masksToDraw =
      activeMask !== null ? [{ mask: result.masks[activeMask], idx: activeMask }] : result.masks.map((m, i) => ({ mask: m, idx: i }))

    for (const { mask, idx } of masksToDraw) {
      const maskImg = new Image()
      maskImg.onload = () => {
        const offscreen = document.createElement('canvas')
        offscreen.width = cw
        offscreen.height = ch
        const octx = offscreen.getContext('2d')!
        octx.drawImage(maskImg, 0, 0, cw, ch)
        const pixel = octx.getImageData(0, 0, cw, ch)
        // Tint mask with color
        const color = MASK_COLORS[idx % MASK_COLORS.length]
        const tmp = document.createElement('canvas')
        tmp.width = cw
        tmp.height = ch
        const tctx = tmp.getContext('2d')!
        tctx.fillStyle = color
        tctx.fillRect(0, 0, cw, ch)
        const tpix = tctx.getImageData(0, 0, cw, ch)
        for (let i = 0; i < pixel.data.length; i += 4) {
          // Only tint where the mask is white (>128)
          if (pixel.data[i] < 128) tpix.data[i + 3] = 0
        }
        tctx.putImageData(tpix, 0, 0)
        ctx.drawImage(tmp, 0, 0)
      }
      maskImg.src = `data:image/png;base64,${mask}`
    }
  }

  // Draw box prompt
  if (box) {
    const [x1, y1, x2, y2] = box
    ctx.strokeStyle = '#f59e0b'
    ctx.lineWidth = 2
    ctx.setLineDash([6, 3])
    ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY)
    ctx.setLineDash([])
  }

  // Draw click points
  for (const pt of points) {
    ctx.beginPath()
    ctx.arc(pt.x * scaleX, pt.y * scaleY, 7, 0, Math.PI * 2)
    ctx.fillStyle = pt.label === 1 ? '#22c55e' : '#ef4444'
    ctx.fill()
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 2
    ctx.stroke()
  }
}

export function SegmentPanel({ image, onResult }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const [points, setPoints] = useState<ClickPoint[]>([])
  const [boxStart, setBoxStart] = useState<{ x: number; y: number } | null>(null)
  const [box, setBox] = useState<[number, number, number, number] | null>(null)
  const [drawingBox, setDrawingBox] = useState(false)
  const [pointMode, setPointMode] = useState<'fg' | 'bg' | 'box'>('fg')
  const [outputFormat, setOutputFormat] = useState<'masks' | 'polygons' | 'both'>('masks')
  const [multimask, setMultimask] = useState(true)
  const [result, setResult] = useState<SegmentResponse | null>(null)
  const [activeMask, setActiveMask] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [splitView, setSplitView] = useState(false)

  const redraw = useCallback(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return
    drawOverlay(canvas, img, points, box, result, activeMask)
  }, [points, box, result, activeMask])

  // Keep a loaded <img> for canvas drawing
  useEffect(() => {
    if (!image) {
      imgRef.current = null
      setResult(null)
      setPoints([])
      setBox(null)
      setActiveMask(null)
      return
    }
    const img = new Image()
    img.src = image.previewUrl
    img.onload = () => {
      imgRef.current = img
      redraw()
    }
  }, [image, redraw])

  useEffect(() => {
    redraw()
  }, [redraw])

  const toImageCoords = (e: React.MouseEvent<HTMLCanvasElement>): { x: number; y: number } => {
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    const scaleX = (imgRef.current?.naturalWidth ?? canvas.width) / canvas.width
    const scaleY = (imgRef.current?.naturalHeight ?? canvas.height) / canvas.height
    return {
      x: Math.round((e.clientX - rect.left) * scaleX),
      y: Math.round((e.clientY - rect.top) * scaleY),
    }
  }

  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!image) return
    if (pointMode === 'box') {
      const pos = toImageCoords(e)
      setBoxStart(pos)
      setDrawingBox(true)
    } else {
      const { x, y } = toImageCoords(e)
      setPoints((prev) => [...prev, { x, y, label: pointMode === 'fg' ? 1 : 0 }])
    }
  }

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drawingBox || !boxStart || !canvasRef.current || !imgRef.current) return
    const pos = toImageCoords(e)
    const newBox: [number, number, number, number] = [
      Math.min(boxStart.x, pos.x),
      Math.min(boxStart.y, pos.y),
      Math.max(boxStart.x, pos.x),
      Math.max(boxStart.y, pos.y),
    ]
    setBox(newBox)
  }

  const handleCanvasMouseUp = () => {
    setDrawingBox(false)
    setBoxStart(null)
  }

  const handleRun = async () => {
    if (!image) return
    setLoading(true)
    setError(null)
    setResult(null)
    setActiveMask(null)
    try {
      const res = await segment({
        image: image.file,
        pointCoords: points.length ? points.map((p) => [p.x, p.y]) : undefined,
        pointLabels: points.length ? points.map((p) => p.label) : undefined,
        box: box ?? undefined,
        multimaskOutput: multimask,
        outputFormat,
      })
      setResult(res)
      onResult({
        imageName: image.file.name,
        action: 'segment',
        summary: `${res.scores.length} mask${res.scores.length !== 1 ? 's' : ''} · ${res.processing_time_ms.toFixed(0)} ms`,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  const handleClearPrompts = () => {
    setPoints([])
    setBox(null)
    setResult(null)
    setActiveMask(null)
    setError(null)
  }

  // ── Export helpers ──────────────────────────────────────────────────────────

  const handleExportJson = () => {
    if (!result || !image) return
    downloadJson(result, `segment_${image.file.name.replace(/\.[^.]+$/, '')}.json`)
  }

  const handleExportCsv = () => {
    if (!result || !image) return
    const stem = image.file.name.replace(/\.[^.]+$/, '')
    const rows: (string | number)[][] = [['mask_index', 'score']]
    result.scores.forEach((score, i) => rows.push([i + 1, score.toFixed(4)]))
    downloadCsv(rows, `segment_scores_${stem}.csv`)
  }

  const handleExportPng = () => {
    const canvas = canvasRef.current
    if (!canvas || !image) return
    downloadCanvas(canvas, `segment_overlay_${image.file.name.replace(/\.[^.]+$/, '')}.png`)
  }

  const canvasWidth = 640
  const canvasHeight = image
    ? Math.round((image.height / image.width) * canvasWidth)
    : 360

  return (
    <section className="panel">
      <header className="panel__header">
        <h2 className="panel__title">Prompt Segmentation</h2>
        <div className="panel__actions">
          <button className="btn btn--sm" onClick={handleClearPrompts} disabled={loading}>
            Clear prompts
          </button>
          <button
            className="btn btn--sm btn--primary"
            onClick={() => void handleRun()}
            disabled={loading || !image}
          >
            {loading ? 'Running…' : 'Run /api/segment'}
          </button>
        </div>
      </header>

      {!image && (
        <p className="hint-msg">Load an image above to begin.</p>
      )}

      {error && <p className="error-msg">{error}</p>}

      {loading && (
        <div className="loading-bar" role="status" aria-label="Running segmentation">
          <span className="loading-bar__fill" />
        </div>
      )}

      {image && (
        <>
          {/* Prompt controls */}
          <div className="seg-controls">
            <div className="seg-controls__group">
              <span className="seg-controls__label">Click mode</span>
              <div className="btn-group">
                {(['fg', 'bg', 'box'] as const).map((m) => (
                  <button
                    key={m}
                    className={`btn btn--sm${pointMode === m ? ' btn--active' : ''}`}
                    onClick={() => setPointMode(m)}
                  >
                    {m === 'fg' ? '+ Foreground' : m === 'bg' ? '− Background' : '□ Box'}
                  </button>
                ))}
              </div>
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
            <div className="seg-controls__group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={multimask}
                  onChange={(e) => setMultimask(e.target.checked)}
                />
                Multi-mask
              </label>
            </div>
            <div className="seg-controls__group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={splitView}
                  onChange={(e) => setSplitView(e.target.checked)}
                />
                Side-by-side
              </label>
            </div>
          </div>

          {/* Canvas (and optional original side-by-side) */}
          <div className={splitView ? 'split-view' : undefined}>
            {splitView && (
              <div className="split-view__pane">
                <p className="split-view__label">Original</p>
                <div className="canvas-wrap">
                  <img
                    src={image.previewUrl}
                    alt="Original"
                    className="seg-canvas"
                    style={{ width: canvasWidth, height: canvasHeight, objectFit: 'contain' }}
                  />
                </div>
              </div>
            )}
            <div className={splitView ? 'split-view__pane' : undefined}>
              {splitView && <p className="split-view__label">Result</p>}
              <div className="canvas-wrap">
                <canvas
                  ref={canvasRef}
                  width={canvasWidth}
                  height={canvasHeight}
                  className="seg-canvas"
                  style={{ cursor: pointMode === 'box' ? 'crosshair' : 'cell' }}
                  onMouseDown={handleCanvasMouseDown}
                  onMouseMove={handleCanvasMouseMove}
                  onMouseUp={handleCanvasMouseUp}
                  onMouseLeave={handleCanvasMouseUp}
                />
              </div>
            </div>
          </div>

          {/* Prompt summary */}
          {(points.length > 0 || box) && (
            <div className="prompt-summary">
              {points.length > 0 && (
                <span>
                  {points.filter((p) => p.label === 1).length} fg ·{' '}
                  {points.filter((p) => p.label === 0).length} bg points
                </span>
              )}
              {box && (
                <span>
                  Box [{box.map((v) => Math.round(v)).join(', ')}]
                </span>
              )}
            </div>
          )}
        </>
      )}

      {/* Results */}
      {result && (
        <div className="seg-results">
          <div className="seg-results__header">
            <span className="seg-results__title">
              {result.scores.length} mask{result.scores.length !== 1 ? 's' : ''} · {result.processing_time_ms.toFixed(0)} ms
            </span>
            <span className="seg-results__hint">Click a mask to highlight</span>
          </div>
          <div className="mask-chips">
            {result.scores.map((score, i) => (
              <button
                key={i}
                className={`mask-chip${activeMask === i ? ' mask-chip--active' : ''}`}
                style={{ '--chip-color': MASK_COLORS[i % MASK_COLORS.length] } as React.CSSProperties}
                onClick={() => setActiveMask(activeMask === i ? null : i)}
              >
                <span className="mask-chip__swatch" />
                <span className="mask-chip__label">Mask {i + 1}</span>
                <span className="mask-chip__score">{(score * 100).toFixed(1)}%</span>
              </button>
            ))}
          </div>

          {/* Export bar */}
          <div className="export-bar">
            <span className="export-bar__label">Export</span>
            <button className="btn btn--sm" onClick={handleExportJson}>↓ JSON</button>
            <button className="btn btn--sm" onClick={handleExportCsv}>↓ CSV</button>
            <button className="btn btn--sm" onClick={handleExportPng}>↓ PNG overlay</button>
          </div>
        </div>
      )}
    </section>
  )
}

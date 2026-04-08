import { useState } from 'react'
import { analyze, extractPalette, transform } from '../api/client'
import type { AnalyzeResponse, ExtractPaletteResponse, ImageMeta } from '../types'
import type { TransformOp } from '../api/client'

interface Props {
  image: ImageMeta | null
}

// ---------------------------------------------------------------------------
// Analyze tab
// ---------------------------------------------------------------------------

function AnalyzeTab({ image }: Props) {
  const [numColors, setNumColors] = useState(5)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleRun = async () => {
    if (!image) return
    setLoading(true)
    setError(null)
    try {
      setResult(await analyze({ image: image.file, numColors }))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="seg-controls">
        <div className="seg-controls__group">
          <label className="seg-controls__label" htmlFor="analyze-k">
            Dominant colors: <strong>{numColors}</strong>
          </label>
          <input
            id="analyze-k"
            type="range"
            min={1}
            max={20}
            value={numColors}
            onChange={(e) => setNumColors(Number(e.target.value))}
            className="range-input"
          />
        </div>
        <button
          className="btn btn--sm btn--primary"
          onClick={() => void handleRun()}
          disabled={loading || !image}
        >
          {loading ? 'Analyzing…' : 'Run /api/analyze'}
        </button>
      </div>
      {error && <p className="error-msg">{error}</p>}

      {result && (
        <div className="cv-result">
          <div className="cv-result__meta">
            <dl className="cv-meta-dl">
              <dt>Dimensions</dt>
              <dd>{result.width} × {result.height} px</dd>
              <dt>Channels</dt>
              <dd>{result.channels}</dd>
              <dt>Format</dt>
              <dd>{result.format ?? '—'}</dd>
              <dt>Edge density</dt>
              <dd>{(result.edge_density * 100).toFixed(2)}%</dd>
            </dl>
          </div>

          <div className="cv-result__colors">
            <p className="cv-section-label">Dominant Colors</p>
            <div className="color-chips">
              {result.dominant_colors.map((c, i) => (
                <div key={i} className="color-chip">
                  <div className="color-chip__swatch" style={{ background: c.hex }} />
                  <span className="color-chip__hex">{c.hex}</span>
                  <span className="color-chip__freq">{(c.frequency * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>

          <div className="cv-result__hist">
            <p className="cv-section-label">Histogram Stats (R / G / B)</p>
            <table className="hist-table">
              <thead>
                <tr><th>Stat</th><th>R</th><th>G</th><th>B</th></tr>
              </thead>
              <tbody>
                {(['mean', 'std', 'min', 'max'] as const).map((stat) => (
                  <tr key={stat}>
                    <td>{stat}</td>
                    {result.histogram_stats[stat].map((v, i) => (
                      <td key={i}>{v.toFixed(1)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Transform tab
// ---------------------------------------------------------------------------

const OP_OPTIONS = ['resize', 'crop', 'rotate', 'blur', 'sharpen', 'edge-detect'] as const
type OpName = typeof OP_OPTIONS[number]

function emptyOp(name: OpName): TransformOp {
  switch (name) {
    case 'resize': return { op: 'resize', width: 256, height: 256 }
    case 'crop': return { op: 'crop', x: 0, y: 0, width: 128, height: 128 }
    case 'rotate': return { op: 'rotate', angle: 90 }
    case 'blur': return { op: 'blur', kernel_size: 5 }
    case 'sharpen': return { op: 'sharpen' }
    case 'edge-detect': return { op: 'edge-detect' }
  }
}

function OpRow({
  op,
  idx,
  onChange,
  onRemove,
}: {
  op: TransformOp
  idx: number
  onChange: (i: number, op: TransformOp) => void
  onRemove: (i: number) => void
}) {
  const changeField = (field: string, value: number) => {
    onChange(idx, { ...op, [field]: value } as TransformOp)
  }

  return (
    <div className="op-row">
      <select
        className="select-sm"
        value={op.op}
        onChange={(e) => onChange(idx, emptyOp(e.target.value as OpName))}
      >
        {OP_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>

      {op.op === 'resize' && (
        <>
          <input type="number" className="num-input" value={op.width} min={1} onChange={(e) => changeField('width', Number(e.target.value))} placeholder="w" />
          <input type="number" className="num-input" value={op.height} min={1} onChange={(e) => changeField('height', Number(e.target.value))} placeholder="h" />
        </>
      )}
      {op.op === 'crop' && (
        <>
          <input type="number" className="num-input" value={op.x} min={0} onChange={(e) => changeField('x', Number(e.target.value))} placeholder="x" />
          <input type="number" className="num-input" value={op.y} min={0} onChange={(e) => changeField('y', Number(e.target.value))} placeholder="y" />
          <input type="number" className="num-input" value={op.width} min={1} onChange={(e) => changeField('width', Number(e.target.value))} placeholder="w" />
          <input type="number" className="num-input" value={op.height} min={1} onChange={(e) => changeField('height', Number(e.target.value))} placeholder="h" />
        </>
      )}
      {op.op === 'rotate' && (
        <input type="number" className="num-input" value={op.angle} onChange={(e) => changeField('angle', Number(e.target.value))} placeholder="angle°" />
      )}
      {op.op === 'blur' && (
        <input type="number" className="num-input" value={op.kernel_size ?? 5} min={1} onChange={(e) => changeField('kernel_size', Number(e.target.value))} placeholder="kernel" />
      )}

      <button className="btn btn--sm btn--danger" aria-label="Remove operation" onClick={() => onRemove(idx)}>✕</button>
    </div>
  )
}

function TransformTab({ image }: Props) {
  const [ops, setOps] = useState<TransformOp[]>([])
  const [outputFormat, setOutputFormat] = useState<'PNG' | 'JPEG' | 'WEBP'>('PNG')
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const addOp = () => setOps((prev) => [...prev, emptyOp('resize')])

  const updateOp = (i: number, op: TransformOp) => {
    setOps((prev) => prev.map((o, idx) => (idx === i ? op : o)))
  }

  const removeOp = (i: number) => {
    setOps((prev) => prev.filter((_, idx) => idx !== i))
  }

  const handleRun = async () => {
    if (!image) return
    setLoading(true)
    setError(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(null)
    try {
      const blob = await transform({ image: image.file, operations: ops, outputFormat })
      setPreviewUrl(URL.createObjectURL(blob))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="transform-builder">
        {ops.length === 0 && (
          <p className="hint-msg">Add operations below, then run. An empty pipeline returns the original image re-encoded.</p>
        )}
        {ops.map((op, i) => (
          <OpRow key={i} op={op} idx={i} onChange={updateOp} onRemove={removeOp} />
        ))}
        <button className="btn btn--sm" onClick={addOp}>+ Add operation</button>
      </div>

      <div className="seg-controls" style={{ marginTop: '0.75rem' }}>
        <div className="seg-controls__group">
          <span className="seg-controls__label">Output format</span>
          <select
            className="select-sm"
            value={outputFormat}
            onChange={(e) => setOutputFormat(e.target.value as typeof outputFormat)}
          >
            <option>PNG</option>
            <option>JPEG</option>
            <option>WEBP</option>
          </select>
        </div>
        <button
          className="btn btn--sm btn--primary"
          onClick={() => void handleRun()}
          disabled={loading || !image}
        >
          {loading ? 'Transforming…' : 'Run /api/transform'}
        </button>
      </div>

      {error && <p className="error-msg">{error}</p>}

      {previewUrl && (
        <div className="transform-preview">
          <p className="cv-section-label">Result</p>
          <img src={previewUrl} alt="Transform result" className="transform-preview__img" />
          <a
            className="btn btn--sm"
            href={previewUrl}
            download={`transformed.${outputFormat.toLowerCase()}`}
            style={{ marginTop: '0.5rem', display: 'inline-flex' }}
          >
            ↓ Download
          </a>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Palette tab
// ---------------------------------------------------------------------------

function PaletteTab({ image }: Props) {
  const [numColors, setNumColors] = useState(6)
  const [kulrsFormat, setKulrsFormat] = useState(false)
  const [result, setResult] = useState<ExtractPaletteResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleRun = async () => {
    if (!image) return
    setLoading(true)
    setError(null)
    try {
      setResult(await extractPalette({ image: image.file, numColors, kulrsFormat }))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="seg-controls">
        <div className="seg-controls__group">
          <label className="seg-controls__label" htmlFor="palette-k">
            Colors: <strong>{numColors}</strong>
          </label>
          <input
            id="palette-k"
            type="range"
            min={1}
            max={32}
            value={numColors}
            onChange={(e) => setNumColors(Number(e.target.value))}
            className="range-input"
          />
        </div>
        <div className="seg-controls__group">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={kulrsFormat}
              onChange={(e) => setKulrsFormat(e.target.checked)}
            />
            Kulrs format
          </label>
        </div>
        <button
          className="btn btn--sm btn--primary"
          onClick={() => void handleRun()}
          disabled={loading || !image}
        >
          {loading ? 'Extracting…' : 'Run /api/extract-palette'}
        </button>
      </div>

      {error && <p className="error-msg">{error}</p>}

      {result && (
        <div className="cv-result">
          <p className="cv-section-label">Palette</p>
          <div className="palette-strip">
            {result.colors.map((c, i) => (
              <div key={i} className="palette-swatch" style={{ background: c.hex, flex: c.weight }}>
                <span className="palette-swatch__label">{c.hex}</span>
              </div>
            ))}
          </div>
          <div className="color-chips" style={{ marginTop: '0.75rem' }}>
            {result.colors.map((c, i) => (
              <div key={i} className="color-chip">
                <div className="color-chip__swatch" style={{ background: c.hex }} />
                <span className="color-chip__hex">{c.hex}</span>
                <span className="color-chip__freq">{(c.weight * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
          {result.kulrs && (
            <div style={{ marginTop: '0.75rem' }}>
              <p className="cv-section-label">Kulrs</p>
              <code className="kulrs-json">{JSON.stringify(result.kulrs.colors)}</code>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main panel with internal tabs
// ---------------------------------------------------------------------------

type CvTab = 'analyze' | 'transform' | 'palette'

export function CvToolsPanel({ image }: Props) {
  const [activeTab, setActiveTab] = useState<CvTab>('analyze')

  return (
    <section className="panel">
      <header className="panel__header">
        <h2 className="panel__title">OpenCV Tools</h2>
      </header>

      {!image && <p className="hint-msg">Load an image above to begin.</p>}

      <div className="inner-tabs">
        {(['analyze', 'transform', 'palette'] as CvTab[]).map((tab) => (
          <button
            key={tab}
            className={`inner-tab${activeTab === tab ? ' inner-tab--active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab === 'analyze' ? 'Analyze' : tab === 'transform' ? 'Transform' : 'Palette'}
          </button>
        ))}
      </div>

      <div className="inner-tab-content">
        {activeTab === 'analyze' && <AnalyzeTab image={image} />}
        {activeTab === 'transform' && <TransformTab image={image} />}
        {activeTab === 'palette' && <PaletteTab image={image} />}
      </div>
    </section>
  )
}

import { useEffect, useState, useCallback } from 'react'
import { fetchHealth, fetchInfo } from '../api/client'
import type { HealthResponse, InfoResponse } from '../types'

const POLL_INTERVAL_MS = 10_000

function mb(v: number | null): string {
  return v == null ? '—' : `${v.toFixed(0)} MB`
}

export function StatusPanel() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [info, setInfo] = useState<InfoResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [lastFetch, setLastFetch] = useState<Date | null>(null)

  const refresh = useCallback(async () => {
    try {
      const [h, i] = await Promise.all([fetchHealth(), fetchInfo()])
      setHealth(h)
      setInfo(i)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
    setLastFetch(new Date())
  }, [])

  useEffect(() => {
    void refresh()
    const timer = setInterval(() => void refresh(), POLL_INTERVAL_MS)
    return () => clearInterval(timer)
  }, [refresh])

  const readyClass = health?.ready ? 'badge badge--ready' : 'badge badge--warming'
  const readyLabel = health?.ready ? 'Ready' : 'Warming up'

  return (
    <section className="panel status-panel">
      <header className="panel__header">
        <h2 className="panel__title">Service Status</h2>
        <div className="panel__actions">
          {lastFetch && (
            <span className="status-panel__ts">
              Updated {lastFetch.toLocaleTimeString()}
            </span>
          )}
          <button className="btn btn--sm" onClick={() => void refresh()}>
            Refresh
          </button>
        </div>
      </header>

      {error && <p className="error-msg">{error}</p>}

      {health && (
        <div className="status-panel__grid">
          <div className="status-item">
            <span className="status-item__label">Status</span>
            <span className={readyClass}>{readyLabel}</span>
          </div>
          <div className="status-item">
            <span className="status-item__label">Model variant</span>
            <code className="code-badge">{health.model_variant}</code>
          </div>
          {info && (
            <>
              <div className="status-item">
                <span className="status-item__label">Device</span>
                <span>{info.device_name ?? info.device ?? '—'}</span>
              </div>
              <div className="status-item">
                <span className="status-item__label">VRAM total</span>
                <span>{mb(info.vram_total_mb)}</span>
              </div>
              <div className="status-item">
                <span className="status-item__label">VRAM reserved</span>
                <span>{mb(info.vram_reserved_mb)}</span>
              </div>
              <div className="status-item">
                <span className="status-item__label">VRAM allocated</span>
                <span>{mb(info.vram_allocated_mb)}</span>
              </div>
            </>
          )}
          {health.load_error && (
            <div className="status-item status-item--full">
              <span className="status-item__label">Load error</span>
              <span className="error-msg">{health.load_error}</span>
            </div>
          )}
        </div>
      )}
    </section>
  )
}

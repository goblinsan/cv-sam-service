import type { SessionEntry } from '../types'

interface Props {
  entries: SessionEntry[]
  onClear: () => void
}

const ACTION_LABELS: Record<string, string> = {
  'segment': 'Prompt Seg',
  'auto-segment': 'Auto Seg',
  'analyze': 'Analyze',
  'palette': 'Palette',
  'transform': 'Transform',
}

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  } catch {
    return iso
  }
}

export function SessionHistory({ entries, onClear }: Props) {
  if (entries.length === 0) return null

  return (
    <section className="panel history-panel">
      <header className="panel__header">
        <h2 className="panel__title">Session History</h2>
        <button className="btn btn--sm" onClick={onClear} aria-label="Clear session history">
          Clear
        </button>
      </header>
      <ul className="history-list" role="list">
        {entries.map((e) => (
          <li key={e.id} className="history-entry">
            <span className="history-entry__action">
              {ACTION_LABELS[e.action] ?? e.action}
            </span>
            <span className="history-entry__image" title={e.imageName}>
              {e.imageName}
            </span>
            <span className="history-entry__summary">{e.summary}</span>
            <time className="history-entry__time" dateTime={e.timestamp}>
              {formatTime(e.timestamp)}
            </time>
          </li>
        ))}
      </ul>
    </section>
  )
}

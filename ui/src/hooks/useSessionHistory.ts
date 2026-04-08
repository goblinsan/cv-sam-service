import { useCallback, useState } from 'react'
import type { ActionType, SessionEntry } from '../types'

const HISTORY_KEY = 'cv_sam_session_history'
const MAX_ENTRIES = 20

function loadHistory(): SessionEntry[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    return raw ? (JSON.parse(raw) as SessionEntry[]) : []
  } catch {
    return []
  }
}

function saveHistory(entries: SessionEntry[]): void {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(entries))
  } catch {
    // storage full or unavailable – fail silently
  }
}

export function useSessionHistory() {
  const [entries, setEntries] = useState<SessionEntry[]>(loadHistory)

  const addEntry = useCallback(
    (entry: { imageName: string; action: ActionType; summary: string }) => {
      const newEntry: SessionEntry = {
        ...entry,
        id: crypto.randomUUID(),
        timestamp: new Date().toISOString(),
      }
      setEntries((prev) => {
        const next = [newEntry, ...prev].slice(0, MAX_ENTRIES)
        saveHistory(next)
        return next
      })
    },
    [],
  )

  const clearHistory = useCallback(() => {
    setEntries([])
    localStorage.removeItem(HISTORY_KEY)
  }, [])

  return { entries, addEntry, clearHistory }
}

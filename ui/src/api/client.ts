import type { HealthResponse, InfoResponse } from '../types'

const BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    throw new Error(`GET ${path} → ${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

export function fetchHealth(): Promise<HealthResponse> {
  return get<HealthResponse>('/health')
}

export function fetchInfo(): Promise<InfoResponse> {
  return get<InfoResponse>('/info')
}

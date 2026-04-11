import type {
  AnalyzeResponse,
  AutoSegmentResponse,
  ExtractPaletteResponse,
  HealthResponse,
  InfoResponse,
  SegmentResponse,
} from '../types'

const BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    throw new Error(`GET ${path} → ${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

async function postForm<T>(path: string, body: FormData): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { method: 'POST', body })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`POST ${path} → ${res.status}: ${detail}`)
  }
  return res.json() as Promise<T>
}

async function postFormBinary(path: string, body: FormData): Promise<Blob> {
  const res = await fetch(`${BASE}${path}`, { method: 'POST', body })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`POST ${path} → ${res.status}: ${detail}`)
  }
  return res.blob()
}

export function fetchHealth(): Promise<HealthResponse> {
  return get<HealthResponse>('/health')
}

export function fetchInfo(): Promise<InfoResponse> {
  return get<InfoResponse>('/info')
}

export async function unloadModel(): Promise<void> {
  const res = await fetch(`${BASE}/model/unload`, { method: 'POST' })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`POST /model/unload → ${res.status}: ${detail}`)
  }
}

export interface SegmentParams {
  image: File;
  pointCoords?: [number, number][];
  pointLabels?: number[];
  box?: [number, number, number, number];
  multimaskOutput?: boolean;
  outputFormat?: 'masks' | 'polygons' | 'both';
}

export function segment(params: SegmentParams): Promise<SegmentResponse> {
  const form = new FormData()
  form.append('image', params.image)
  if (params.pointCoords) form.append('point_coords', JSON.stringify(params.pointCoords))
  if (params.pointLabels) form.append('point_labels', JSON.stringify(params.pointLabels))
  if (params.box) form.append('box', JSON.stringify(params.box))
  form.append('multimask_output', String(params.multimaskOutput ?? true))
  const fmt = params.outputFormat ?? 'masks'
  return postForm<SegmentResponse>(`/segment?output_format=${fmt}`, form)
}

export interface AutoSegmentParams {
  image: File;
  maxMasks?: number;
  outputFormat?: 'masks' | 'polygons' | 'both';
}

export function segmentAuto(params: AutoSegmentParams): Promise<AutoSegmentResponse> {
  const form = new FormData()
  form.append('image', params.image)
  const maxMasks = params.maxMasks ?? 50
  const fmt = params.outputFormat ?? 'masks'
  return postForm<AutoSegmentResponse>(`/segment/auto?max_masks=${maxMasks}&output_format=${fmt}`, form)
}

export interface AnalyzeParams {
  image: File;
  numColors?: number;
}

export function analyze(params: AnalyzeParams): Promise<AnalyzeResponse> {
  const form = new FormData()
  form.append('image', params.image)
  const k = params.numColors ?? 5
  return postForm<AnalyzeResponse>(`/analyze?num_colors=${k}`, form)
}

export type TransformOp =
  | { op: 'resize'; width: number; height: number }
  | { op: 'crop'; x: number; y: number; width: number; height: number }
  | { op: 'rotate'; angle: number }
  | { op: 'blur'; kernel_size?: number }
  | { op: 'sharpen' }
  | { op: 'edge-detect' }

export interface TransformParams {
  image: File;
  operations: TransformOp[];
  outputFormat?: 'PNG' | 'JPEG' | 'WEBP';
}

export function transform(params: TransformParams): Promise<Blob> {
  const form = new FormData()
  form.append('image', params.image)
  form.append('operations', JSON.stringify(params.operations))
  form.append('output_format', params.outputFormat ?? 'PNG')
  return postFormBinary('/transform', form)
}

export interface PaletteParams {
  image: File;
  numColors?: number;
  kulrsFormat?: boolean;
}

export function extractPalette(params: PaletteParams): Promise<ExtractPaletteResponse> {
  const form = new FormData()
  form.append('image', params.image)
  const k = params.numColors ?? 6
  const kulrs = params.kulrsFormat ?? false
  return postForm<ExtractPaletteResponse>(`/extract-palette?num_colors=${k}&kulrs_format=${kulrs}`, form)
}

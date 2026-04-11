export interface HealthResponse {
  status: string;
  ready: boolean;
  loading: boolean;
  model_variant: string;
  load_error: string | null;
}

export interface InfoResponse {
  ready: boolean;
  loading: boolean;
  model_variant: string;
  device: string | null;
  device_name: string | null;
  vram_total_mb: number | null;
  vram_reserved_mb: number | null;
  vram_allocated_mb: number | null;
  load_error: string | null;
}

export interface ImageMeta {
  file: File;
  width: number;
  height: number;
  format: string;
  previewUrl: string;
}

// ---------------------------------------------------------------------------
// Segment (prompted)
// ---------------------------------------------------------------------------

export interface SegmentResponse {
  masks: string[] | null;
  polygons: number[][][] | null;
  scores: number[];
  processing_time_ms: number;
}

// ---------------------------------------------------------------------------
// Auto-segment
// ---------------------------------------------------------------------------

export interface AutoSegment {
  mask: string | null;
  polygon: number[][] | null;
  score: number;
  stability_score: number;
  area: number;
  bbox: [number, number, number, number];
}

export interface AutoSegmentResponse {
  segments: AutoSegment[];
  count: number;
  processing_time_ms: number;
}

// ---------------------------------------------------------------------------
// Analyze
// ---------------------------------------------------------------------------

export interface ColorInfo {
  hex: string;
  rgb: [number, number, number];
  frequency: number;
}

export interface HistogramStats {
  mean: number[];
  std: number[];
  min: number[];
  max: number[];
}

export interface AnalyzeResponse {
  width: number;
  height: number;
  channels: number;
  format: string | null;
  dominant_colors: ColorInfo[];
  edge_density: number;
  histogram_stats: HistogramStats;
}

// ---------------------------------------------------------------------------
// Extract-palette
// ---------------------------------------------------------------------------

export interface PaletteColor {
  hex: string;
  rgb: [number, number, number];
  weight: number;
}

export interface ExtractPaletteResponse {
  colors: PaletteColor[];
  kulrs: { colors: string[] } | null;
}

// ---------------------------------------------------------------------------
// Session history
// ---------------------------------------------------------------------------

export type ActionType = 'segment' | 'auto-segment' | 'analyze' | 'palette' | 'transform'

export interface SessionEntry {
  id: string;
  imageName: string;
  action: ActionType;
  timestamp: string;
  summary: string;
}

export interface HealthResponse {
  status: string;
  ready: boolean;
  model_variant: string;
  load_error: string | null;
}

export interface InfoResponse {
  ready: boolean;
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

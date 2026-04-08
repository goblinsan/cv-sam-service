/** Trigger a browser download of a JSON object. */
export function downloadJson(data: unknown, filename: string): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  triggerDownload(blob, filename)
}

/** Trigger a browser download of a 2-D string/number array as CSV. */
export function downloadCsv(rows: (string | number)[][], filename: string): void {
  const csv = rows.map((row) => row.map(escapeCsvField).join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  triggerDownload(blob, filename)
}

/** Trigger a browser download of a canvas element as PNG. */
export function downloadCanvas(canvas: HTMLCanvasElement, filename: string): void {
  canvas.toBlob((blob) => {
    if (blob) triggerDownload(blob, filename)
  }, 'image/png')
}

/** Trigger a browser download from a base64-encoded PNG string. */
export function downloadBase64Png(base64: string, filename: string): void {
  const byteChars = atob(base64)
  const byteNums = new Uint8Array(byteChars.length)
  for (let i = 0; i < byteChars.length; i++) {
    byteNums[i] = byteChars.charCodeAt(i)
  }
  const blob = new Blob([byteNums], { type: 'image/png' })
  triggerDownload(blob, filename)
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

function escapeCsvField(field: string | number): string {
  const str = String(field)
  if (str.includes(',') || str.includes('"') || str.includes('\n')) {
    return `"${str.replace(/"/g, '""')}"`
  }
  return str
}

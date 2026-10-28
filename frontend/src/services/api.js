const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Thrown for any API failure so callers can distinguish "backend reachable
// but returned an error" (has .status) from "backend unreachable at all"
// (network/CORS failure, .status is null) and show an appropriate message.
export class ApiError extends Error {
  constructor(message, status = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function handle(res) {
  if (!res.ok) {
    let detail = res.statusText || `Request failed (${res.status})`
    try {
      const body = await res.json()
      detail = body.detail || detail
    } catch (_) {
      /* response wasn't JSON -- keep the status-text fallback */
    }
    throw new ApiError(detail, res.status)
  }
  return res.json()
}

async function request(path, options) {
  let res
  try {
    res = await fetch(`${BASE_URL}${path}`, options)
  } catch (networkErr) {
    // fetch() itself threw: backend unreachable, wrong URL, CORS blocked, offline, etc.
    throw new ApiError(
      `Could not reach the AutoDQ backend at ${BASE_URL}. Is it running? (${networkErr.message})`
    )
  }
  return handle(res)
}

export async function uploadDataset(file) {
  const formData = new FormData()
  formData.append('file', file)
  return request('/api/datasets/upload', { method: 'POST', body: formData })
}

export async function analyzeDataset(datasetId, { targetColumn, domain }) {
  return request(`/api/datasets/${datasetId}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target_column: targetColumn || null, domain: domain || 'general' }),
  })
}

export async function getDatasetInfo(datasetId) {
  return request(`/api/datasets/${datasetId}`)
}

export async function listDatasets() {
  return request('/api/datasets')
}

export function reportUrl(datasetId, format = 'html') {
  return `${BASE_URL}/api/datasets/${datasetId}/report?format=${format}`
}

export async function cleanDataset(datasetId) {
  return request(`/api/datasets/${datasetId}/clean`, { method: 'POST' })
}

export function downloadUrl(datasetId) {
  return `${BASE_URL}/api/datasets/${datasetId}/download`
}

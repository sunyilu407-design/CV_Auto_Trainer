const API_BASE = '/api'

interface ApiResponse<T> {
  code: number
  msg: string
  data: T | null
}

async function request<T>(
  path: string,
  options?: RequestInit & { timeout?: number }
): Promise<T> {
  const controller = new AbortController()
  const timeout = options?.timeout ?? 30000

  const timer = setTimeout(() => controller.abort(), timeout)

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...options,
      signal: controller.signal,
      headers: {
        ...(options?.body && { 'Content-Type': 'application/json' }),
        ...options?.headers,
      },
    })
    clearTimeout(timer)

    const json: ApiResponse<T> = await res.json()
    if (json.code !== 0) {
      throw new Error(json.msg || 'API Error')
    }
    return json.data as T
  } catch (e) {
    clearTimeout(timer)
    throw e
  }
}

// Task APIs
export interface Task {
  id: string
  name: string
  status: string
  created_at: string
  updated_at: string
}

export const taskApi = {
  list: () => request<Task[]>('/tasks'),
  get: (id: string) => request<Task>(`/tasks/${id}`),
  create: (name: string) =>
    request<Task>('/tasks', { method: 'POST', body: JSON.stringify({ name }) }),
  delete: (id: string) =>
    request<void>(`/tasks/${id}`, { method: 'DELETE' }),
}

// VLM APIs
export const vlmApi = {
  parse: (imagesBase64: string[], userText: string) =>
    request<{ classes: unknown[]; confidence: number }>('/vlm/parse', {
      method: 'POST',
      body: JSON.stringify({ images_base64: imagesBase64, user_text: userText }),
    }),
  updateResult: (taskId: string, classes: unknown[]) =>
    request<void>(`/vlm/result/${taskId}`, {
      method: 'PUT',
      body: JSON.stringify({ classes }),
    }),
  test: (params: {
    provider: string
    base_url: string
    api_key: string
    api_format?: string
    model?: string
  }) =>
    request<{ success: boolean; message: string }>('/vlm/test', {
      method: 'POST',
      body: JSON.stringify(params),
    }),
}

// Training APIs
export interface TrainStartPayload {
  task_id: string
  model: string
  epochs: number
  imgsz: number
  lr0: number
  patience: number
  conf: number
  iou: number
  export_formats: string[]
  train_mode: 'local' | 'cloud'
  gpu_type: string
  resume_last: boolean
}

export const trainingApi = {
  start: (payload: TrainStartPayload) =>
    request<{ instance_id?: string }>('/training/start', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  getStatus: (taskId: string) =>
    request<{
      state: string
      current_epoch: number
      total_epochs: number
      current_map: number
    }>(`/training/${taskId}/status`),
  cancel: (taskId: string) =>
    request<void>(`/training/${taskId}/cancel`, { method: 'POST' }),
}

// Files APIs
export const filesApi = {
  upload: (taskId: string, formData: FormData) =>
    request<{ path: string }>(`/files/upload?task_id=${taskId}`, {
      method: 'POST',
      body: formData,
    }),
  getArtifacts: (taskId: string) =>
    request<{ name: string; path: string; size: number }[]>(
      `/files/${taskId}/artifacts`
    ),
  downloadArtifact: (taskId: string, filename: string) =>
    `${API_BASE}/files/${taskId}/artifacts/${filename}`,
}

// Settings APIs
export const settingsApi = {
  get: () => request<UserSettings>('/settings'),
  update: (settings: unknown) =>
    request<void>('/settings', {
      method: 'PUT',
      body: JSON.stringify(settings),
    }),
}

export interface UserSettings {
  vlm_provider: string
  vlm_base_url: string
  vlm_api_key: string
  vlm_api_format: string
  vlm_model: string
  cloud_provider: string
  ssh_host: string
  ssh_port: number
  ssh_username: string
  ssh_password: string
  ssh_private_key_path: string
  remote_work_dir: string
  autodl_token: string
  default_model: string
  default_augment_strength: string
  default_delete_original: boolean
  default_gpu_type: string
  default_train_mode: string
}

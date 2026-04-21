import { useAuthStore } from '../store/authStore'

const API_BASE = '/api'

interface ApiResponse<T> {
  code: number
  msg: string
  data: T | null
}

interface ApiErrorResponse {
  code?: number
  msg?: string
  detail?: string
}

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

async function request<T>(
  path: string,
  options?: RequestInit & { timeout?: number }
): Promise<T> {
  const controller = new AbortController()
  const timeout = options?.timeout ?? 30000
  const isFormData = typeof FormData !== 'undefined' && options?.body instanceof FormData
  const headers = new Headers(options?.headers)

  if (options?.body && !isFormData && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }
  if (!headers.has('Authorization')) {
    const token = useAuthStore.getState().token
    if (token) {
      headers.set('Authorization', `Bearer ${token}`)
    }
  }

  const timer = setTimeout(() => controller.abort(), timeout)

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...options,
      signal: controller.signal,
      headers,
    })
    clearTimeout(timer)

    let json: ApiResponse<T> | ApiErrorResponse | T | null = null
    try {
      json = await res.json()
    } catch {
      json = null
    }

    const jsonRecord = isObjectRecord(json) ? json : null
    const code = typeof jsonRecord?.code === 'number' ? jsonRecord.code : undefined
    const message =
      (typeof jsonRecord?.msg === 'string' ? jsonRecord.msg : undefined) ||
      (typeof jsonRecord?.detail === 'string' ? jsonRecord.detail : undefined) ||
      `HTTP ${res.status}`

    if (!res.ok) {
      throw new Error(message)
    }

    if (code !== undefined) {
      if (code !== 0) {
        throw new Error(message)
      }
      return (json as ApiResponse<T>).data as T
    }

    return json as T
  } catch (e) {
    clearTimeout(timer)
    if (e instanceof DOMException && e.name === 'AbortError') {
      throw new Error('请求超时，请稍后重试')
    }
    throw e
  }
}

async function downloadBlob(
  path: string,
  options?: RequestInit & { timeout?: number }
): Promise<Blob> {
  const controller = new AbortController()
  const timeout = options?.timeout ?? 30000
  const headers = new Headers(options?.headers)

  if (!headers.has('Authorization')) {
    const token = useAuthStore.getState().token
    if (token) {
      headers.set('Authorization', `Bearer ${token}`)
    }
  }

  const timer = setTimeout(() => controller.abort(), timeout)

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...options,
      signal: controller.signal,
      headers,
    })
    clearTimeout(timer)

    if (!res.ok) {
      let message = `HTTP ${res.status}`
      try {
        const json = (await res.json()) as ApiErrorResponse
        message = json.detail || json.msg || message
      } catch {
        // ignore
      }
      throw new Error(message)
    }

    return await res.blob()
  } catch (e) {
    clearTimeout(timer)
    if (e instanceof DOMException && e.name === 'AbortError') {
      throw new Error('请求超时，请稍后重试')
    }
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
  clone: (id: string, options?: {
    new_name?: string
    include_plan?: boolean
    include_train_config?: boolean
    include_augment_config?: boolean
    as_template?: boolean
  }) =>
    request<Task>(`/tasks/${id}/clone`, {
      method: 'POST',
      body: JSON.stringify(options ?? {}),
    }),
}

export interface AlgorithmPlanTarget {
  class_name: string
  prompt: string
  role: string
  requires_training: boolean
}

export interface AlgorithmPlanRegion {
  region_id: string
  name: string
  source: string
  required: boolean
}

export interface AlgorithmPlanTemporalConstraint {
  constraint_id: string
  type: string
  duration_seconds: number
}

export interface AlgorithmPlanEvent {
  event_code: string
  name: string
  trigger: {
    target_class: string
    region_id: string
    temporal_constraint_id: string
  }
}

export interface NegotiationSummary {
  scenario_label: string
  objects: string[]
  regions: string[]
  duration_seconds: number | null
  events: string[]
}

export interface CapabilityDraft {
  capability_id: string
  label: string
  trainable: boolean
  kind: 'detection' | 'classification' | 'tracking' | 'rule'
}

export interface ModelPipelineStep {
  step_id: string
  role: 'primary_detector' | 'secondary_detector' | 'classifier' | 'feature_matcher' | 'tracker' | 'rule_engine'
  recommended_model_id: string
  alternative_model_ids?: string[]
  reason_zh: string
  requires_training: boolean
  training_priority?: number
  estimated_training_hours?: number
  input_size?: number
  epochs?: number
  reuse_cache_id?: string
  reuse_weight_path?: string
  reuse_info_zh?: string
}

export interface TrainingStrategy {
  total_models_to_train: number
  estimated_total_hours: number
  train_mode_recommendation: 'local' | 'cloud_ssh' | 'cloud_autodl'
  train_mode_reason_zh?: string
  special_requirements?: string[]
}

export interface AlgorithmPlanDraft {
  summary: string
  summary_zh?: string
  scenario_type: string
  difficulty_level?: string
  negotiation_summary: NegotiationSummary
  capabilities: CapabilityDraft[]
  runtime_modes: string[]
  targets: AlgorithmPlanTarget[]
  regions: AlgorithmPlanRegion[]
  temporal_constraints: AlgorithmPlanTemporalConstraint[]
  events: AlgorithmPlanEvent[]
  model_pipeline?: ModelPipelineStep[]
  training_strategy?: TrainingStrategy
  revision_history?: Array<{ role: 'user' | 'assistant' | 'system'; content: string }>
  revision_snapshots?: Array<{ version: number; summary_zh: string; timestamp: number; plan: Record<string, unknown> }>
  training_requirements: {
    detector_training_required: boolean
    tracking_required: boolean
    rule_engine_required: boolean
  }
  confidence: number
}

export interface PretrainedModelInfo {
  model_id: string
  family: string
  variant: string
  display_name: string
  display_name_zh: string
  task_types: string[]
  params_m: number
  map50_coco: number | null
  fps_gpu: number | null
  fps_cpu: number | null
  min_device_tier: string
  recommended_device_tiers: string[]
  description_zh: string
  strengths: string[]
  weaknesses: string[]
  use_cases: string[]
  export_formats: string[]
}

export interface CachedModelInfo {
  cache_id: string
  source_model_id: string
  classes: string[]
  scenario_type: string
  map50: number | null
  trained_at: number
  reuse_count: number
}

export interface VideoValidationResult {
  validation_passed: boolean | null
  confidence: number
  analysis_zh: string
  suggestions_zh: string[]
  frame_count_analyzed: number
}

export interface RuntimeCapability {
  local_training_available: boolean
  preferred_device: 'cpu' | 'cuda' | 'mps'
  available_export_formats: ('onnx' | 'engine' | 'coreml' | 'openvino')[]
  supports_cloud_training: boolean
}

export interface TrainingRecommendation {
  recommended_model: string
  train_mode: 'local' | 'cloud'
  export_formats: ('onnx' | 'engine' | 'coreml' | 'openvino')[]
  requires_detector_training: boolean
  recommended_config: {
    model: string
    train_mode: 'local' | 'cloud'
    export_formats: ('onnx' | 'engine' | 'coreml' | 'openvino')[]
    imgsz: number
    epochs: number
    lr0: number
    patience: number
    conf: number
    iou: number
    gpu_type: string
  }
  reason_summary: string
  source_map: Record<string, string>
}

export interface AlgorithmPlanRecord {
  task_id: string
  status: string
  algorithm_plan: AlgorithmPlanDraft
  pipeline_config?: {
    version: string
    metadata: {
      summary: string
      scenario_type: string
      confidence?: number | null
    }
    inputs: {
      runtime_modes: string[]
    }
    detectors: Array<{
      detector_id: string
      detector_type: string
      target_classes: string[]
    }>
    trackers: Array<{
      tracker_id: string
      tracker_type: string
      source_detector_id: string
    }>
    regions: AlgorithmPlanRegion[]
    temporal_windows: AlgorithmPlanTemporalConstraint[]
    rules: Array<{
      rule_id: string
      rule_type: string
      event_code: string
      target_class: string
      region_id: string
      duration_seconds: number
    }>
    outputs: Array<{
      output_id: string
      type: string
      event_codes: string[]
    }>
    packaging: {
      format: string
      entrypoint: string
      config_path: string
    }
    training_recommendation: TrainingRecommendation
  }
}

export interface AlgorithmPlanNegotiationRecord {
  task_id: string
  negotiation_summary: NegotiationSummary
  offline_evaluation?: {
    status?: string
    clips?: Array<Record<string, unknown>>
  } | null
}

export interface VLMParseSuccess {
  status: 'success'
  message: string
  retryable: boolean
  classes: unknown[]
  raw_vlm_response: string
  confidence?: number | null
}

export interface VLMParseFailure {
  status: 'failed'
  message: string
  retryable: boolean
  classes: unknown[]
  raw_vlm_response: string
  confidence?: number | null
}

export type VLMParseResult = VLMParseSuccess | VLMParseFailure

// VLM APIs
export const vlmApi = {
  parse: (
    imagesBase64: string[],
    userText: string,
    sampleBoxes: Array<{ x: number; y: number; width: number; height: number }> = []
  ) =>
    request<VLMParseResult>('/vlm/parse', {
      method: 'POST',
      body: JSON.stringify({
        images_base64: imagesBase64,
        user_text: userText,
        sample_boxes: sampleBoxes,
      }),
      timeout: 210000,
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

export const algorithmApi = {
  generatePlan: (payload: {
    task_id: string
    user_description: string
    vlm_result: { classes: unknown[] } | null
    runtime_capability?: RuntimeCapability
    images_base64?: string[]
    gpu_type?: string
    platform?: string
    device_description?: string
    image_count?: number
    use_vlm_planner?: boolean
  }) =>
    request<AlgorithmPlanRecord>('/algorithm/plan', {
      method: 'POST',
      body: JSON.stringify(payload),
      timeout: 120000,
    }),
  getPlan: (taskId: string) =>
    request<AlgorithmPlanRecord>(`/algorithm/plan/${taskId}`),
  negotiatePlan: (
    taskId: string,
    payload: {
      negotiation_summary: NegotiationSummary
      offline_evaluation?: {
        status?: string
        clips?: Array<Record<string, unknown>>
      } | null
    }
  ) =>
    request<AlgorithmPlanNegotiationRecord>(`/algorithm/plan/${taskId}/negotiate`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  revisePlan: (
    taskId: string,
    payload: {
      user_feedback: string
      runtime_capability?: RuntimeCapability
      gpu_type?: string
      platform?: string
      device_description?: string
    }
  ) =>
    request<AlgorithmPlanRecord>(`/algorithm/plan/${taskId}/revise`, {
      method: 'POST',
      body: JSON.stringify(payload),
      timeout: 120000,
    }),
  rollbackPlan: (taskId: string, version: number) =>
    request<AlgorithmPlanRecord>(`/algorithm/plan/${taskId}/rollback`, {
      method: 'POST',
      body: JSON.stringify({ version }),
    }),
  confirmPlan: (
    taskId: string,
    payload?: {
      region_overrides?: Array<{
        region_id: string
        name: string
        bbox_xywhn: [number, number, number, number]
      }>
      runtime_capability?: RuntimeCapability
    }
  ) =>
    request<AlgorithmPlanRecord>(`/algorithm/plan/${taskId}/confirm`, {
      method: 'POST',
      body: JSON.stringify(payload ?? {}),
    }),
  exportPackage: (taskId: string) =>
    request<{
      bundle_dir: string
      pipeline_path: string
      manifest_path: string
      readme_path: string
      entrypoint_path: string
    }>(`/algorithm/package/${taskId}`, {
      method: 'POST',
    }),
  preview: (
    taskId: string,
    payload: {
      region_overrides?: Array<{
        region_id: string
        name: string
        bbox_xywhn: [number, number, number, number]
      }>
      sample_boxes: Array<{
        bbox_xywhn: [number, number, number, number]
        class_name: string
        conf?: number
      }>
    }
  ) =>
    request<{
      pipeline_config: AlgorithmPlanRecord['pipeline_config']
      track_states: Array<{
        track_id: string
        class_name: string
        present_duration_seconds: number
        bbox_xywhn: [number, number, number, number]
      }>
      events: Array<{
        event_code: string
        rule_id: string
        track_id: string
        class_name: string
        region_id: string
        duration_seconds: number
      }>
    }>(`/algorithm/preview/${taskId}`, {
      method: 'POST',
      body: JSON.stringify(payload),
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

export interface TrainingReport {
  overall_assessment_zh: string
  convergence_zh: string
  class_performance_zh: string
  improvement_suggestions_zh: string[]
  score: number | null
  charts_analyzed: string[]
}

export interface TrainingEstimateStep {
  step_id: string
  model_id: string
  role: string
  source: 'train' | 'reuse'
  duration_min: number
  cost_cny: number
}

export interface TrainingEstimate {
  gpu_type: string
  hourly_rate_cny: number
  total_images: number
  total_duration_min: number
  total_cost_cny: number
  steps: TrainingEstimateStep[]
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
      done: boolean
      error?: string | null
      artifact_paths?: Record<string, string>
      autodl_recovery?: AutoDLRecoveryInfo | null
      current_model_index?: number | null
      total_models?: number | null
      current_step_id?: string | null
      current_model_id?: string | null
      current_model_source?: 'trained' | 'reuse' | null
      multi_model_artifacts?: Record<string, {
        source: 'trained' | 'reuse' | 'failed'
        model_id: string
        role: string
        cache_id?: string
        weight_path?: string
        artifacts?: Record<string, string>
      }> | null
    }>(`/training/${taskId}/status`),
  getRecovery: (taskId: string) =>
    request<{
      recovery_info: AutoDLRecoveryInfo
      instructions_zh: AutoDLRecoveryStep[]
    }>(`/training/${taskId}/recovery`),
  cancel: (taskId: string) =>
    request<void>(`/training/${taskId}/cancel`, { method: 'POST' }),
  getReport: (taskId: string) =>
    request<TrainingReport>(`/training/${taskId}/report`, { timeout: 120000 }),
  estimate: (params: { task_id: string; model: string; epochs: number; imgsz: number; gpu_type: string }) =>
    request<TrainingEstimate>('/training/estimate', {
      method: 'POST',
      body: JSON.stringify(params),
    }),
}

export interface AutoDLRecoveryInfo {
  instance_id: string
  error_msg?: string
  train_command?: string
  data_yaml_path?: string
  project_dir?: string
  weights_path?: string
  ssh_host?: string
  ssh_port?: number
  ssh_username?: string
  ssh_password_masked?: string
  autodl_console_url?: string
  ssh_retrieval_failed?: boolean
}

export interface AutoDLRecoveryStep {
  step: number
  title: string
  description: string
  action: string
  password?: string
}

// Files APIs
export const filesApi = {
  upload: (taskId: string, formData: FormData, subdir?: string) =>
    request<{ path: string }>(`/files/upload?task_id=${taskId}${subdir ? `&subdir=${encodeURIComponent(subdir)}` : ''}`, {
      method: 'POST',
      body: formData,
    }),
  uploadVideo: (taskId: string, formData: FormData, purpose: string = 'training') =>
    request<{
      video_path: string
      video_info: { fps: number; total_frames: number; width: number; height: number; duration_seconds: number }
      frame_count?: number
      frames_dir?: string
      frames_base64?: string[]
    }>(`/files/upload-video?task_id=${taskId}&purpose=${purpose}`, {
      method: 'POST',
      body: formData,
      timeout: 120000,
    }),
  uploadVideoWithProgress: (
    taskId: string,
    formData: FormData,
    purpose: string = 'training',
    onProgress?: (pct: number) => void,
  ): Promise<{
    video_path: string
    video_info: { fps: number; total_frames: number; width: number; height: number; duration_seconds: number }
    frame_count?: number
    frames_dir?: string
    frames_base64?: string[]
  }> => {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${API_BASE}/files/upload-video?task_id=${taskId}&purpose=${purpose}`)
      const token = useAuthStore.getState().token
      if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`)
      xhr.upload.onprogress = (e) => {
        if (onProgress && e.lengthComputable) {
          onProgress(Math.round((e.loaded / e.total) * 100))
        }
      }
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const json = JSON.parse(xhr.responseText)
            if (typeof json === 'object' && json !== null && 'code' in json) {
              if (json.code === 0) return resolve(json.data)
              return reject(new Error(json.msg || 'upload failed'))
            }
            resolve(json)
          } catch (e) {
            reject(e)
          }
        } else {
          reject(new Error(`HTTP ${xhr.status}`))
        }
      }
      xhr.onerror = () => reject(new Error('网络错误'))
      xhr.ontimeout = () => reject(new Error('上传超时'))
      xhr.timeout = 300000
      xhr.send(formData)
    })
  },
  getArtifacts: (taskId: string) =>
    request<{ name: string; path: string; size: number }[]>(
      `/files/${taskId}/artifacts`
    ),
  downloadArtifact: (taskId: string, filename: string) =>
    downloadBlob(`/files/${taskId}/artifacts/${filename}`, {
      headers: {
        Accept: 'application/octet-stream',
      },
    }),
}

// Model Registry APIs
export const modelRegistryApi = {
  listModels: (params?: { device_tier?: string; task_type?: string; family?: string }) => {
    const qs = new URLSearchParams()
    if (params?.device_tier) qs.set('device_tier', params.device_tier)
    if (params?.task_type) qs.set('task_type', params.task_type)
    if (params?.family) qs.set('family', params.family)
    const query = qs.toString()
    return request<{ models: PretrainedModelInfo[]; total: number }>(
      `/algorithm/models${query ? `?${query}` : ''}`
    )
  },
  listCachedModels: () =>
    request<{ cached_models: CachedModelInfo[]; total: number }>('/algorithm/models/cached'),
  detectDeviceTier: (gpu_type?: string, platform?: string) => {
    const qs = new URLSearchParams()
    if (gpu_type) qs.set('gpu_type', gpu_type)
    if (platform) qs.set('platform', platform)
    return request<{ device_tier: string }>(`/algorithm/device-tier?${qs.toString()}`)
  },
}

// Video Validation API
export const videoValidationApi = {
  validate: (taskId: string, formData: FormData) =>
    request<VideoValidationResult>(`/algorithm/validate-video/${taskId}`, {
      method: 'POST',
      body: formData,
      timeout: 180000,
    }),
  validateWithProgress: (
    taskId: string,
    formData: FormData,
    onProgress?: (pct: number) => void,
  ): Promise<VideoValidationResult> => {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${API_BASE}/algorithm/validate-video/${taskId}`)
      const token = useAuthStore.getState().token
      if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`)
      xhr.upload.onprogress = (e) => {
        if (onProgress && e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100))
      }
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const json = JSON.parse(xhr.responseText)
            if (typeof json === 'object' && json !== null && 'code' in json) {
              if (json.code === 0) return resolve(json.data)
              return reject(new Error(json.msg || 'validate failed'))
            }
            resolve(json)
          } catch (e) { reject(e) }
        } else {
          reject(new Error(`HTTP ${xhr.status}`))
        }
      }
      xhr.onerror = () => reject(new Error('网络错误'))
      xhr.ontimeout = () => reject(new Error('验证超时'))
      xhr.timeout = 300000
      xhr.send(formData)
    })
  },
}

// Cached Models API
export interface CachedModelEntry {
  cache_id: string
  source_model_id: string
  task_id: string
  classes: string[]
  class_count: number
  scenario_type: string
  map50: number | null
  map50_95: number | null
  weight_path: string
  trained_at: number
  image_count: number
  epochs_completed: number
  reuse_count: number
  tags: string[]
  weight_exists: boolean
  weight_size_mb: number | null
}

export const modelsApi = {
  listCache: () => request<CachedModelEntry[]>('/models/cache'),
  deleteCache: (cacheId: string, deleteWeight = true) =>
    request<{ deleted: boolean; cache_id: string; deleted_weight_file: boolean }>(
      `/models/cache/${encodeURIComponent(cacheId)}?delete_weight_file=${deleteWeight}`,
      { method: 'DELETE' },
    ),
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

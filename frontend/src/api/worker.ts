const WORKER_HOST =
  (import.meta.env.VITE_WORKER_HOST as string | undefined)?.trim() ||
  `${window.location.hostname}:7860`

export const WORKER_HTTP_BASE =
  (import.meta.env.VITE_WORKER_HTTP_URL as string | undefined)?.trim() ||
  `${window.location.protocol === 'https:' ? 'https' : 'http'}://${WORKER_HOST}`

const WS_BASE =
  (import.meta.env.VITE_WORKER_WS_URL as string | undefined)?.trim() ||
  `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${WORKER_HOST}/ws`

export type WorkerMessageType =
  | 'progress'
  | 'stage_complete'
  | 'gpu_info'
  | 'error'
  | 'training_progress'
  | 'training_complete'
  | 'training_error'
  | 'pong'
  | 'cancel_ack'

export interface WorkerMessage {
  type: WorkerMessageType
  [key: string]: unknown
}

export interface ModelPathSummary {
  path: string
  exists: boolean
  size_bytes: number
  file_count: number
}

export interface ModelPrepState {
  running: boolean
  include_moondream: boolean
  include_locate_anything: boolean
  include_eagle_vqa: boolean
  steps: Record<string, { status: string; message: string }>
  error: string | null
  status: {
    yolo_world: {
      model: string
      selected_path: string
      worker_path: ModelPathSummary
      cwd_path: ModelPathSummary
      installed: boolean
    }
    clip: {
      model: string
      cache: ModelPathSummary
      installed: boolean
    }
    moondream: {
      model: string
      cache: ModelPathSummary
      installed: boolean
      incomplete_files: string[]
      complete_snapshots?: string[]
      snapshot_count?: number
    }
    locate_anything?: {
      model: string
      cache: ModelPathSummary
      installed: boolean
      size_bytes: number
    }
    eagle_vqa?: {
      model: string
      cache: ModelPathSummary
      installed: boolean
      size_bytes: number
    }
  } | null
}

export async function fetchModelStatus() {
  const res = await fetch(`${WORKER_HTTP_BASE}/model-status`)
  if (!res.ok) throw new Error('读取模型状态失败')
  return res.json() as Promise<ModelPrepState>
}

export async function prepareModels(options: {
  includeMoondream?: boolean
  includeLocateAnything?: boolean
  includeEagleVqa?: boolean
} = {}) {
  const {
    includeMoondream = false,
    includeLocateAnything = false,
    includeEagleVqa = false,
  } = options

  const res = await fetch(`${WORKER_HTTP_BASE}/prepare-models`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      include_moondream: includeMoondream,
      include_locate_anything: includeLocateAnything,
      include_eagle_vqa: includeEagleVqa,
    }),
  })
  if (!res.ok) throw new Error('启动模型准备失败')
  return res.json() as Promise<ModelPrepState>
}

export interface DetectionPreviewClass {
  class_name: string
  prompt: string
  prompt_aliases?: string[]
  negative_prompt?: string
  color_hint?: string | null
}

export interface DetectionPreviewResult {
  total_images: number
  raw_box_count: number
  accepted_box_count?: number
  candidate_box_count?: number
  diagnostic_conf_threshold?: number
  accepted_conf_threshold?: number
  imgsz?: number
  prompts_used?: string[]
  suggestions?: string[]
  message: string
  results: Array<{
    image_name: string
    image_base64: string
    detections: Array<{
      class_name: string
      prompt: string
      confidence: number
      accepted?: boolean
      bbox_xywhn: number[]
    }>
  }>
}

export async function previewYoloWorldDetection(payload: {
  image_dir: string
  classes: DetectionPreviewClass[]
  max_images?: number
  conf_threshold?: number
  diagnostic_conf_threshold?: number
  iou_threshold?: number
  imgsz?: number
}) {
  const res = await fetch(`${WORKER_HTTP_BASE}/detection-preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error('YOLO-World 预览失败')
  return res.json() as Promise<DetectionPreviewResult>
}

type MessageHandler = (msg: WorkerMessage) => void

class WorkerClient {
  private ws: WebSocket | null = null
  private handlers: Set<MessageHandler> = new Set()
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private pingTimer: ReturnType<typeof setInterval> | null = null
  private pendingMessages: unknown[] = []
  private lastPong = 0
  private url: string

  constructor(url = WS_BASE) {
    this.url = url
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return

    this.ws = new WebSocket(this.url)

    this.ws.onopen = () => {
      this.lastPong = Date.now()
      this.startPing()
      this.pendingMessages.forEach((message) => {
        this.ws?.send(JSON.stringify(message))
      })
      this.pendingMessages = []
    }

    this.ws.onmessage = (event) => {
      try {
        const msg: WorkerMessage = JSON.parse(event.data)
        this.lastPong = Date.now()
        if (msg.type === 'pong') {
          return
        }
        this.handlers.forEach((h) => h(msg))
      } catch {
        // ignore parse errors
      }
    }

    this.ws.onclose = () => {
      this.stopPing()
      this.scheduleReconnect()
    }

    this.ws.onerror = () => {
      this.ws?.close()
    }
  }

  private startPing() {
    this.pingTimer = setInterval(() => {
      if (Date.now() - this.lastPong > 30000) {
        // 30s no pong, reconnect
        this.ws?.close()
        return
      }
      this.send({ type: 'ping' })
    }, 15000)
  }

  private stopPing() {
    if (this.pingTimer) {
      clearInterval(this.pingTimer)
      this.pingTimer = null
    }
  }

  private scheduleReconnect() {
    if (this.reconnectTimer) return
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null
      this.connect()
    }, 3000)
  }

  send(data: unknown) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
      return
    }
    this.pendingMessages.push(data)
  }

  onMessage(handler: MessageHandler) {
    this.handlers.add(handler)
    return () => this.handlers.delete(handler)
  }

  disconnect() {
    this.stopPing()
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    this.ws?.close()
    this.ws = null
  }

  // Commands
  startDetection(payload: {
    image_dir: string
    classes: Array<{
      class_name: string
      prompt: string
      prompt_aliases?: string[]
      negative_prompt?: string
      color_hint?: string | null
    }>
    output_raw_dir: string
    output_label_dir: string
    output_image_dir: string
    conf_threshold?: number
    iou_threshold?: number
    imgsz?: number
    batch_size?: number
    qa_threshold?: number
    use_existing_labels?: boolean
    skip_quality_check?: boolean
  }) {
    this.send({ type: 'start_detection', payload })
  }

  startAugmentation(payload: {
    src_image_dir: string
    src_label_dir: string
    output_image_dir: string
    output_label_dir: string
    target_count: number
    strength?: 'light' | 'medium' | 'heavy'
    enabled?: {
      geometric?: boolean
      color?: boolean
      noise?: boolean
      weather?: boolean
      occlusion?: boolean
    }
    delete_original?: boolean
    min_visibility?: number
    max_per_image?: number
  }) {
    this.send({ type: 'start_augmentation', payload })
  }

  startLocalTraining(payload: {
    dataset_dir: string
    /** 增量训练时使用合并后的 data.yaml 路径 */
    data_yaml?: string
    train_config: Record<string, unknown>
  }) {
    this.send({ type: 'start_local_training', payload })
  }

  cancel() {
    this.send({ type: 'cancel' })
  }
}

export const workerClient = new WorkerClient()

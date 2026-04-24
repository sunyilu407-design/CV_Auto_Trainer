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
        if (msg.type === 'pong') {
          this.lastPong = Date.now()
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
      negative_prompt?: string
      color_hint?: string | null
    }>
    output_raw_dir: string
    output_label_dir: string
    output_image_dir: string
    conf_threshold?: number
    iou_threshold?: number
    batch_size?: number
    qa_threshold?: number
    use_existing_labels?: boolean
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
  }) {
    this.send({ type: 'start_augmentation', payload })
  }

  startLocalTraining(payload: {
    dataset_dir: string
    train_config: Record<string, unknown>
  }) {
    this.send({ type: 'start_local_training', payload })
  }

  cancel() {
    this.send({ type: 'cancel' })
  }
}

export const workerClient = new WorkerClient()

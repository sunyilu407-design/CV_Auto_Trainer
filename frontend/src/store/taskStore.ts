import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type {
  AlgorithmPlanDraft as ApiAlgorithmPlanDraft,
  AlgorithmPlanEvent as ApiAlgorithmPlanEvent,
  AlgorithmPlanRegion as ApiAlgorithmPlanRegion,
  AlgorithmPlanTarget as ApiAlgorithmPlanTarget,
  AlgorithmPlanTemporalConstraint as ApiAlgorithmPlanTemporalConstraint,
  TrainingRecommendation,
} from '../api/backend'

export type Stage =
  | 'upload'
  | 'intent_confirm'
  | 'algorithm_plan'
  | 'environment'
  | 'manual_annotation'
  | 'seed_training'
  | 'review_auto_labels'
  | 'labeling'
  | 'augment'
  | 'review'
  | 'offline_validation'
  | 'train_config'
  | 'training'
  | 'video_inference'
  | 'delivery'

export type AlgorithmPlanTarget = ApiAlgorithmPlanTarget
export type AlgorithmPlanRegion = ApiAlgorithmPlanRegion
export type AlgorithmPlanTemporalConstraint = ApiAlgorithmPlanTemporalConstraint
export type AlgorithmPlanEvent = ApiAlgorithmPlanEvent
export type AlgorithmPlanDraft = ApiAlgorithmPlanDraft

export interface StoredAlgorithmPlan {
  task_id: string
  status: string
  algorithm_plan: AlgorithmPlanDraft
  offline_evaluation?: {
    validation_passed?: boolean
    confidence?: number
    analysis_zh?: string
    suggestions_zh?: string[]
  }
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

export type DeviceProfileId =
  | 'embedded_low'
  | 'embedded_high'
  | 'laptop_cpu'
  | 'desktop_gpu'
  | 'apple_silicon'
  | 'cloud_server'
  | 'auto'

export interface DeviceProfile {
  id: DeviceProfileId
  label: string
  description: string
  gpuType: string
  platform: string
  deviceTier: string
}

export const DEVICE_PROFILES: Record<DeviceProfileId, DeviceProfile> = {
  auto: {
    id: 'auto',
    label: '让系统推荐',
    description: '我不确定自己的设备情况，系统自动判断',
    gpuType: '',
    platform: '',
    deviceTier: 'desktop_cpu',
  },
  embedded_low: {
    id: 'embedded_low',
    label: '小型嵌入式设备',
    description: '树莓派 / Jetson Nano 等算力受限设备',
    gpuType: 'Jetson Nano',
    platform: 'linux',
    deviceTier: 'edge_low',
  },
  embedded_high: {
    id: 'embedded_high',
    label: '工业级嵌入式',
    description: 'Jetson Xavier / Orin 等中高端边缘设备',
    gpuType: 'Jetson Orin',
    platform: 'linux',
    deviceTier: 'edge_high',
  },
  laptop_cpu: {
    id: 'laptop_cpu',
    label: '普通电脑 (无独立显卡)',
    description: 'Intel / AMD CPU 的笔记本或台式机',
    gpuType: 'CPU',
    platform: 'linux',
    deviceTier: 'desktop_cpu',
  },
  desktop_gpu: {
    id: 'desktop_gpu',
    label: '带显卡的电脑',
    description: 'NVIDIA 显卡 (RTX 3060 / 4090 等)',
    gpuType: 'RTX 4090',
    platform: 'linux',
    deviceTier: 'desktop_gpu',
  },
  apple_silicon: {
    id: 'apple_silicon',
    label: '苹果电脑',
    description: 'MacBook / iMac M1/M2/M3 系列芯片',
    gpuType: 'Apple Silicon',
    platform: 'darwin',
    deviceTier: 'apple_silicon',
  },
  cloud_server: {
    id: 'cloud_server',
    label: '云服务器',
    description: '云端高端 GPU（A100 / H100 / L40）',
    gpuType: 'A100',
    platform: 'linux',
    deviceTier: 'server_gpu',
  },
}

export interface VLMClass {
  class_name: string
  prompt: string
  negative_prompt: string
  color_hint: string | null
  display_name_zh?: string
  display_prompt_zh?: string
  display_negative_prompt_zh?: string
  display_color_hint_zh?: string | null
  // VLM 推理出的视觉特征
  estimated_size_hint?: string
  typical_perspective?: string
  rotation_invariant?: boolean
  occlusion_tolerance?: string
  color_consistency?: string
  data_augmentation_priority?: string[]
}

export interface VLMResult {
  classes: VLMClass[]
  raw_vlm_response: string
  confidence: number | null
  // VLM 全局推理结果
  scenario_hint?: string
  difficulty_hint?: string
  visual_insights?: string[]
  special_considerations?: string[]
}

export type VLMStatus = 'idle' | 'success' | 'failed'

export interface AugmentConfig {
  targetCount: number
  strength: 'light' | 'medium' | 'heavy'
  enabled: {
    geometric: boolean
    color: boolean
    noise: boolean
    weather: boolean
    occlusion: boolean
  }
  deleteOriginalImages: boolean
  // 高级参数
  minVisibility: number  // bbox 最小可见比例，低于此值丢弃
  maxPerImage: number   // 每张原图最多生成几张增强图
}

export interface TrainConfig {
  model: string
  epochs: number
  imgsz: number
  lr0: number
  patience: number
  conf: number
  iou: number
  exportFormats: ('onnx' | 'engine' | 'coreml' | 'openvino')[]
  trainMode: 'local' | 'cloud'
  gpuType: string
  // 本地 GPU 训练时使用的设备号，默认 0
  localDevice: number
  // 快速预览模式：少量数据、短 epoch，快速验证效果
  previewMode: boolean
  previewMaxImages: number
  previewMaxEpochs: number
  previewImgsz: number
  // 增量训练模式
  incrementalMode: boolean
  baseModelPath: string | null
  /** 增量训练时，合并后数据集的路径（由 startIncremental API 返回） */
  incrementalDatasetDir: string | null
  /** 增量训练时，合并后 data.yaml 的路径（由 startIncremental API 返回） */
  incrementalDataYaml: string | null
}

export interface PreviewResult {
  imageName: string
  detections: Array<{
    className: string
    confidence: number
    bbox: [number, number, number, number]
  }>
  imageBase64?: string
}

// ---------------------------------------------------------------------------
// 需求协商 (Negotiation)
// ---------------------------------------------------------------------------

export interface NegotiationMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  metadata?: {
    should_preview?: boolean
    config_updated?: boolean
    converged?: boolean
  }
}

export interface DetectionRules {
  conf_threshold: number
  iou_threshold: number
  post_filters: Array<{ type: string; value?: number; unit?: string }>
}

export interface VocabEntry {
  primary: string
  aliases: string[]
  context_anchors: string[]
}

export interface AlgorithmHints {
  scenario_type?: string
  needs_tracking?: boolean
  needs_ocr?: boolean
  events?: Array<{ name_zh: string; trigger: string }>
  regions?: Array<{ label: string; purpose: string }>
  performance_hint?: string
  multi_model_needed?: boolean
  suggested_pipeline_roles?: string[]
}

export interface NegotiatedConfig {
  classes: VLMClass[]
  detection_rules: DetectionRules
  vocab: Record<string, VocabEntry>
  algorithm_hints: AlgorithmHints | null
}

export interface TrainingProgress {
  state: string
  currentEpoch: number
  totalEpochs: number
  currentMap: number
  startedAt: string
}

export interface TaskState {
  taskId: string | null
  taskName: string
  stage: Stage

  // 阶段一
  sampleImages: File[]
  datasetImages: File[]
  sampleImageBoxes: Array<{ imageIndex: number; boxes: Array<{ x: number; y: number; width: number; height: number }> }>
  algorithmRegionBoxes: Array<{ x: number; y: number; width: number; height: number }>
  userDescription: string
  deviceProfileId: DeviceProfileId
  vlmResult: VLMResult | null
  vlmStatus: VLMStatus
  vlmErrorMessage: string | null
  vlmFallbackMode: boolean

  // 需求协商 (negotiation)
  conversationId: string | null
  negotiationMessages: NegotiationMessage[]
  negotiatedConfig: NegotiatedConfig | null
  negotiationConverged: boolean

  // 增量打标 (snowball)
  snowballMode: boolean
  snowballRound: number
  seedAnnotatedCount: number
  seedModelPath: string | null
  seedModelMap: number | null
  seedAutoLabelStats: {
    autoAccepted: number
    needsReview: number
    noDetection: number
    avgConfidence: number
  } | null

  // 阶段二
  skipLabeling: boolean
  skipQualityCheck: boolean
  labelingImageDir: string | null
  labelingProgress: {
    current: number
    total: number
    phase: 'detection' | 'quality_check' | 'loading_moondream'
  }
  labeledImageCount: number

  // 阶段二点五
  augConfig: AugmentConfig
  wasAugmented: boolean
  totalImageCount: number
  augmentationGenerated: number

  // 阶段三
  splitStats: { train: number; val: number; test: number }
  qualityReport: DataQualityReport | null

  // 阶段四
  trainConfig: TrainConfig
  trainConfigOverrides: Partial<Record<keyof TrainConfig, boolean>>
  trainingProgress: TrainingProgress | null
  artifacts: Record<string, string>
  algorithmPlan: StoredAlgorithmPlan | null
  previewResults: PreviewResult[]

  // Actions
  setSplitStats: (stats: { train: number; val: number; test: number }) => void
  setQualityReport: (report: DataQualityReport | null) => void
  setPreviewResults: (results: PreviewResult[]) => void
  setStage: (stage: Stage) => void
  setTaskMeta: (taskId: string, taskName: string) => void
  setSampleImages: (images: File[]) => void
  setDatasetImages: (images: File[]) => void
  setSampleImageBoxes: (
    imageIndex: number,
    boxes: Array<{ x: number; y: number; width: number; height: number }>
  ) => void
  setAlgorithmRegionBoxes: (boxes: Array<{ x: number; y: number; width: number; height: number }>) => void
  setUserDescription: (desc: string) => void
  setDeviceProfileId: (id: DeviceProfileId) => void
  setVLMResult: (result: VLMResult | null) => void
  setVLMStatus: (status: VLMStatus, message?: string | null) => void
  updateVLMClass: (index: number, updates: Partial<VLMClass>) => void
  removeVLMClass: (index: number) => void
  addVLMClass: (cls: VLMClass) => void
  setSkipLabeling: (skip: boolean) => void
  setSkipQualityCheck: (skip: boolean) => void
  setLabelingImageDir: (dir: string | null) => void
  setAugConfig: (config: Partial<AugmentConfig>) => void
  setWasAugmented: (value: boolean) => void
  setTrainConfig: (config: Partial<TrainConfig>) => void
  applyRecommendedTrainConfig: (config: Partial<TrainConfig>) => void
  setLabeledImageCount: (count: number) => void
  setTotalImageCount: (count: number) => void
  setTrainingProgress: (progress: TrainingProgress | null) => void
  setAlgorithmPlan: (plan: StoredAlgorithmPlan | null) => void
  setArtifacts: (artifacts: Record<string, string>) => void
  setSnowballMode: (mode: boolean) => void
  setSnowballRound: (round: number) => void
  setSeedAnnotatedCount: (count: number) => void
  setSeedModelPath: (path: string | null) => void
  setSeedModelMap: (map: number | null) => void
  setSeedAutoLabelStats: (stats: TaskState['seedAutoLabelStats']) => void
  // 需求协商 actions
  setConversationId: (id: string | null) => void
  addNegotiationMessage: (msg: NegotiationMessage) => void
  setNegotiationMessages: (msgs: NegotiationMessage[]) => void
  setNegotiatedConfig: (config: NegotiatedConfig | null) => void
  setNegotiationConverged: (v: boolean) => void
  resetNegotiation: () => void
  reset: () => void
}

export interface DataQualityReport {
  totalImages: number
  classDistribution: Array<{
    className: string
    boxCount: number
    avgBoxesPerImage: number
  }>
  avgBoxesPerImage: number
  warnings: string[]
}

const defaultAugConfig: AugmentConfig = {
  targetCount: 100,
  strength: 'medium',
  enabled: {
    geometric: true,
    color: true,
    noise: true,
    weather: false,
    occlusion: false,
  },
  deleteOriginalImages: false,
  minVisibility: 0.1,
  maxPerImage: 10,
}

const defaultTrainConfig: TrainConfig = {
  model: 'yolo11s.pt',
  epochs: 100,
  imgsz: 640,
  lr0: 0.01,
  patience: 20,
  conf: 0.25,
  iou: 0.7,
  exportFormats: ['onnx'],
  trainMode: 'local',
  gpuType: 'RTX 4090',
  localDevice: 0,
  previewMode: false,
  previewMaxImages: 20,
  previewMaxEpochs: 30,
  previewImgsz: 416,
  incrementalMode: false,
  baseModelPath: null,
  incrementalDatasetDir: null,
  incrementalDataYaml: null,
}

export const useTaskStore = create<TaskState>()(
  persist(
    (set) => ({
  taskId: null,
  taskName: '',
  stage: 'upload',

  sampleImages: [],
  datasetImages: [],
  sampleImageBoxes: [],
  algorithmRegionBoxes: [],
  userDescription: '',
  deviceProfileId: 'auto',
  vlmResult: null,
  vlmStatus: 'idle',
  vlmErrorMessage: null,
  vlmFallbackMode: false,

  conversationId: null,
  negotiationMessages: [],
  negotiatedConfig: null,
  negotiationConverged: false,

  snowballMode: false,
  snowballRound: 0,
  seedAnnotatedCount: 0,
  seedModelPath: null,
  seedModelMap: null,
  seedAutoLabelStats: null,
  incrementalResult: null as any,
  trainingHistory: [] as any[],

  skipLabeling: false,
  skipQualityCheck: true,
  labelingImageDir: null,

  labelingProgress: { current: 0, total: 0, phase: 'detection' },
  labeledImageCount: 0,

  augConfig: defaultAugConfig,
  wasAugmented: false,
  totalImageCount: 0,
  augmentationGenerated: 0,

  splitStats: { train: 0, val: 0, test: 0 },
  qualityReport: null,

  trainConfig: defaultTrainConfig,
  trainConfigOverrides: {},
  trainingProgress: null,
  artifacts: {},
  algorithmPlan: null,
  previewResults: [],

  setStage: (stage) => set({ stage }),

  setTaskMeta: (taskId, taskName) => set({
    taskId,
    taskName,
    skipLabeling: false,
    labelingImageDir: null,
    labeledImageCount: 0,
    labelingProgress: { current: 0, total: 0, phase: 'detection' },
  }),

  setSampleImages: (images) => set({ sampleImages: images }),

  setDatasetImages: (images) => set({ datasetImages: images }),

  setSampleImageBoxes: (imageIndex, boxes) =>
    set((state) => {
      const rest = state.sampleImageBoxes.filter((item) => item.imageIndex !== imageIndex)
      return {
        sampleImageBoxes: [...rest, { imageIndex, boxes }].sort((a, b) => a.imageIndex - b.imageIndex),
      }
    }),

  setAlgorithmRegionBoxes: (boxes) => set({ algorithmRegionBoxes: boxes }),

  setUserDescription: (desc) => set({ userDescription: desc }),

  setDeviceProfileId: (id) => set({ deviceProfileId: id }),

  setVLMResult: (result) => set({ vlmResult: result }),

  setVLMStatus: (status, message = null) =>
    set({
      vlmStatus: status,
      vlmErrorMessage: message,
      vlmFallbackMode: status === 'failed',
    }),

  setSkipLabeling: (skip: boolean) => set({ skipLabeling: skip }),

  setSkipQualityCheck: (skip: boolean) => set({ skipQualityCheck: skip }),

  setLabelingImageDir: (dir: string | null) => set({ labelingImageDir: dir }),

  updateVLMClass: (index, updates) =>
    set((state) => {
      if (!state.vlmResult) return state
      const newClasses = [...state.vlmResult.classes]
      newClasses[index] = { ...newClasses[index], ...updates }
      return { vlmResult: { ...state.vlmResult, classes: newClasses } }
    }),

  removeVLMClass: (index) =>
    set((state) => {
      if (!state.vlmResult) return state
      const newClasses = state.vlmResult.classes.filter((_, i) => i !== index)
      return { vlmResult: { ...state.vlmResult, classes: newClasses } }
    }),

  addVLMClass: (cls) =>
    set((state) => {
      if (!state.vlmResult) {
        return {
          vlmResult: {
            classes: [cls],
            raw_vlm_response: '',
            confidence: null,
          },
        }
      }
      return {
        vlmResult: {
          ...state.vlmResult,
          classes: [...state.vlmResult.classes, cls],
        },
      }
    }),

  setAugConfig: (config) =>
    set((state) => ({ augConfig: { ...state.augConfig, ...config } })),

  setWasAugmented: (value) => set({ wasAugmented: value }),

  setTrainConfig: (config) =>
    set((state) => {
      const trainConfigOverrides = { ...state.trainConfigOverrides }
      for (const key of Object.keys(config) as Array<keyof TrainConfig>) {
        trainConfigOverrides[key] = true
      }
      return {
        trainConfig: { ...state.trainConfig, ...config },
        trainConfigOverrides,
      }
    }),

  applyRecommendedTrainConfig: (config) =>
    set((state) => {
      const nextTrainConfig = { ...state.trainConfig }
      for (const key of Object.keys(config) as Array<keyof TrainConfig>) {
        if (state.trainConfigOverrides[key]) {
          continue
        }
        // 跳过预览模式相关字段（前端独立控制）
        if (key === 'previewMode' || key === 'previewMaxImages' || key === 'previewMaxEpochs' || key === 'previewImgsz') {
          continue
        }
        const value = config[key]
        if (value !== undefined) {
          ;(nextTrainConfig as Record<keyof TrainConfig, TrainConfig[keyof TrainConfig]>)[key] = value
        }
      }
      return { trainConfig: nextTrainConfig }
    }),

  setLabeledImageCount: (count) => set({ labeledImageCount: count }),

  setTotalImageCount: (count) => set({ totalImageCount: count }),

  setSplitStats: (stats) => set({ splitStats: stats }),

  setQualityReport: (report) => set({ qualityReport: report }),

  setPreviewResults: (results) => set({ previewResults: results }),

  setTrainingProgress: (progress) => set({ trainingProgress: progress }),

  setAlgorithmPlan: (plan) => set({ algorithmPlan: plan }),

  setArtifacts: (artifacts) => set({ artifacts }),

  setSnowballMode: (mode) => set({ snowballMode: mode }),
  setSnowballRound: (round) => set({ snowballRound: round }),
  setSeedAnnotatedCount: (count) => set({ seedAnnotatedCount: count }),
  setSeedModelPath: (path) => set({ seedModelPath: path }),
  setSeedModelMap: (map) => set({ seedModelMap: map }),
  setSeedAutoLabelStats: (stats) => set({ seedAutoLabelStats: stats }),

  // 需求协商 actions
  setConversationId: (id) => set({ conversationId: id }),
  addNegotiationMessage: (msg) =>
    set((state) => ({ negotiationMessages: [...state.negotiationMessages, msg] })),
  setNegotiationMessages: (msgs) => set({ negotiationMessages: msgs }),
  setNegotiatedConfig: (config) => set({ negotiatedConfig: config }),
  setNegotiationConverged: (v) => set({ negotiationConverged: v }),
  resetNegotiation: () =>
    set({
      conversationId: null,
      negotiationMessages: [],
      negotiatedConfig: null,
      negotiationConverged: false,
    }),

  reset: () =>
    set({
      taskId: null,
      taskName: '',
      stage: 'upload',
      sampleImages: [],
      datasetImages: [],
      sampleImageBoxes: [],
      algorithmRegionBoxes: [],
      userDescription: '',
      deviceProfileId: 'auto',
      vlmResult: null,
      vlmStatus: 'idle',
      vlmErrorMessage: null,
      vlmFallbackMode: false,
      conversationId: null,
      negotiationMessages: [],
      negotiatedConfig: null,
      negotiationConverged: false,
      snowballMode: false,
      snowballRound: 0,
      seedAnnotatedCount: 0,
      seedModelPath: null,
      seedModelMap: null,
      seedAutoLabelStats: null,
      skipLabeling: false,
      skipQualityCheck: true,
      labelingImageDir: null,
      labelingProgress: { current: 0, total: 0, phase: 'detection' },
      labeledImageCount: 0,
      augConfig: defaultAugConfig,
      wasAugmented: false,
      totalImageCount: 0,
      augmentationGenerated: 0,
      splitStats: { train: 0, val: 0, test: 0 },
      qualityReport: null,
      trainConfig: defaultTrainConfig,
      trainConfigOverrides: {},
      trainingProgress: null,
      artifacts: {},
      algorithmPlan: null,
      previewResults: [],
    }),
    }),
    {
      name: 'cv-auto-trainer-task',
      partialize: (state) => {
        // Exclude non-serializable File[] fields
        const { sampleImages, datasetImages, ...rest } = state
        return rest as Partial<TaskState>
      },
    },
  ),
)

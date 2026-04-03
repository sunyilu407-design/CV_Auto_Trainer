import { create } from 'zustand'

export type Stage =
  | 'upload'
  | 'intent_confirm'
  | 'labeling'
  | 'augment'
  | 'review'
  | 'train_config'
  | 'training'
  | 'delivery'

export interface VLMClass {
  class_name: string
  prompt: string
  negative_prompt: string
  color_hint: string | null
}

export interface VLMResult {
  classes: VLMClass[]
  raw_vlm_response: string
  confidence: number
}

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
  sampleImageBoxes: Array<{ imageIndex: number; boxes: Array<{ x: number; y: number; width: number; height: number }> }>
  userDescription: string
  vlmResult: VLMResult | null

  // 阶段二
  labelingProgress: {
    current: number
    total: number
    phase: 'detection' | 'quality_check'
  }
  labeledImageCount: number

  // 阶段二点五
  augConfig: AugmentConfig
  totalImageCount: number

  // 阶段三
  splitStats: { train: number; val: number; test: number }
  qualityReport: DataQualityReport | null

  // 阶段四
  trainConfig: TrainConfig
  trainingProgress: TrainingProgress | null
  artifacts: Record<string, string>

  // Actions
  setStage: (stage: Stage) => void
  setSampleImages: (images: File[]) => void
  setUserDescription: (desc: string) => void
  setVLMResult: (result: VLMResult) => void
  updateVLMClass: (index: number, updates: Partial<VLMClass>) => void
  setAugConfig: (config: Partial<AugmentConfig>) => void
  setTrainConfig: (config: Partial<TrainConfig>) => void
  setLabeledImageCount: (count: number) => void
  setTotalImageCount: (count: number) => void
  setTrainingProgress: (progress: TrainingProgress | null) => void
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
}

export const useTaskStore = create<TaskState>((set) => ({
  taskId: null,
  taskName: '',
  stage: 'upload',

  sampleImages: [],
  sampleImageBoxes: [],
  userDescription: '',
  vlmResult: null,

  labelingProgress: { current: 0, total: 0, phase: 'detection' },
  labeledImageCount: 0,

  augConfig: defaultAugConfig,
  totalImageCount: 0,

  splitStats: { train: 0, val: 0, test: 0 },
  qualityReport: null,

  trainConfig: defaultTrainConfig,
  trainingProgress: null,
  artifacts: {},

  setStage: (stage) => set({ stage }),

  setSampleImages: (images) => set({ sampleImages: images }),

  setUserDescription: (desc) => set({ userDescription: desc }),

  setVLMResult: (result) => set({ vlmResult: result }),

  updateVLMClass: (index, updates) =>
    set((state) => {
      if (!state.vlmResult) return state
      const newClasses = [...state.vlmResult.classes]
      newClasses[index] = { ...newClasses[index], ...updates }
      return { vlmResult: { ...state.vlmResult, classes: newClasses } }
    }),

  setAugConfig: (config) =>
    set((state) => ({ augConfig: { ...state.augConfig, ...config } })),

  setTrainConfig: (config) =>
    set((state) => ({ trainConfig: { ...state.trainConfig, ...config } })),

  setLabeledImageCount: (count) => set({ labeledImageCount: count }),

  setTotalImageCount: (count) => set({ totalImageCount: count }),

  setTrainingProgress: (progress) => set({ trainingProgress: progress }),

  reset: () =>
    set({
      taskId: null,
      taskName: '',
      stage: 'upload',
      sampleImages: [],
      sampleImageBoxes: [],
      userDescription: '',
      vlmResult: null,
      labelingProgress: { current: 0, total: 0, phase: 'detection' },
      labeledImageCount: 0,
      augConfig: defaultAugConfig,
      totalImageCount: 0,
      splitStats: { train: 0, val: 0, test: 0 },
      qualityReport: null,
      trainConfig: defaultTrainConfig,
      trainingProgress: null,
      artifacts: {},
    }),
}))

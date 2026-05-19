import { useEffect, useState, useCallback } from 'react'
import { useTaskStore } from '../store/taskStore'
import { useSettingsStore } from '../store/settingsStore'
import { trainingApi, TrainingEstimate } from '../api/backend'
import { workerClient } from '../api/worker'

// ─── Model Catalog ────────────────────────────────────────────────────────────
// Grouped by deployment device tier.
// Only models that support ultralytics-based training are listed.
// Non-ultralytics models (SAM-2, EfficientDet, YOLOX-custom) are excluded.

interface ModelOption {
  value: string
  label: string
  family: string
  /** COCO mAP50-95, for display only */
  map: string
  /** Params in millions */
  params: string
  /** GFLOPS */
  flops: string
  /** GPU FPS on RTX 4090 */
  fps: string
  /** Training task types */
  tasks: string[]
  /** Device tier recommendation */
  tiers: ('edge' | 'desktop' | 'server')[]
  note?: string
}

const TRAIN_MODELS: ModelOption[] = [
  // ── YOLOv5 ────────────────────────────────────────────────────────────────
  { value: 'yolov5n.pt', label: 'YOLOv5n', family: 'YOLOv5', map: '28.0', params: '1.9M', flops: '4.5G', fps: '450', tasks: ['检测'], tiers: ['edge', 'desktop'], note: '极轻量' },
  { value: 'yolov5s.pt', label: 'YOLOv5s', family: 'YOLOv5', map: '37.4', params: '7.2M', flops: '16.5G', fps: '370', tasks: ['检测'], tiers: ['edge', 'desktop'], note: '成熟稳定' },
  { value: 'yolov5m.pt', label: 'YOLOv5m', family: 'YOLOv5', map: '45.4', params: '21.2M', flops: '49G', fps: '230', tasks: ['检测'], tiers: ['desktop', 'server'] },
  { value: 'yolov5l.pt', label: 'YOLOv5l', family: 'YOLOv5', map: '49.0', params: '46.5M', flops: '109G', fps: '140', tasks: ['检测'], tiers: ['desktop', 'server'] },
  { value: 'yolov5x.pt', label: 'YOLOv5x', family: 'YOLOv5', map: '50.7', params: '86.7M', flops: '206G', fps: '85', tasks: ['检测'], tiers: ['server'] },

  // ── YOLOv8 ────────────────────────────────────────────────────────────────
  { value: 'yolov8n.pt', label: 'YOLOv8n', family: 'YOLOv8', map: '37.3', params: '3.2M', flops: '8.7G', fps: '520', tasks: ['检测', '分割', '姿态'], tiers: ['edge', 'desktop'], note: '推荐' },
  { value: 'yolov8s.pt', label: 'YOLOv8s', family: 'YOLOv8', map: '44.9', params: '11.2M', flops: '28.6G', fps: '410', tasks: ['检测', '分割', '姿态'], tiers: ['edge', 'desktop'], note: '工业首选' },
  { value: 'yolov8m.pt', label: 'YOLOv8m', family: 'YOLOv8', map: '50.2', params: '25.9M', flops: '79G', fps: '280', tasks: ['检测', '分割', '姿态'], tiers: ['desktop', 'server'] },
  { value: 'yolov8l.pt', label: 'YOLOv8l', family: 'YOLOv8', map: '52.9', params: '43.7M', flops: '165G', fps: '170', tasks: ['检测', '分割', '姿态'], tiers: ['server'] },
  { value: 'yolov8x.pt', label: 'YOLOv8x', family: 'YOLOv8', map: '54.5', params: '68.2M', flops: '258G', fps: '100', tasks: ['检测', '分割', '姿态'], tiers: ['server'] },

  // ── YOLOv9 ────────────────────────────────────────────────────────────────
  { value: 'yolov9n.pt', label: 'YOLOv9n', family: 'YOLOv9', map: '38.3', params: '2.0M', flops: '3.8G', fps: '500', tasks: ['检测'], tiers: ['edge', 'desktop'], note: 'GELAN 新架构' },
  { value: 'yolov9s.pt', label: 'YOLOv9s', family: 'YOLOv9', map: '40.2', params: '7.1M', flops: '17.5G', fps: '380', tasks: ['检测'], tiers: ['edge', 'desktop'], note: '精度优于 v8s' },
  { value: 'yolov9m.pt', label: 'YOLOv9m', family: 'YOLOv9', map: '42.8', params: '20.1M', flops: '52G', fps: '250', tasks: ['检测'], tiers: ['desktop', 'server'] },
  { value: 'yolov9l.pt', label: 'YOLOv9l', family: 'YOLOv9', map: '43.7', params: '37.5M', flops: '101G', fps: '160', tasks: ['检测'], tiers: ['server'] },
  { value: 'yolov9x.pt', label: 'YOLOv9x', family: 'YOLOv9', map: '44.9', params: '75.1M', flops: '185G', fps: '90', tasks: ['检测'], tiers: ['server'] },

  // ── YOLOv10 ────────────────────────────────────────────────────────────────
  { value: 'yolov10n.pt', label: 'YOLOv10n', family: 'YOLOv10', map: '39.8', params: '2.3M', flops: '6.7G', fps: '600', tasks: ['检测'], tiers: ['edge', 'desktop'], note: 'NMS-free 最快' },
  { value: 'yolov10s.pt', label: 'YOLOv10s', family: 'YOLOv10', map: '47.5', params: '7.2M', flops: '21.4G', fps: '480', tasks: ['检测'], tiers: ['edge', 'desktop'], note: 'NMS-free 首选' },
  { value: 'yolov10m.pt', label: 'YOLOv10m', family: 'YOLOv10', map: '51.0', params: '15.4M', flops: '59G', fps: '330', tasks: ['检测'], tiers: ['desktop', 'server'] },
  { value: 'yolov10l.pt', label: 'YOLOv10l', family: 'YOLOv10', map: '52.8', params: '24.4M', flops: '91G', fps: '220', tasks: ['检测'], tiers: ['server'] },
  { value: 'yolov10x.pt', label: 'YOLOv10x', family: 'YOLOv10', map: '54.5', params: '29.0M', flops: '137G', fps: '140', tasks: ['检测'], tiers: ['server'] },

  // ── YOLO11 ────────────────────────────────────────────────────────────────
  { value: 'yolo11n.pt', label: 'YOLO11n', family: 'YOLO11', map: '39.5', params: '2.6M', flops: '6.5G', fps: '580', tasks: ['检测', '分割', '姿态', 'OBB'], tiers: ['edge', 'desktop'], note: '最新+OBB' },
  { value: 'yolo11s.pt', label: 'YOLO11s', family: 'YOLO11', map: '47.0', params: '9.4M', flops: '21.5G', fps: '450', tasks: ['检测', '分割', '姿态', 'OBB'], tiers: ['edge', 'desktop'], note: '推荐默认', },
  { value: 'yolo11m.pt', label: 'YOLO11m', family: 'YOLO11', map: '51.5', params: '20.1M', flops: '68G', fps: '310', tasks: ['检测', '分割', '姿态', 'OBB'], tiers: ['desktop', 'server'] },
  { value: 'yolo11l.pt', label: 'YOLO11l', family: 'YOLO11', map: '53.4', params: '25.3M', flops: '87G', fps: '210', tasks: ['检测', '分割', '姿态', 'OBB'], tiers: ['server'] },

  // ── YOLO26 ────────────────────────────────────────────────────────────────
  { value: 'yolo26n.pt', label: 'YOLO26n', family: 'YOLO26', map: '41.0', params: '2.0M', flops: '5.0G', fps: '620', tasks: ['检测', '分割', '姿态', 'OBB'], tiers: ['edge', 'desktop'], note: '2025 最新' },
  { value: 'yolo26s.pt', label: 'YOLO26s', family: 'YOLO26', map: '49.0', params: '7.2M', flops: '16.5G', fps: '490', tasks: ['检测', '分割', '姿态', 'OBB'], tiers: ['edge', 'desktop'], note: '最新最优性价比' },
  { value: 'yolo26m.pt', label: 'YOLO26m', family: 'YOLO26', map: '53.0', params: '16.0M', flops: '55G', fps: '340', tasks: ['检测', '分割', '姿态', 'OBB'], tiers: ['desktop', 'server'] },
  { value: 'yolo26l.pt', label: 'YOLO26l', family: 'YOLO26', map: '54.5', params: '22.0M', flops: '76G', fps: '230', tasks: ['检测', '分割', '姿态', 'OBB'], tiers: ['server'] },

  // ── RT-DETR (Transformer) ─────────────────────────────────────────────────
  { value: 'rtdetr-s.pt', label: 'RT-DETR-S', family: 'RT-DETR', map: '50.0', params: '20M', flops: '60G', fps: '200', tasks: ['检测'], tiers: ['desktop', 'server'], note: 'Transformer · 密集场景' },
  { value: 'rtdetr-m.pt', label: 'RT-DETR-M', family: 'RT-DETR', map: '51.8', params: '27M', flops: '85G', fps: '175', tasks: ['检测'], tiers: ['desktop', 'server'] },
  { value: 'rtdetr-l.pt', label: 'RT-DETR-L', family: 'RT-DETR', map: '53.0', params: '32M', flops: '110G', fps: '160', tasks: ['检测'], tiers: ['server'], note: '密集/遮挡首选' },
  { value: 'rtdetr-x.pt', label: 'RT-DETR-X', family: 'RT-DETR', map: '54.8', params: '67M', flops: '234G', fps: '95', tasks: ['检测'], tiers: ['server'] },

  // ── YOLOX ────────────────────────────────────────────────────────────────
  { value: 'yolox-s', label: 'YOLOX-S', family: 'YOLOX', map: '44.3', params: '9.0M', flops: '27G', fps: '380', tasks: ['检测'], tiers: ['edge', 'desktop'], note: 'Anchor-Free · 无需锚框调参' },
  { value: 'yolox-m', label: 'YOLOX-M', family: 'YOLOX', map: '47.5', params: '25.3M', flops: '74G', fps: '240', tasks: ['检测'], tiers: ['desktop', 'server'] },
  { value: 'yolox-l', label: 'YOLOX-L', family: 'YOLOX', map: '49.6', params: '54.2M', flops: '156G', fps: '150', tasks: ['检测'], tiers: ['server'] },
]

type DeviceTier = 'edge' | 'desktop' | 'server'

const TIER_LABELS: Record<DeviceTier, { label: string; icon: string; color: string; desc: string }> = {
  edge: {
    label: '边缘设备',
    icon: '⚡',
    color: '#16a34a',
    desc: 'Jetson Nano/Xavier、RTX 1650、Mac M1、低功耗嵌入式',
  },
  desktop: {
    label: '桌面 GPU',
    icon: '🖥️',
    color: '#2563eb',
    desc: 'RTX 3060~4090、Mac M2/M3、高端台式机',
  },
  server: {
    label: '服务器 GPU',
    icon: '🖧',
    color: '#9333ea',
    desc: 'RTX 3090/4090、A100/H100、昇腾、云端实例',
  },
}

const SOURCE_LABELS: Record<string, string> = {
  algorithm: '算法推荐',
  runtime: '环境推荐',
  user_default: '用户默认',
  system_default: '系统默认',
}

const SOURCE_STYLES: Record<string, { background: string; color: string }> = {
  algorithm: { background: 'rgba(10,114,239,0.08)', color: 'var(--develop-blue)' },
  runtime: { background: 'rgba(222,29,141,0.08)', color: 'var(--preview-pink)' },
  user_default: { background: 'rgba(17,24,39,0.06)', color: 'var(--gray-700)' },
  system_default: { background: 'rgba(107,114,128,0.10)', color: 'var(--gray-500)' },
  override: { background: 'rgba(255,91,79,0.10)', color: 'var(--ship-red)' },
}

function SourceBadge({ label, variant }: { label: string; variant: keyof typeof SOURCE_STYLES }) {
  const style = SOURCE_STYLES[variant]
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: '3px 8px',
        borderRadius: 999,
        fontSize: 11,
        fontWeight: 600,
        letterSpacing: '0.02em',
        background: style.background,
        color: style.color,
        whiteSpace: 'nowrap',
      }}
    >
      {label}
    </span>
  )
}

export default function TrainConfig() {
  const { taskId, trainConfig, trainConfigOverrides, setTrainConfig, applyRecommendedTrainConfig, setStage, setTrainingProgress, setArtifacts, setStage: setAppStage, algorithmPlan } = useTaskStore()
  const { settings } = useSettingsStore()
  const trainingRecommendation = algorithmPlan?.pipeline_config?.training_recommendation
  const recommendationSummary = trainingRecommendation?.reason_summary

  const trainableStepsCount = (algorithmPlan?.algorithm_plan?.model_pipeline ?? []).filter(
    (s) => (s.role === 'primary_detector' || s.role === 'secondary_detector' || s.role === 'classifier')
      && (s.requires_training !== false || s.reuse_cache_id)
  ).length
  const isMultiModel = trainableStepsCount > 1
  const localBlocked = isMultiModel && trainConfig.trainMode === 'local'

  useEffect(() => {
    if (!trainingRecommendation?.recommended_config) return
    applyRecommendedTrainConfig({
      model: trainingRecommendation.recommended_config.model,
      epochs: trainingRecommendation.recommended_config.epochs,
      imgsz: trainingRecommendation.recommended_config.imgsz,
      lr0: trainingRecommendation.recommended_config.lr0,
      patience: trainingRecommendation.recommended_config.patience,
      conf: trainingRecommendation.recommended_config.conf,
      iou: trainingRecommendation.recommended_config.iou,
      exportFormats: trainingRecommendation.recommended_config.export_formats,
      trainMode: trainingRecommendation.recommended_config.train_mode,
      gpuType: trainingRecommendation.recommended_config.gpu_type,
    })
  }, [trainingRecommendation, applyRecommendedTrainConfig])

  const getSourceMeta = (field: keyof typeof trainConfig, sourceKey?: string) => {
    if (trainConfigOverrides[field]) {
      return { label: '已手动修改', variant: 'override' as const }
    }
    const source = trainingRecommendation?.source_map?.[sourceKey ?? field] ?? 'system_default'
    return {
      label: SOURCE_LABELS[source] ?? '系统默认',
      variant: (source in SOURCE_STYLES ? source : 'system_default') as keyof typeof SOURCE_STYLES,
    }
  }

  const renderFieldHeader = (label: string, field: keyof typeof trainConfig, sourceKey?: string) => {
    const meta = getSourceMeta(field, sourceKey)
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginBottom: 8 }}>
        <span className="form-label" style={{ marginBottom: 0 }}>{label}</span>
        <SourceBadge label={meta.label} variant={meta.variant} />
      </div>
    )
  }

  const [estimate, setEstimate] = useState<TrainingEstimate | null>(null)
  const [estimateLoading, setEstimateLoading] = useState(false)

  const fetchEstimate = useCallback(async () => {
    if (!taskId || trainConfig.trainMode === 'local') {
      setEstimate(null)
      return
    }
    setEstimateLoading(true)
    try {
      const est = await trainingApi.estimate({
        task_id: taskId,
        model: trainConfig.model,
        epochs: trainConfig.epochs,
        imgsz: trainConfig.imgsz,
        gpu_type: trainConfig.gpuType,
      })
      setEstimate(est)
    } catch {
      setEstimate(null)
    } finally {
      setEstimateLoading(false)
    }
  }, [taskId, trainConfig.model, trainConfig.epochs, trainConfig.imgsz, trainConfig.gpuType, trainConfig.trainMode])

  useEffect(() => {
    const timer = setTimeout(fetchEstimate, 400)
    return () => clearTimeout(timer)
  }, [fetchEstimate])

  const handleStartTraining = async () => {
    if (!taskId) {
      alert('缺少任务 ID，请重新从上传阶段开始')
      return
    }

    // 预览模式：用少量数据快速验证效果
    let effectiveConfig = trainConfig.previewMode ? {
      ...trainConfig,
      epochs: trainConfig.previewMaxEpochs,
      imgsz: trainConfig.previewImgsz,
    } : { ...trainConfig }

    // 增量训练模式：使用 baseModelPath 和调整参数
    if (trainConfig.incrementalMode && trainConfig.baseModelPath) {
      effectiveConfig = {
        ...effectiveConfig,
        model: trainConfig.baseModelPath,
        epochs: Math.min(effectiveConfig.epochs, 30),
        lr0: Math.min(effectiveConfig.lr0, 0.005),
        patience: Math.min(effectiveConfig.patience, 10),
      }
    }

    setStage('training')
    setTrainingProgress({
      state: 'starting',
      currentEpoch: 0,
      totalEpochs: effectiveConfig.epochs,
      currentMap: 0,
      startedAt: new Date().toISOString(),
    })

    if (trainConfig.trainMode === 'local') {
      workerClient.connect()
      workerClient.onMessage((msg) => {
        if (msg.type === 'training_progress') {
          setTrainingProgress({
            state: 'training',
            currentEpoch: msg.currentEpoch as number,
            totalEpochs: effectiveConfig.epochs,
            currentMap: msg.currentMap as number,
            startedAt: new Date().toISOString(),
          })
        }
        if (msg.type === 'training_complete') {
          const artifacts = (msg as unknown as { artifacts: Record<string, string> }).artifacts
          setArtifacts(artifacts ?? {})
          // 立即标记完成状态，触发预览推理 effect
          setTrainingProgress({
            state: 'done',
            currentEpoch: effectiveConfig.epochs,
            totalEpochs: effectiveConfig.epochs,
            currentMap: 0,
            startedAt: new Date().toISOString(),
          })
          // 非预览模式直接跳转
          if (!trainConfig.previewMode) {
            setAppStage('delivery')
          }
        }
        if (msg.type === 'training_error') {
          alert(`训练出错: ${String((msg as unknown as { message: string }).message ?? '未知错误')}`)
          setAppStage('train_config')
        }
      })
      const datasetDir = trainConfig.incrementalMode && trainConfig.incrementalDatasetDir
        ? trainConfig.incrementalDatasetDir.replace(/^.*?\/uploads\//, '../backend/uploads/')
        : `../backend/uploads/${taskId}/dataset`
      const dataYaml = trainConfig.incrementalMode && trainConfig.incrementalDataYaml
        ? trainConfig.incrementalDataYaml.replace(/^.*?\/uploads\//, '../backend/uploads/')
        : undefined
      workerClient.startLocalTraining({
        dataset_dir: datasetDir,
        data_yaml: dataYaml,
        train_config: effectiveConfig as unknown as Record<string, unknown>,
      })
    } else {
      try {
        await trainingApi.start({
          task_id: taskId,
          model: trainConfig.model,
          epochs: trainConfig.epochs,
          imgsz: trainConfig.imgsz,
          lr0: trainConfig.lr0,
          patience: trainConfig.patience,
          conf: trainConfig.conf,
          iou: trainConfig.iou,
          export_formats: trainConfig.exportFormats,
          train_mode: trainConfig.trainMode,
          gpu_type: trainConfig.gpuType,
          local_device: trainConfig.localDevice,
          resume_last: false,
        })
      } catch (e) {
        alert(`启动失败: ${e}`)
        setStage('train_config')
      }
    }
  }

  return (
    <div>
      {/* Page Header */}
      <div className="page-header" style={{ marginBottom: 32 }}>
        <div className="flex items-center gap-3 mb-4">
          <div className="badge" style={{ background: 'var(--preview-pink)', color: '#fff', fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Stage 3</div>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: 'var(--preview-pink)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3"/>
              <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/>
            </svg>
          </div>
        </div>
        <h1 className="page-title">训练配置</h1>
        <p className="page-subtitle">
          {trainConfig.incrementalMode
            ? `增量训练 v${(algorithmPlan?.algorithm_plan?.training_version ?? 0) + 1}：基于已有模型微调`
            : '选择模型、训练模式与超参数'}
        </p>
      </div>

      {trainingRecommendation && (
        <div className="card-section" style={{ marginBottom: 16, background: 'rgba(10,114,239,0.03)' }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-500)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 12 }}>算法推荐</h3>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
            <span className="badge badge-blue">Model {trainingRecommendation.recommended_model}</span>
            <span className="badge badge-pink">Mode {trainingRecommendation.train_mode}</span>
            <span className="badge badge-dark">Export {trainingRecommendation.export_formats.join(', ')}</span>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
            <SourceBadge label={`模型：${getSourceMeta('model', 'model').label}`} variant={getSourceMeta('model', 'model').variant} />
            <SourceBadge label={`训练模式：${getSourceMeta('trainMode', 'train_mode').label}`} variant={getSourceMeta('trainMode', 'train_mode').variant} />
            <SourceBadge label={`导出格式：${getSourceMeta('exportFormats', 'export_formats').label}`} variant={getSourceMeta('exportFormats', 'export_formats').variant} />
          </div>
          <p style={{ fontSize: 12, color: 'var(--gray-500)', lineHeight: 1.6 }}>
            {recommendationSummary || '当前训练默认值已经按算法规划自动预填。你仍然可以手动覆盖这些建议。'}
          </p>
        </div>
      )}

      {/* Train Mode */}
      <div className="card-section" style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginBottom: 12 }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-500)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 0 }}>训练模式</h3>
          <SourceBadge label={getSourceMeta('trainMode', 'train_mode').label} variant={getSourceMeta('trainMode', 'train_mode').variant} />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {[
            {
              value: 'local',
              icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>,
              label: '本地训练',
              desc: '使用本机 GPU，无需上传数据',
              accent: 'var(--develop-blue)',
            },
            {
              value: 'cloud',
              icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M18 10h-1.26A8 8 0 109 20h9a5 5 0 000-10z"/></svg>,
              label: '云端训练',
              desc: settings.cloudProvider === 'autodl' ? 'AutoDL 自动调度' : 'SSH 云服务器',
              accent: 'var(--preview-pink)',
            },
          ].map((mode) => {
            const isActive = trainConfig.trainMode === mode.value
            return (
              <label
                key={mode.value}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 10,
                  padding: '16px 20px',
                  borderRadius: 10,
                  border: `2px solid ${isActive ? mode.accent : 'var(--gray-100)'}`,
                  background: isActive ? `${mode.accent}08` : 'var(--gray-50)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                <input
                  type="radio"
                  name="trainMode"
                  value={mode.value}
                  checked={isActive}
                  onChange={() => setTrainConfig({ trainMode: mode.value as 'local' | 'cloud' })}
                  style={{ display: 'none' }}
                />
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div style={{ color: isActive ? mode.accent : 'var(--gray-400)', transition: 'color 0.15s' }}>{mode.icon}</div>
                  <span style={{ fontSize: 14, fontWeight: 600, color: isActive ? mode.accent : 'var(--gray-700)' }}>{mode.label}</span>
                  {isActive && (
                    <div style={{ marginLeft: 'auto', width: 16, height: 16, borderRadius: '50%', background: mode.accent, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3"><polyline points="20 6 9 17 4 12"/></svg>
                    </div>
                  )}
                </div>
                <p style={{ fontSize: 12, color: 'var(--gray-500)', margin: 0 }}>{mode.desc}</p>
              </label>
            )
          })}
        </div>
      </div>

      {/* Incremental Mode Banner */}
      {trainConfig.incrementalMode && (
        <div
          style={{
            padding: '14px 18px',
            marginBottom: 16,
            background: 'rgba(245,158,11,0.06)',
            border: '1px solid rgba(245,158,11,0.3)',
            borderRadius: 10,
            display: 'flex',
            alignItems: 'flex-start',
            gap: 12,
          }}
        >
          <span style={{ fontSize: 18, flexShrink: 0 }}>⚡</span>
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <span style={{ fontSize: 13, fontWeight: 700, color: '#92400e' }}>增量训练模式</span>
              {trainConfig.incrementalDatasetDir && (
                <span style={{ fontSize: 11, padding: '1px 8px', borderRadius: 4, background: 'rgba(245,158,11,0.15)', color: '#92400e' }}>
                  合并数据集已就绪
                </span>
              )}
            </div>
            <div style={{ fontSize: 12, color: '#a16207', lineHeight: 1.6 }}>
              基于 <code style={{ fontFamily: 'var(--font-mono)', background: 'rgba(245,158,11,0.1)', padding: '1px 4px', borderRadius: 3 }}>{trainConfig.baseModelPath?.split('/').pop() ?? '已选模型'}</code> 继续微调。
              新的 badcase 图片已通过已有模型自动预标注。
              学习率 / Epoch / 早停已自动调低以保护已有精度。
            </div>
          </div>
        </div>
      )}

      {/* Model Select */}
      <div className="card-section" style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginBottom: 12 }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-500)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 0 }}>
            {trainConfig.incrementalMode ? '模型 (增量模式已锁定)' : '模型选择'}
          </h3>
          {!trainConfig.incrementalMode && <SourceBadge label={getSourceMeta('model', 'model').label} variant={getSourceMeta('model', 'model').variant} />}
        </div>
        {trainConfig.incrementalMode ? (
          <div style={{ padding: '12px 16px', background: 'var(--gray-50)', borderRadius: 8, fontSize: 13, color: 'var(--gray-600)' }}>
            使用上次训练的 <strong>best.pt</strong> 作为基础模型（增量模式不可切换预训练模型）
          </div>
        ) : (
          <div>
            {/* Active model summary bar */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 12, padding: '10px 14px',
              background: 'rgba(10,114,239,0.06)', borderRadius: 8, marginBottom: 14,
              border: '1px solid rgba(10,114,239,0.15)',
            }}>
              <span style={{ fontSize: 12, color: 'var(--gray-400)', minWidth: 60 }}>已选模型</span>
              <span style={{ fontWeight: 700, fontSize: 14, color: 'var(--develop-blue)' }}>
                {TRAIN_MODELS.find((m) => m.value === trainConfig.model)?.label ?? trainConfig.model}
              </span>
              {(() => {
                const m = TRAIN_MODELS.find((m) => m.value === trainConfig.model)
                if (!m) return null
                return (
                  <span style={{ display: 'flex', gap: 8 }}>
                    {m.tasks.map((t) => (
                      <span key={t} style={{ fontSize: 11, padding: '1px 6px', borderRadius: 4, background: 'rgba(10,114,239,0.1)', color: 'var(--develop-blue)' }}>{t}</span>
                    ))}
                    <span style={{ fontSize: 11, color: 'var(--gray-400)' }}>
                      mAP: {m.map} · {m.params} · {m.fps} FPS
                    </span>
                  </span>
                )
              })()}
            </div>

            {/* Tier-grouped model grid */}
            {(['edge', 'desktop', 'server'] as DeviceTier[]).map((tier) => {
              const tierModels = TRAIN_MODELS.filter((m) => m.tiers.includes(tier))
              if (!tierModels.length) return null
              const tierInfo = TIER_LABELS[tier]
              return (
                <div key={tier} style={{ marginBottom: 20 }}>
                  {/* Tier header */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                    <span style={{ fontSize: 14 }}>{tierInfo.icon}</span>
                    <span style={{ fontWeight: 600, fontSize: 13, color: tierInfo.color }}>{tierInfo.label}</span>
                    <span style={{ fontSize: 11, color: 'var(--gray-400)' }}>{tierInfo.desc}</span>
                  </div>
                  {/* Model cards grid */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: 8 }}>
                    {tierModels.map((m) => {
                      const isActive = trainConfig.model === m.value
                      return (
                        <label
                          key={m.value}
                          style={{
                            display: 'flex', flexDirection: 'column', gap: 4,
                            padding: '10px 10px',
                            borderRadius: 8,
                            border: `1.5px solid ${isActive ? tierInfo.color : 'var(--gray-100)'}`,
                            background: isActive ? `${tierInfo.color}0c` : 'var(--gray-50)',
                            cursor: 'pointer',
                            transition: 'all 0.15s ease',
                          }}
                        >
                          <input
                            type="radio"
                            name="model"
                            value={m.value}
                            checked={isActive}
                            onChange={() => setTrainConfig({ model: m.value })}
                            style={{ display: 'none' }}
                          />
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 4 }}>
                            <span style={{ fontSize: 12, fontWeight: 700, color: isActive ? tierInfo.color : 'var(--gray-700)' }}>
                              {m.label}
                            </span>
                            {m.note && (
                              <span style={{ fontSize: 9, padding: '1px 4px', borderRadius: 3, background: `${tierInfo.color}15`, color: tierInfo.color, whiteSpace: 'nowrap' }}>
                                {m.note}
                              </span>
                            )}
                          </div>
                          <div style={{ fontSize: 10, color: 'var(--gray-400)', display: 'flex', flexDirection: 'column', gap: 1 }}>
                            <span>mAP: {m.map} · {m.params}</span>
                            <span>{m.fps} FPS</span>
                          </div>
                        </label>
                      )
                    })}
                  </div>
                </div>
              )
            })}

            {/* Column legend */}
            <div style={{ display: 'flex', gap: 16, marginTop: 8, flexWrap: 'wrap' }}>
              <span style={{ fontSize: 11, color: 'var(--gray-400)' }}>
                mAP = COCO mAP50-95（越高越好） · FPS = RTX 4090 推理帧率（越高越好） · params = 参数量（越小越轻）
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Hyperparameters */}
      <div className="card-section" style={{ marginBottom: 16 }}>
        <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-500)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 12 }}>超参数</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {[
            { key: 'epochs', label: 'Epochs', sub: '训练轮次', min: 10, max: 500 },
            { key: 'imgsz', label: 'Image Size', sub: '图像尺寸', options: [416, 512, 640, 1280] },
            { key: 'lr0', label: '学习率 lr0', sub: '', min: 0.0001, max: 0.1, step: 0.001 },
            { key: 'patience', label: 'Patience', sub: '早停', min: 5, max: 100 },
            { key: 'conf', label: '推理置信度', sub: '', min: 0.1, max: 0.9, step: 0.05 },
            { key: 'iou', label: 'NMS IoU', sub: '', min: 0.3, max: 0.9, step: 0.05 },
          ].map((param) => (
            <div key={param.key} className="form-group">
              {renderFieldHeader(param.label, param.key as keyof typeof trainConfig, param.key)}
              {param.options ? (
                <select
                  className="input"
                  value={(trainConfig as unknown as Record<string, unknown>)[param.key] as number}
                  onChange={(e) => setTrainConfig({ [param.key]: Number(e.target.value) } as unknown as typeof trainConfig)}
                >
                  {param.options.map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              ) : (
                <input
                  type="number"
                  className="input"
                  min={param.min}
                  max={param.max}
                  step={param.step ?? 1}
                  value={(trainConfig as unknown as Record<string, unknown>)[param.key] as number}
                  onChange={(e) => setTrainConfig({ [param.key]: Number(e.target.value) } as unknown as typeof trainConfig)}
                />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Export Formats */}
      <div className="card-section" style={{ marginBottom: 32 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginBottom: 12 }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-500)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 0 }}>导出格式</h3>
          <SourceBadge label={getSourceMeta('exportFormats', 'export_formats').label} variant={getSourceMeta('exportFormats', 'export_formats').variant} />
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {(['onnx', 'engine', 'coreml', 'openvino'] as const).map((fmt) => {
            const isActive = trainConfig.exportFormats.includes(fmt)
            return (
              <label
                key={fmt}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '8px 14px',
                  borderRadius: 6,
                  border: `1.5px solid ${isActive ? 'var(--develop-blue)' : 'var(--gray-100)'}`,
                  background: isActive ? 'rgba(10,114,239,0.05)' : 'var(--gray-50)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                <input
                  type="checkbox"
                  checked={isActive}
                  onChange={(e) => {
                    if (e.target.checked) setTrainConfig({ exportFormats: [...trainConfig.exportFormats, fmt] })
                    else setTrainConfig({ exportFormats: trainConfig.exportFormats.filter((f) => f !== fmt) })
                  }}
                  style={{ accentColor: 'var(--develop-blue)' }}
                />
                <span style={{ fontSize: 13, fontFamily: 'var(--font-mono)', fontWeight: 600, color: isActive ? 'var(--develop-blue)' : 'var(--gray-500)' }}>
                  .{fmt.toUpperCase()}
                </span>
              </label>
            )
          })}
        </div>
      </div>

      {localBlocked && (
        <div style={{
          marginBottom: 16,
          padding: '14px 16px',
          borderRadius: 8,
          background: '#fff8e1',
          border: '1px solid #f59e0b',
          fontSize: 13,
          color: '#92400e',
          lineHeight: 1.7,
        }}>
          <strong>⚠️ 本地模式暂不支持多模型流水线</strong>
          <div style={{ marginTop: 6, fontSize: 12 }}>
            当前方案包含 <strong>{trainableStepsCount}</strong> 个需要训练的模型，本地训练一次只能跑一个。
            请切换到 <strong>云端训练</strong>（会按优先级依次训练每个模型），或返回方案页简化为单一检测器。
          </div>
        </div>
      )}

      {/* Local GPU Device Select */}
      {trainConfig.trainMode === 'local' && (
        <div className="card-section" style={{ marginBottom: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginBottom: 12 }}>
            <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-500)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 0 }}>本地 GPU 设备</h3>
            <SourceBadge label={getSourceMeta('localDevice').label} variant={getSourceMeta('localDevice').variant} />
          </div>
          <select
            className="input"
            value={trainConfig.localDevice}
            onChange={(e) => setTrainConfig({ localDevice: Number(e.target.value) })}
            style={{ maxWidth: 200 }}
          >
            <option value={0}>GPU 0（默认）</option>
            <option value={1}>GPU 1</option>
            <option value={2}>GPU 2</option>
            <option value={3}>GPU 3</option>
          </select>
          <p style={{ fontSize: 12, color: 'var(--gray-400)', marginTop: 6 }}>
            如果你有多个 GPU，可以选择使用哪一个。默认使用第一个 GPU。
          </p>
        </div>
      )}
      <div className="card-section" style={{ marginBottom: 16, background: trainConfig.previewMode ? 'rgba(16,185,129,0.04)' : undefined, border: trainConfig.previewMode ? '1px solid rgba(16,185,129,0.2)' : undefined }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16 }}>
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
              <span className="badge" style={{ background: trainConfig.previewMode ? 'var(--success-green)' : 'var(--gray-200)', color: trainConfig.previewMode ? '#fff' : 'var(--gray-500)', fontSize: 11, fontWeight: 600 }}>Preview Mode</span>
              <h3 style={{ fontSize: 15, fontWeight: 600, margin: 0 }}>快速预览训练效果</h3>
            </div>
            <p style={{ fontSize: 12, color: 'var(--gray-500)', margin: 0, lineHeight: 1.6 }}>
              用少量数据（{trainConfig.previewMaxImages} 张）+ 短 epoch（{trainConfig.previewMaxEpochs}）+ 小分辨率（{trainConfig.previewImgsz}）快速验证效果，
              通常 1~5 分钟即可看到结果。训练完成后自动在样板上推理，直观判断识别是否正确。
            </p>
          </div>
          <label style={{ position: 'relative', display: 'inline-flex', cursor: 'pointer', flexShrink: 0 }}>
            <input
              type="checkbox"
              checked={trainConfig.previewMode}
              onChange={(e) => setTrainConfig({ previewMode: e.target.checked })}
              style={{ width: 0, height: 0, opacity: 0 }}
            />
            <div
              style={{
                width: 44, height: 24,
                borderRadius: 12,
                background: trainConfig.previewMode ? 'var(--success-green)' : 'var(--gray-200)',
                position: 'relative',
                transition: 'all 0.2s ease',
              }}
            >
              <div
                style={{
                  width: 18, height: 18, borderRadius: '50%',
                  background: '#fff',
                  position: 'absolute',
                  top: 3,
                  left: trainConfig.previewMode ? 23 : 3,
                  transition: 'left 0.2s ease',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.15)',
                }}
              />
            </div>
          </label>
        </div>

        {trainConfig.previewMode && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--gray-100)' }}>
            <div className="form-group">
              <label className="form-label">最多图片数</label>
              <input
                type="number"
                className="input"
                min={5}
                max={100}
                value={trainConfig.previewMaxImages}
                onChange={(e) => setTrainConfig({ previewMaxImages: Number(e.target.value) })}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Epoch 数</label>
              <input
                type="number"
                className="input"
                min={5}
                max={100}
                value={trainConfig.previewMaxEpochs}
                onChange={(e) => setTrainConfig({ previewMaxEpochs: Number(e.target.value) })}
              />
            </div>
            <div className="form-group">
              <label className="form-label">图像分辨率</label>
              <select
                className="input"
                value={trainConfig.previewImgsz}
                onChange={(e) => setTrainConfig({ previewImgsz: Number(e.target.value) })}
              >
                {[320, 416, 512, 640].map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
          </div>
        )}
      </div>

      {/* Cost Estimate */}
      {trainConfig.trainMode === 'cloud' && estimate && (
        <div className="card-section" style={{ marginBottom: 16, background: 'rgba(10,114,239,0.03)', border: '1px solid rgba(10,114,239,0.12)' }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-500)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 12 }}>
            费用预估
            {estimateLoading && <span style={{ fontWeight: 400, fontSize: 11, color: 'var(--gray-400)', marginLeft: 8 }}>更新中...</span>}
          </h3>
          <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', marginBottom: 12 }}>
            <div>
              <div style={{ fontSize: 11, color: 'var(--gray-400)', marginBottom: 2 }}>预计时长</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--gray-900)' }}>
                {estimate.total_duration_min < 60
                  ? `${estimate.total_duration_min} 分钟`
                  : `${(estimate.total_duration_min / 60).toFixed(1)} 小时`}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: 'var(--gray-400)', marginBottom: 2 }}>预计费用</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--preview-pink)' }}>¥{estimate.total_cost_cny.toFixed(2)}</div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: 'var(--gray-400)', marginBottom: 2 }}>GPU</div>
              <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--gray-700)' }}>{estimate.gpu_type} (¥{estimate.hourly_rate_cny}/h)</div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: 'var(--gray-400)', marginBottom: 2 }}>数据集</div>
              <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--gray-700)' }}>{estimate.total_images} 张</div>
            </div>
          </div>
          {estimate.steps.length > 1 && (
            <div style={{ fontSize: 12, color: 'var(--gray-500)', lineHeight: 1.8 }}>
              {estimate.steps.map((s) => (
                <div key={s.step_id}>
                  <span style={{ fontWeight: 600 }}>{s.step_id}</span>
                  {' · '}{s.model_id}
                  {s.source === 'reuse'
                    ? ' · 复用缓存（免费）'
                    : ` · ~${s.duration_min} 分钟 · ¥${s.cost_cny}`}
                </div>
              ))}
            </div>
          )}
          <div style={{ fontSize: 11, color: 'var(--gray-400)', marginTop: 8 }}>
            * 费用预估仅供参考，实际费用以云平台账单为准
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-3">
        <button
          className="btn btn-primary"
          onClick={handleStartTraining}
          disabled={localBlocked}
          style={{
            padding: '12px 28px',
            fontSize: 15,
            fontWeight: 600,
            background: localBlocked ? 'var(--gray-300)' : 'var(--preview-pink)',
            cursor: localBlocked ? 'not-allowed' : 'pointer',
          }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          {trainConfig.trainMode === 'local' ? '开始本地训练' : '开始云端训练'}
        </button>
        <button className="btn btn-secondary" onClick={() => setStage('review')} style={{ padding: '12px 20px' }}>
          返回
        </button>
      </div>
    </div>
  )
}

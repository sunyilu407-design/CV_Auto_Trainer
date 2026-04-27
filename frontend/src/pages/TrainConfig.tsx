import { useEffect, useState, useCallback } from 'react'
import { useTaskStore } from '../store/taskStore'
import { useSettingsStore } from '../store/settingsStore'
import { trainingApi, TrainingEstimate } from '../api/backend'
import { workerClient } from '../api/worker'

const MODELS = [
  { value: 'yolo11n.pt', label: 'YOLO11n', sub: '极限压缩，推荐嵌入式' },
  { value: 'yolo11s.pt', label: 'YOLO11s', sub: '推荐，默认' },
  { value: 'yolo11m.pt', label: 'YOLO11m', sub: '工控机/服务器' },
  { value: 'yolo11l.pt', label: 'YOLO11l', sub: '高精度' },
  { value: 'rtdetr-l.pt', label: 'RT-DETR-L', sub: '密集/遮挡场景' },
]

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
    const effectiveConfig = trainConfig.previewMode ? {
      ...trainConfig,
      epochs: trainConfig.previewMaxEpochs,
      imgsz: trainConfig.previewImgsz,
    } : trainConfig

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
      workerClient.startLocalTraining({
        dataset_dir: `../backend/uploads/${taskId}/dataset`,
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
        <p className="page-subtitle">选择模型、训练模式与超参数</p>
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

      {/* Model Select */}
      <div className="card-section" style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginBottom: 12 }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-500)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 0 }}>模型选择</h3>
          <SourceBadge label={getSourceMeta('model', 'model').label} variant={getSourceMeta('model', 'model').variant} />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8 }}>
          {MODELS.map((m) => {
            const isActive = trainConfig.model === m.value
            return (
              <label
                key={m.value}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 6,
                  padding: '12px 8px',
                  borderRadius: 8,
                  border: `1.5px solid ${isActive ? 'var(--develop-blue)' : 'var(--gray-100)'}`,
                  background: isActive ? 'rgba(10,114,239,0.05)' : 'var(--gray-50)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                  textAlign: 'center',
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
                <span style={{ fontSize: 13, fontWeight: 700, color: isActive ? 'var(--develop-blue)' : 'var(--gray-600)', letterSpacing: '-0.5px' }}>{m.label}</span>
                <span style={{ fontSize: 10, color: 'var(--gray-400)' }}>{m.sub}</span>
              </label>
            )
          })}
        </div>
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

import { useEffect, useMemo, useState } from 'react'
import { fetchModelStatus, ModelPrepState, prepareModels } from '../api/worker'
import { useTaskStore } from '../store/taskStore'

function formatSize(bytes: number) {
  if (!bytes) return '0 B'
  const gb = bytes / 1024 / 1024 / 1024
  if (gb >= 1) return `${gb.toFixed(2)} GB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function StepCard({ title, subtitle, status, size }: { title: string; subtitle: string; status: string; size?: string }) {
  const color = status === 'complete' ? 'var(--success-green)' : status === 'running' ? 'var(--develop-blue)' : status === 'optional' ? '#f59e0b' : 'var(--gray-400)'
  const label = status === 'complete' ? '已就绪' : status === 'running' ? '准备中' : status === 'optional' ? '可选' : '等待中'

  return (
    <div style={{ padding: 16, borderRadius: 10, background: '#fff', border: '1px solid var(--gray-100)', boxShadow: 'var(--shadow-sm)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, marginBottom: 8 }}>
        <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--gray-900)' }}>{title}</span>
        <span style={{ padding: '3px 8px', borderRadius: 999, background: `${color}14`, color, fontSize: 11, fontWeight: 600 }}>{label}</span>
      </div>
      <p style={{ margin: 0, fontSize: 12, color: 'var(--gray-500)', lineHeight: 1.6 }}>{subtitle}</p>
      {size && <p style={{ margin: '8px 0 0', fontSize: 11, color: 'var(--gray-400)' }}>缓存大小：{size}</p>}
    </div>
  )
}

export default function EnvironmentPrep() {
  const { setStage, skipQualityCheck, setSkipQualityCheck, skipLabeling } = useTaskStore()
  const [state, setState] = useState<ModelPrepState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [autoStarted, setAutoStarted] = useState(false)

  const includeMoondream = !skipQualityCheck && !skipLabeling

  const refresh = async () => {
    try {
      setState(await fetchModelStatus())
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : '无法连接 Worker')
    }
  }

  useEffect(() => {
    refresh()
    const interval = setInterval(refresh, 3000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const status = state?.status
    if (!status || autoStarted || skipLabeling || state?.running) return
    if (!status.yolo_world.installed || !status.clip.installed) {
      setAutoStarted(true)
      prepareModels(false)
        .then(setState)
        .catch((e) => setError(e instanceof Error ? e.message : '启动模型准备失败'))
    }
  }, [autoStarted, skipLabeling, state])

  const status = state?.status
  const yoloReady = Boolean(status?.yolo_world.installed)
  const clipReady = Boolean(status?.clip.installed)
  const moondreamReady = Boolean(status?.moondream.installed)
  const requiredReady = skipLabeling || (yoloReady && clipReady)
  const canContinue = requiredReady && (skipQualityCheck || moondreamReady || skipLabeling)

  const progressText = useMemo(() => {
    if (error) return error
    if (state?.error) return state.error
    if (state?.running) return '模型正在后台准备中，可以留在本页观察进度'
    if (canContinue) return '环境已满足当前选择，可以进入数据准备'
    return '请先准备必需模型，或选择跳过 Moondream VQA 质检'
  }, [canContinue, error, state])

  const startPrepare = async () => {
    try {
      setState(await prepareModels(includeMoondream))
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : '启动模型准备失败')
    }
  }

  return (
    <div>
      <div className="page-header" style={{ marginBottom: 28 }}>
        <div className="badge" style={{ background: 'var(--develop-blue)', color: '#fff', fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 12 }}>
          Environment
        </div>
        <h1 className="page-title">环境准备</h1>
        <p className="page-subtitle">在开始自动打标前，先确认本机模型缓存是否就绪，避免用户进入打标后才等待大模型下载。</p>
      </div>

      <div className="card-section" style={{ marginBottom: 20 }}>
        <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-500)', marginBottom: 14, textTransform: 'uppercase', letterSpacing: '0.05em' }}>模型依赖</h3>
        <div style={{ display: 'grid', gap: 12 }}>
          <StepCard
            title="YOLO-World"
            subtitle={state?.steps.yolo_world.message ?? '用于第一段开放词表目标检测'}
            status={state?.steps.yolo_world.status ?? 'pending'}
            size={status ? formatSize(status.yolo_world.worker_path.size_bytes || status.yolo_world.cwd_path.size_bytes) : undefined}
          />
          <StepCard
            title="CLIP ViT-B-32"
            subtitle={state?.steps.clip.message ?? 'YOLO-World 设置类别时需要的文本编码权重'}
            status={state?.steps.clip.status ?? 'pending'}
            size={status ? formatSize(status.clip.cache.size_bytes) : undefined}
          />
          <StepCard
            title="Moondream2"
            subtitle={state?.steps.moondream.message ?? '用于第二段 VQA 质检，体积较大，可按需启用'}
            status={skipQualityCheck ? 'optional' : (state?.steps.moondream.status ?? 'pending')}
            size={status ? formatSize(status.moondream.cache.size_bytes) : undefined}
          />
        </div>
      </div>

      {!skipLabeling && (
        <div className="card-section" style={{ marginBottom: 20 }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-500)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.05em' }}>质检策略</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <button
              className="btn"
              onClick={() => setSkipQualityCheck(true)}
              style={{ justifyContent: 'flex-start', background: skipQualityCheck ? 'var(--success-green)' : '#fff', color: skipQualityCheck ? '#fff' : 'var(--gray-700)', border: `1px solid ${skipQualityCheck ? 'var(--success-green)' : 'var(--gray-200)'}` }}
            >
              跳过 Moondream 质检
            </button>
            <button
              className="btn"
              onClick={() => setSkipQualityCheck(false)}
              style={{ justifyContent: 'flex-start', background: !skipQualityCheck ? 'var(--preview-pink)' : '#fff', color: !skipQualityCheck ? '#fff' : 'var(--gray-700)', border: `1px solid ${!skipQualityCheck ? 'var(--preview-pink)' : 'var(--gray-200)'}` }}
            >
              启用 Moondream 质检
            </button>
          </div>
          <p style={{ fontSize: 12, color: 'var(--gray-500)', lineHeight: 1.7, margin: '12px 0 0' }}>
            默认跳过 VQA 质检，只使用 YOLO-World 初筛，避免首次下载 7GB 级模型阻塞流程；需要更严格数据质量时再启用 Moondream。
          </p>
        </div>
      )}

      <div className="card-section" style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16 }}>
          <div>
            <h3 style={{ fontSize: 14, fontWeight: 600, color: state?.error || error ? 'var(--ship-red)' : 'var(--gray-800)', marginBottom: 4 }}>准备状态</h3>
            <p style={{ fontSize: 12, color: state?.error || error ? 'var(--ship-red)' : 'var(--gray-500)', margin: 0, lineHeight: 1.6 }}>{progressText}</p>
          </div>
          <button className="btn btn-secondary" onClick={startPrepare} disabled={state?.running}>
            {state?.running ? '准备中...' : includeMoondream ? '准备全部模型' : '准备必需模型'}
          </button>
        </div>
      </div>

      <div className="flex gap-3">
        <button className="btn btn-secondary" onClick={() => setStage('algorithm_plan')}>返回能力草案</button>
        <button className="btn btn-primary" onClick={() => setStage('labeling')} disabled={!canContinue}>
          进入数据准备
        </button>
      </div>
    </div>
  )
}

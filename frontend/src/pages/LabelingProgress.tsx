import { useEffect, useRef, useState } from 'react'
import { workerClient } from '../api/worker'
import { useTaskStore } from '../store/taskStore'
import { buildDetectionClass } from '../utils/detectionPrompts'
import GpuMonitor from '../components/GpuMonitor'
import ModelStatus from '../components/ModelStatus'

interface QcStats {
  total_boxes?: number
  passed_boxes?: number
  rejected_boxes?: number
  rejected_too_small?: number
  reject_reasons?: Record<string, number>
  thresholds?: Record<string, number>
  min_confidence?: number
}

interface ClassBalanceItem {
  class: string
  count: number
  level: 'error' | 'warning'
  message: string
}

interface ClassBalance {
  counts: Record<string, number>
  warnings: ClassBalanceItem[]
  errors: ClassBalanceItem[]
  blocked: boolean
}

interface LabelingResult {
  image_count?: number
  raw_box_count?: number
  detected_image_count?: number
  passed_box_count?: number
  quality_filtered_box_count?: number
  labeled_count?: number
  mode?: string
  message?: string
  suggestions?: string[]
  qc_stats?: QcStats | null
  class_balance?: ClassBalance | null
}

const REJECT_REASON_LABEL: Record<string, string> = {
  category_mismatch: '类别不匹配',
  box_too_loose: '框过松',
  object_cut_off: '目标被截断',
  too_blurry: '模糊',
  occluded: '遮挡严重',
  low_overall: '综合分低',
}

export default function LabelingProgress() {
  const { labelingProgress, setStage, setLabeledImageCount, taskId, vlmResult, datasetImages, skipLabeling, skipQualityCheck, labelingImageDir } = useTaskStore()
  const startedRef = useRef(false)
  const [labelingIssue, setLabelingIssue] = useState<LabelingResult | null>(null)
  const [runId, setRunId] = useState(0)

  useEffect(() => {
    workerClient.connect()

    const unsub = workerClient.onMessage((msg) => {
      if (msg.type === 'progress') {
        const phase = (msg.stage as string) === 'quality_check_skipped'
          ? 'quality_check_skipped'
          : (msg.stage as 'detection' | 'quality_check' | 'loading_moondream')

        if (phase === 'quality_check_skipped') {
          useTaskStore.setState({
            labelingProgress: {
              current: labelingProgress.total,
              total: labelingProgress.total,
              phase: 'quality_check',
            },
          })
          return
        }

        useTaskStore.setState({
          labelingProgress: {
            current: msg.current as number,
            total: msg.total as number,
            phase,
          },
        })
      }
      if (msg.type === 'stage_complete' && msg.stage === 'labeling') {
        const result = msg.result as LabelingResult
        const labeledCount = result.labeled_count ?? 0
        setLabeledImageCount(labeledCount)
        if (labeledCount <= 0) {
          setLabelingIssue(result)
          return
        }
        // 类别平衡阻断：任一类样本框 < 50 时不进入增强，保留页面让用户决定
        if (result.class_balance?.blocked) {
          setLabelingIssue(result)
          return
        }
        setLabelingIssue(null)
        setStage('augment')
      }
      if (msg.type === 'error') {
        setLabeledImageCount(0)
        setLabelingIssue({
          message: `打标出错: ${String(msg.message ?? '未知错误')}`,
          suggestions: ['查看 Worker 日志中的原始错误。', '确认模型已安装、图片目录存在，并检查提示词是否与图片目标匹配。'],
        })
      }
    })

    return () => {
      unsub()
    }
  }, [setStage, setLabeledImageCount, labelingProgress.total])

  useEffect(() => {
    if (startedRef.current || !taskId || !vlmResult) return

    startedRef.current = true
    setLabelingIssue(null)
    useTaskStore.setState({
      labelingProgress: {
        current: 0,
        total: datasetImages.length,
        phase: skipLabeling ? 'quality_check' : 'detection',
      },
    })
    workerClient.connect()
    workerClient.startDetection({
      image_dir: labelingImageDir || `../backend/uploads/${taskId}/images`,
      classes: vlmResult.classes.map(buildDetectionClass),
      output_raw_dir: `../backend/uploads/${taskId}/raw`,
      output_label_dir: `../backend/uploads/${taskId}/labels`,
      output_image_dir: `../backend/uploads/${taskId}/labeled_images`,
      conf_threshold: 0.12,
      imgsz: 1280,
      use_existing_labels: skipLabeling,
      skip_quality_check: skipQualityCheck,
    })
  }, [taskId, vlmResult, datasetImages.length, skipLabeling, skipQualityCheck, labelingImageDir, runId])

  const progressPercent =
    labelingProgress.total > 0
      ? Math.round((labelingProgress.current / labelingProgress.total) * 100)
      : 0

  const isDetection = labelingProgress.phase === 'detection'
  const isQualityCheck = labelingProgress.phase === 'quality_check'
  const isLoadingMoondream = labelingProgress.phase === 'loading_moondream'
  const isSkipped = isQualityCheck && (skipLabeling || skipQualityCheck)

  return (
    <div>
      {/* Page Header */}
      <div className="page-header" style={{ marginBottom: 32 }}>
        <div className="flex items-center gap-3 mb-4">
          <div
            className="badge"
            style={{
            background: isSkipped ? 'var(--success-green)' : isLoadingMoondream ? 'var(--preview-pink)' : 'var(--develop-blue)',
            color: '#fff',
            fontSize: 11,
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
          }}
          >
            Stage 2
          </div>
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 8,
              background: isSkipped ? 'var(--success-green)' : isLoadingMoondream ? 'var(--preview-pink)' : 'var(--develop-blue)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {isSkipped ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
            )}
          </div>
        </div>
        <h1 className="page-title">
          {isSkipped ? skipLabeling ? '导入已有标注' : 'YOLO-World 初筛进行中' : isLoadingMoondream ? 'Moondream 模型下载中' : '两段式打标进行中'}
        </h1>
        <p className="page-subtitle">
          {isSkipped
            ? skipLabeling ? '跳过自动打标，直接使用你上传的 YOLO 标注文件' : '跳过 Moondream VQA 质检，仅使用 YOLO-World 初筛结果'
            : isLoadingMoondream
              ? '首次使用需要下载质检模型，预计 1-5 分钟，请保持网络畅通'
              : 'YOLO-World 目标检测 → Moondream VQA 质检，全自动流水线'}
        </p>
      </div>

      {/* GPU Monitor — only show for GPU-based labeling */}
      {!skipLabeling && (
        <div className="card-section" style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-500)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            GPU 状态
          </h3>
          <GpuMonitor />
        </div>
      )}

      {!skipLabeling && (
        <div className="card-section" style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-500)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            模型状态
          </h3>
          <ModelStatus />
        </div>
      )}

      {/* Phase Pipeline */}
      <div className="card-section" style={{ marginBottom: 24 }}>
        {/* Phase Steps */}
        <div style={{ display: 'flex', gap: 0, marginBottom: 28 }}>
          {isSkipped ? (
            // Skipped mode: single step showing reading existing labels
            <>
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
                <div
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: '50%',
                    background: 'var(--success-green)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 0 0 4px rgba(16,185,129,0.15)',
                  }}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <p style={{ fontSize: 13, fontWeight: 600, color: 'var(--success-green)' }}>{skipLabeling ? '标注导入' : 'YOLO 初筛'}</p>
                  <p style={{ fontSize: 11, color: 'var(--gray-400)', marginTop: 2 }}>{skipLabeling ? '读取 YOLO .txt 文件' : '生成候选标注框'}</p>
                </div>
              </div>
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
                <div
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: '50%',
                    background: 'var(--success-green)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 0 0 4px rgba(16,185,129,0.15)',
                  }}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <p style={{ fontSize: 13, fontWeight: 600, color: 'var(--success-green)' }}>VQA 质检</p>
                  <p style={{ fontSize: 11, color: 'var(--gray-400)', marginTop: 2 }}>{skipLabeling ? '已跳过（预标注）' : '已跳过（快速模式）'}</p>
                </div>
              </div>
            </>
          ) : (
            <>
              {[
                { label: 'YOLO-World 检测', sub: '目标框生成', color: 'var(--develop-blue)', done: !isDetection && !isLoadingMoondream },
                { label: 'Moondream 质检', sub: '清晰度/完整性/一致性', color: 'var(--preview-pink)', done: false },
              ].map((step, i) => (
                <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, position: 'relative' }}>
                  {i > 0 && (
                    <div style={{ position: 'absolute', right: '50%', top: 16, width: '100%', height: 2, background: 'var(--gray-100)', zIndex: 0 }} />
                  )}
                  <div
                    style={{
                      width: 32,
                      height: 32,
                      borderRadius: '50%',
                      background: step.done ? step.color : 'var(--gray-100)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      zIndex: 1,
                      transition: 'all 0.3s ease',
                      boxShadow: !step.done ? 'var(--shadow-border)' : `0 0 0 4px ${step.color}22`,
                    }}
                  >
                    {step.done ? (
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                    ) : (isDetection && i === 0) || (isLoadingMoondream && i === 1) ? (
                      <div style={{ width: 8, height: 8, borderRadius: '50%', background: step.color, animation: 'pulse 1.5s ease-in-out infinite' }} />
                    ) : (
                      <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--gray-400)' }}>{i + 1}</span>
                    )}
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <p style={{ fontSize: 13, fontWeight: 600, color: step.done || ((!isDetection && !isLoadingMoondream && i === 0) || (isLoadingMoondream && i === 1)) ? step.color : 'var(--gray-400)' }}>{step.label}</p>
                    <p style={{ fontSize: 11, color: 'var(--gray-400)', marginTop: 2 }}>
                      {isLoadingMoondream && i === 1 ? '模型下载中...' : step.sub}
                    </p>
                  </div>
                </div>
              ))}
            </>
          )}
        </div>

        {/* Current Phase Banner */}
        <div
          style={{
            padding: '12px 16px',
            borderRadius: 8,
            background: isSkipped
              ? 'rgba(16,185,129,0.06)'
              : isLoadingMoondream
                ? 'rgba(222,29,141,0.08)'
                : isDetection
                  ? 'rgba(10,114,239,0.06)'
                  : 'rgba(222,29,141,0.06)',
            border: `1px solid ${isSkipped ? 'rgba(16,185,129,0.15)' : isLoadingMoondream ? 'rgba(222,29,141,0.2)' : isDetection ? 'rgba(10,114,239,0.15)' : 'rgba(222,29,141,0.15)'}`,
            marginBottom: 20,
            display: 'flex',
            alignItems: 'center',
            gap: 10,
          }}
        >
          <div
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: isSkipped ? 'var(--success-green)' : isLoadingMoondream ? 'var(--preview-pink)' : isDetection ? 'var(--develop-blue)' : 'var(--preview-pink)',
              animation: isSkipped || isLoadingMoondream ? 'none' : 'pulse 1.5s ease-in-out infinite',
              flexShrink: 0,
            }}
          />
          <span style={{ fontSize: 13, color: 'var(--gray-700)', fontWeight: 500 }}>
            当前阶段：
            <strong style={{ color: isSkipped ? 'var(--success-green)' : isLoadingMoondream ? 'var(--preview-pink)' : isDetection ? 'var(--develop-blue)' : 'var(--preview-pink)' }}>
              {isSkipped
                ? skipLabeling ? '导入已有标注' : 'YOLO-World 初筛'
                : isLoadingMoondream
                  ? 'Moondream 模型下载中'
                  : isDetection
                    ? 'YOLO-World 目标检测'
                    : 'Moondream VQA 质检'}
            </strong>
          </span>
        </div>

        {/* Progress Bar */}
        <div style={{ marginBottom: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
            <span style={{ fontSize: 13, color: 'var(--gray-600)' }}>
              {isSkipped ? skipLabeling ? '标注文件处理进度' : '初筛处理进度' : isLoadingMoondream ? '模型加载中' : '处理进度'}
            </span>
            <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-900)' }}>
              {isLoadingMoondream ? '下载中...' : `${progressPercent}%`}
            </span>
          </div>
          <div className="progress-bar" style={{ height: 6, borderRadius: 3 }}>
            {isLoadingMoondream ? (
              // Loading state: animated gradient bar
              <div
                className="progress-bar-fill animated"
                style={{
                  width: '100%',
                  background: 'linear-gradient(90deg, var(--preview-pink) 0%, #e879a0 50%, var(--preview-pink) 100%)',
                  backgroundSize: '200% 100%',
                  animation: 'shimmer 1.5s ease-in-out infinite',
                }}
              />
            ) : (
              <div
                className="progress-bar-fill animated"
                style={{
                  width: `${progressPercent}%`,
                  background: isSkipped ? 'var(--success-green)' : isDetection ? 'var(--develop-blue)' : 'var(--preview-pink)',
                }}
              />
            )}
          </div>
        </div>

        {/* Counter */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
          <span style={{ fontSize: 12, color: 'var(--gray-400)' }}>
            {isLoadingMoondream
              ? '首次加载中，请稍候'
              : `${labelingProgress.current} / ${labelingProgress.total} 张图片`}
          </span>
          <span style={{ fontSize: 11, color: 'var(--gray-400)' }}>
            {isSkipped ? skipLabeling ? '直接导入中，请稍候' : '跳过质检，稍后直接进入增强' : isLoadingMoondream ? '下载完成后自动继续' : '自动处理中，请勿关闭页面'}
          </span>
        </div>
      </div>

      {labelingIssue && (
        <div className="card-section" style={{ marginBottom: 24, border: '1px solid rgba(255,91,79,0.25)', background: 'rgba(255,91,79,0.06)' }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12, marginBottom: 16 }}>
            <div style={{ width: 32, height: 32, borderRadius: 8, background: 'var(--ship-red)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
            </div>
            <div>
              <h3 style={{ fontSize: 16, fontWeight: 700, color: 'var(--gray-900)', marginBottom: 6 }}>
                {(labelingIssue.labeled_count ?? 0) <= 0
                  ? '本轮没有生成可训练标注'
                  : '类别样本数量不足，已暂缓进入数据增强'}
              </h3>
              <p style={{ fontSize: 13, color: 'var(--gray-700)', lineHeight: 1.7, margin: 0 }}>
                {labelingIssue.message || ((labelingIssue.labeled_count ?? 0) <= 0
                  ? '打标结果为空，系统已阻止进入数据增强，避免后续使用空数据集训练。'
                  : '至少有一个类别样本框 < 50，按经验直接进入训练大概率欠拟合，请补样本或人工补框。')}
              </p>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 8, marginBottom: 16 }}>
            {[
              ['输入图片', labelingIssue.image_count],
              ['YOLO 候选框', labelingIssue.raw_box_count],
              ['通过质检框', labelingIssue.passed_box_count],
              ['有效标注图', labelingIssue.labeled_count],
            ].map(([label, value]) => (
              <div key={label} style={{ padding: '10px 12px', borderRadius: 8, background: '#fff', border: '1px solid rgba(255,91,79,0.12)' }}>
                <div style={{ fontSize: 11, color: 'var(--gray-400)', marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--gray-900)' }}>{typeof value === 'number' ? value : '--'}</div>
              </div>
            ))}
          </div>

          {typeof labelingIssue.quality_filtered_box_count === 'number' && labelingIssue.quality_filtered_box_count > 0 && (
            <div style={{ marginBottom: 14, padding: '10px 12px', borderRadius: 8, background: '#fff', border: '1px solid rgba(245,158,11,0.25)', color: '#92400e', fontSize: 13 }}>
              Moondream VQA 质检过滤了 {labelingIssue.quality_filtered_box_count} 个候选框。
            </div>
          )}

          {labelingIssue.qc_stats?.reject_reasons && Object.values(labelingIssue.qc_stats.reject_reasons).some((v) => v > 0) && (
            <div style={{ marginBottom: 14, padding: '12px 14px', borderRadius: 8, background: '#fff', border: '1px solid rgba(245,158,11,0.25)' }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-800)', marginBottom: 8 }}>五维度质检拒绝原因分布</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0,1fr))', gap: 8 }}>
                {Object.entries(labelingIssue.qc_stats.reject_reasons).map(([reason, cnt]) => (
                  <div key={reason} style={{ padding: '8px 10px', borderRadius: 6, background: cnt > 0 ? 'rgba(245,158,11,0.08)' : 'var(--gray-50)', border: `1px solid ${cnt > 0 ? 'rgba(245,158,11,0.2)' : 'var(--gray-100)'}` }}>
                    <div style={{ fontSize: 11, color: 'var(--gray-500)' }}>{REJECT_REASON_LABEL[reason] || reason}</div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: cnt > 0 ? '#92400e' : 'var(--gray-400)' }}>{cnt}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {labelingIssue.class_balance && (labelingIssue.class_balance.errors.length > 0 || labelingIssue.class_balance.warnings.length > 0) && (
            <div style={{ marginBottom: 14, padding: '12px 14px', borderRadius: 8, background: '#fff', border: '1px solid rgba(255,91,79,0.2)' }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-800)', marginBottom: 8 }}>类别平衡检查</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, marginBottom: 10 }}>
                {Object.entries(labelingIssue.class_balance.counts).map(([name, cnt]) => (
                  <div key={name} style={{ padding: '6px 10px', borderRadius: 6, background: 'var(--gray-50)', border: '1px solid var(--gray-100)' }}>
                    <div style={{ fontSize: 11, color: 'var(--gray-500)' }}>{name}</div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: cnt < 50 ? 'var(--ship-red)' : 'var(--gray-900)' }}>{cnt}</div>
                  </div>
                ))}
              </div>
              <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, lineHeight: 1.7 }}>
                {labelingIssue.class_balance.errors.map((it) => (
                  <li key={`e-${it.class}`} style={{ color: 'var(--ship-red)' }}>🔴 {it.message}</li>
                ))}
                {labelingIssue.class_balance.warnings.map((it) => (
                  <li key={`w-${it.class}`} style={{ color: '#92400e' }}>⚠️ {it.message}</li>
                ))}
              </ul>
            </div>
          )}

          {(labelingIssue.suggestions ?? []).length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-700)', marginBottom: 8 }}>建议调整</div>
              <ul style={{ margin: 0, paddingLeft: 18, color: 'var(--gray-600)', fontSize: 13, lineHeight: 1.8 }}>
                {labelingIssue.suggestions!.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="flex gap-3">
            <button className="btn btn-secondary" onClick={() => setStage('intent_confirm')}>
              返回调整需求
            </button>
            {!skipQualityCheck && (labelingIssue.raw_box_count ?? 0) > 0 && (labelingIssue.passed_box_count ?? 0) <= 0 && (
              <button className="btn btn-primary" onClick={() => setStage('environment')}>
                跳过 VQA 重试
              </button>
            )}
            {(labelingIssue.labeled_count ?? 0) > 0 && labelingIssue.class_balance?.blocked && (
              <button
                className="btn btn-primary"
                onClick={() => {
                  setLabelingIssue(null)
                  setStage('augment')
                }}
                title="忽略类别样本不足警告，强制进入数据增强（不推荐）"
              >
                忽略警告继续
              </button>
            )}
            <button
              className="btn btn-ghost"
              onClick={() => {
                startedRef.current = false
                setRunId((value) => value + 1)
              }}
            >
              重新打标
            </button>
          </div>
        </div>
      )}

      {/* Cancel */}
      <div className="flex gap-3">
        <button
          onClick={() => {
            workerClient.cancel()
            setStage('upload')
          }}
          className="btn btn-danger"
          style={{ padding: '10px 20px' }}
        >
          取消
        </button>
      </div>
    </div>
  )
}

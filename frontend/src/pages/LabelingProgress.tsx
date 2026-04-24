import { useEffect, useRef } from 'react'
import { useTaskStore } from '../store/taskStore'
import { workerClient } from '../api/worker'
import GpuMonitor from '../components/GpuMonitor'

export default function LabelingProgress() {
  const { labelingProgress, setStage, setLabeledImageCount, taskId, vlmResult, datasetImages, skipLabeling } = useTaskStore()
  const startedRef = useRef(false)

  useEffect(() => {
    workerClient.connect()

    const unsub = workerClient.onMessage((msg) => {
      if (msg.type === 'progress') {
        const phase = (msg.stage as string) === 'quality_check_skipped'
          ? 'quality_check_skipped'
          : (msg.stage as 'detection' | 'quality_check')

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
        const result = msg.result as { labeled_count?: number }
        setLabeledImageCount(result.labeled_count ?? 0)
        setStage('augment')
      }
      if (msg.type === 'error') {
        alert(`打标出错: ${msg.message}`)
      }
    })

    return () => {
      unsub()
    }
  }, [setStage, setLabeledImageCount, labelingProgress.total])

  useEffect(() => {
    if (startedRef.current || !taskId || !vlmResult) return

    startedRef.current = true
    useTaskStore.setState({
      labelingProgress: {
        current: 0,
        total: datasetImages.length,
        phase: skipLabeling ? 'quality_check' : 'detection',
      },
    })
    workerClient.connect()
    workerClient.startDetection({
      image_dir: `../backend/uploads/${taskId}/images`,
      classes: vlmResult.classes.map((cls) => ({
        class_name: cls.class_name,
        prompt: cls.prompt,
        negative_prompt: cls.negative_prompt,
        color_hint: cls.color_hint,
      })),
      output_raw_dir: `../backend/uploads/${taskId}/raw`,
      output_label_dir: `../backend/uploads/${taskId}/labels`,
      output_image_dir: `../backend/uploads/${taskId}/labeled_images`,
      use_existing_labels: skipLabeling,
    })
  }, [taskId, vlmResult, datasetImages.length, skipLabeling])

  const progressPercent =
    labelingProgress.total > 0
      ? Math.round((labelingProgress.current / labelingProgress.total) * 100)
      : 0

  const isDetection = labelingProgress.phase === 'detection'
  const isSkipped = labelingProgress.phase === 'quality_check' && skipLabeling

  return (
    <div>
      {/* Page Header */}
      <div className="page-header" style={{ marginBottom: 32 }}>
        <div className="flex items-center gap-3 mb-4">
          <div
            className="badge"
            style={{
              background: isSkipped ? 'var(--success-green)' : 'var(--develop-blue)',
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
              background: isSkipped ? 'var(--success-green)' : 'var(--develop-blue)',
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
          {isSkipped ? '导入已有标注' : '两段式打标进行中'}
        </h1>
        <p className="page-subtitle">
          {isSkipped
            ? '跳过自动打标，直接使用你上传的 YOLO 标注文件'
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
                  <p style={{ fontSize: 13, fontWeight: 600, color: 'var(--success-green)' }}>标注导入</p>
                  <p style={{ fontSize: 11, color: 'var(--gray-400)', marginTop: 2 }}>读取 YOLO .txt 文件</p>
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
                  <p style={{ fontSize: 11, color: 'var(--gray-400)', marginTop: 2 }}>已跳过（预标注）</p>
                </div>
              </div>
            </>
          ) : (
            <>
              {[
                { label: 'YOLO-World 检测', sub: '目标框生成', color: 'var(--develop-blue)', done: !isDetection },
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
                    ) : isDetection && i === 0 ? (
                      <div style={{ width: 8, height: 8, borderRadius: '50%', background: step.color, animation: 'pulse 1.5s ease-in-out infinite' }} />
                    ) : (
                      <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--gray-400)' }}>{i + 1}</span>
                    )}
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <p style={{ fontSize: 13, fontWeight: 600, color: step.done || (!isDetection && i === 0) ? step.color : 'var(--gray-400)' }}>{step.label}</p>
                    <p style={{ fontSize: 11, color: 'var(--gray-400)', marginTop: 2 }}>{step.sub}</p>
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
              : isDetection
                ? 'rgba(10,114,239,0.06)'
                : 'rgba(222,29,141,0.06)',
            border: `1px solid ${isSkipped ? 'rgba(16,185,129,0.15)' : isDetection ? 'rgba(10,114,239,0.15)' : 'rgba(222,29,141,0.15)'}`,
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
              background: isSkipped ? 'var(--success-green)' : isDetection ? 'var(--develop-blue)' : 'var(--preview-pink)',
              animation: isSkipped ? 'none' : 'pulse 1.5s ease-in-out infinite',
              flexShrink: 0,
            }}
          />
          <span style={{ fontSize: 13, color: 'var(--gray-700)', fontWeight: 500 }}>
            当前阶段：
            <strong style={{ color: isSkipped ? 'var(--success-green)' : isDetection ? 'var(--develop-blue)' : 'var(--preview-pink)' }}>
              {isSkipped
                ? '导入已有标注'
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
              {isSkipped ? '标注文件处理进度' : '处理进度'}
            </span>
            <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-900)' }}>{progressPercent}%</span>
          </div>
          <div className="progress-bar" style={{ height: 6, borderRadius: 3 }}>
            <div
              className="progress-bar-fill animated"
              style={{
                width: `${progressPercent}%`,
                background: isSkipped ? 'var(--success-green)' : isDetection ? 'var(--develop-blue)' : 'var(--preview-pink)',
              }}
            />
          </div>
        </div>

        {/* Counter */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
          <span style={{ fontSize: 12, color: 'var(--gray-400)' }}>
            {labelingProgress.current} / {labelingProgress.total} 张图片
          </span>
          <span style={{ fontSize: 11, color: 'var(--gray-400)' }}>
            {isSkipped ? '直接导入中，请稍候' : '自动处理中，请勿关闭页面'}
          </span>
        </div>
      </div>

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

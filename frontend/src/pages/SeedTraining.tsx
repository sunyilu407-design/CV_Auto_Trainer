import { useState, useEffect, useRef } from 'react'
import { useTaskStore } from '../store/taskStore'

type TrainPhase = 'idle' | 'preparing' | 'training' | 'auto_labeling' | 'done' | 'error'

export default function SeedTraining() {
  const {
    taskId,
    setStage,
    setSeedModelPath,
    setSeedModelMap,
    setSeedAutoLabelStats,
    setLabeledImageCount,
    seedAnnotatedCount,
  } = useTaskStore()

  const [phase, setPhase] = useState<TrainPhase>('idle')
  const [progress, setProgress] = useState({ epoch: 0, totalEpochs: 50, map50: 0 })
  const [autoLabelProgress, setAutoLabelProgress] = useState({ current: 0, total: 0 })
  const [error, setError] = useState<string | null>(null)
  const [highConf, setHighConf] = useState(0.5)
  const [lowConf, setLowConf] = useState(0.25)
  const wsRef = useRef<WebSocket | null>(null)

  const startTraining = () => {
    if (!taskId) return
    setPhase('preparing')
    setError(null)

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws/${taskId}`)
    wsRef.current = ws

    ws.onopen = () => {
      const classNames = useTaskStore.getState().vlmResult?.classes.map(
        (c) => c.class_name,
      ) || []
      ws.send(
        JSON.stringify({
          type: 'start_seed_training',
          payload: {
            task_id: taskId,
            class_names: classNames,
            high_conf: highConf,
            low_conf: lowConf,
          },
        }),
      )
    }

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        switch (msg.type) {
          case 'seed_training_started':
            setPhase('training')
            break
          case 'seed_training_progress':
            setProgress({
              epoch: msg.currentEpoch || 0,
              totalEpochs: msg.totalEpochs || 50,
              map50: msg.currentMap || 0,
            })
            break
          case 'seed_training_complete':
            setSeedModelPath(msg.seed_model_path)
            setSeedModelMap(msg.best_map)
            setPhase('auto_labeling')
            break
          case 'seed_auto_label_progress':
            setAutoLabelProgress({
              current: msg.current || 0,
              total: msg.total || 0,
            })
            break
          case 'seed_auto_label_complete':
            setSeedAutoLabelStats({
              autoAccepted: msg.auto_accepted || 0,
              needsReview: msg.needs_review || 0,
              noDetection: msg.no_detection || 0,
              avgConfidence: msg.avg_confidence || 0,
            })
            setLabeledImageCount(msg.total_labeled || 0)
            setPhase('done')
            break
          case 'error':
            setError(msg.message || 'Unknown error')
            setPhase('error')
            break
        }
      } catch {
        // ignore
      }
    }

    ws.onerror = () => {
      setError('WebSocket connection failed')
      setPhase('error')
    }
    ws.onclose = () => {
      wsRef.current = null
    }
  }

  useEffect(() => {
    return () => {
      wsRef.current?.close()
    }
  }, [])

  const progressPct =
    phase === 'training' && progress.totalEpochs > 0
      ? Math.round((progress.epoch / progress.totalEpochs) * 100)
      : 0

  const autoLabelPct =
    phase === 'auto_labeling' && autoLabelProgress.total > 0
      ? Math.round((autoLabelProgress.current / autoLabelProgress.total) * 100)
      : 0

  return (
    <div style={{ maxWidth: 600, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>Seed Training</h2>
      <p style={{ fontSize: 13, color: 'var(--gray-500)', margin: '0 0 24px' }}>
        Training a lightweight model on your {seedAnnotatedCount} annotated images, then auto-labeling the rest.
      </p>

      {/* Status card */}
      <div
        style={{
          background: '#fff',
          borderRadius: 12,
          border: '1px solid var(--gray-100)',
          padding: 24,
          marginBottom: 16,
        }}
      >
        {phase === 'idle' && (
          <div style={{ textAlign: 'center' }}>
            {/* Confidence threshold controls */}
            <div style={{ background: 'var(--gray-50)', borderRadius: 8, padding: '14px 16px', marginBottom: 16, textAlign: 'left' }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--gray-600)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                自动标注置信度阈值
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 12 }}>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--gray-500)', display: 'block', marginBottom: 4 }}>
                    自动采纳阈值 <span style={{ color: '#16a34a', fontWeight: 700 }}>≥ {highConf}</span>
                  </label>
                  <input
                    type="range" min="0.1" max="0.9" step="0.05"
                    value={highConf}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value)
                      setHighConf(v)
                      if (v < lowConf) setLowConf(v)
                    }}
                    style={{ width: '100%', accentColor: '#16a34a' }}
                  />
                  <div style={{ fontSize: 10, color: 'var(--gray-400)', marginTop: 2 }}>高置信 → 直接采纳</div>
                </div>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--gray-500)', display: 'block', marginBottom: 4 }}>
                    待审核阈值 <span style={{ color: '#f59e0b', fontWeight: 700 }}>≥ {lowConf}</span>
                  </label>
                  <input
                    type="range" min="0.05" max="0.8" step="0.05"
                    value={lowConf}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value)
                      setLowConf(v)
                      if (v > highConf) setHighConf(v)
                    }}
                    style={{ width: '100%', accentColor: '#f59e0b' }}
                  />
                  <div style={{ fontSize: 10, color: 'var(--gray-400)', marginTop: 2 }}>低置信 → 需人工审核</div>
                </div>
              </div>
              <div style={{ fontSize: 11, color: 'var(--gray-400)', lineHeight: 1.6 }}>
                阈值越低 → 自动采纳越多，审核工作量越大
              </div>
            </div>
            <p style={{ marginBottom: 16, color: 'var(--gray-600)', fontSize: 14 }}>
              已标注 {seedAnnotatedCount} 张图，将训练 yolov8n 后对剩余图片自动打标
            </p>
            <button className="btn btn-primary" onClick={startTraining}>
              Start Seed Training
            </button>
          </div>
        )}

        {phase === 'preparing' && (
          <div style={{ textAlign: 'center', color: 'var(--gray-500)' }}>
            Preparing dataset...
          </div>
        )}

        {phase === 'training' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, fontSize: 13 }}>
              <span>Training: Epoch {progress.epoch}/{progress.totalEpochs}</span>
              <span style={{ fontWeight: 600 }}>mAP50: {(progress.map50 * 100).toFixed(1)}%</span>
            </div>
            <div
              style={{
                height: 8,
                background: 'var(--gray-100)',
                borderRadius: 4,
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  height: '100%',
                  width: `${progressPct}%`,
                  background: '#0a72ef',
                  borderRadius: 4,
                  transition: 'width 0.3s',
                }}
              />
            </div>
          </div>
        )}

        {phase === 'auto_labeling' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, fontSize: 13 }}>
              <span>Auto-labeling remaining images...</span>
              <span>{autoLabelProgress.current}/{autoLabelProgress.total}</span>
            </div>
            <div
              style={{
                height: 8,
                background: 'var(--gray-100)',
                borderRadius: 4,
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  height: '100%',
                  width: `${autoLabelPct}%`,
                  background: '#10b981',
                  borderRadius: 4,
                  transition: 'width 0.3s',
                }}
              />
            </div>
          </div>
        )}

        {phase === 'done' && (
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 32, marginBottom: 8 }}>✓</div>
            <p style={{ fontWeight: 600, marginBottom: 4 }}>Seed training & auto-labeling complete</p>
            {(() => {
              const stats = useTaskStore.getState().seedAutoLabelStats
              return (
                <div style={{ marginTop: 12, padding: '12px 16px', borderRadius: 8, background: 'var(--gray-50)', textAlign: 'left', fontSize: 13, color: 'var(--gray-600)' }}>
                  <div style={{ marginBottom: 8 }}>
                    <span style={{ color: '#16a34a', fontWeight: 700 }}>✓ 自动采纳 {stats?.autoAccepted ?? 0}</span>
                    {' · '}
                    <span style={{ color: '#f59e0b', fontWeight: 700 }}>待审核 {stats?.needsReview ?? 0}</span>
                    {' · '}
                    <span>无检测 {stats?.noDetection ?? 0}</span>
                  </div>
                  <div>阈值：采纳 ≥ {highConf} · 待审 ≥ {lowConf}</div>
                </div>
              )
            })()}
            {(useTaskStore.getState().seedAutoLabelStats?.needsReview ?? 0) > 0 ? (
              <p style={{ fontSize: 13, color: '#f59e0b', marginTop: 8 }}>
                {(useTaskStore.getState().seedAutoLabelStats?.needsReview ?? 0)} 个低置信框需要审核。
              </p>
            ) : (
              <p style={{ fontSize: 13, color: 'var(--gray-500)', marginTop: 8 }}>
                Proceeding to data augmentation and full training pipeline.
              </p>
            )}
          </div>
        )}

        {phase === 'error' && (
          <div style={{ textAlign: 'center' }}>
            <p style={{ color: '#ef4444', fontWeight: 600, marginBottom: 12 }}>{error}</p>
            <button className="btn btn-secondary" onClick={startTraining}>
              Retry
            </button>
          </div>
        )}
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <button
          className="btn btn-secondary btn-sm"
          onClick={() => setStage('manual_annotation')}
          disabled={phase === 'training' || phase === 'auto_labeling'}
        >
          Back to Annotation
        </button>
        {phase === 'done' && (
          <div style={{ display: 'flex', gap: 8 }}>
            {useTaskStore.getState().seedAutoLabelStats?.needsReview ? (
              <button className="btn btn-secondary btn-sm" onClick={() => setStage('review_auto_labels')} style={{ borderColor: '#f59e0b', color: '#f59e0b' }}>
                Review Low-Confidence Boxes
              </button>
            ) : null}
            <button className="btn btn-primary btn-sm" onClick={() => setStage('augment')}>
              Continue to Augmentation →
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

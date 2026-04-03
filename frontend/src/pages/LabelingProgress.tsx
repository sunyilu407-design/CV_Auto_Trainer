import { useEffect } from 'react'
import { useTaskStore } from '../store/taskStore'
import { workerClient } from '../api/worker'
import GpuMonitor from '../components/GpuMonitor'

export default function LabelingProgress() {
  const { labelingProgress, setStage, setLabeledImageCount } = useTaskStore()

  useEffect(() => {
    workerClient.connect()

    const unsub = workerClient.onMessage((msg) => {
      if (msg.type === 'progress') {
        const phase = msg.stage as 'detection' | 'quality_check'
        useTaskStore.setState({
          labelingProgress: {
            current: msg.current as number,
            total: msg.total as number,
            phase,
          },
        })
      }
      if (msg.type === 'stage_complete' && msg.stage === 'labeling') {
        setLabeledImageCount(msg.result as unknown as number)
        setStage('augment')
      }
      if (msg.type === 'error') {
        alert(`打标出错: ${msg.message}`)
      }
    })

    return () => {
      unsub()
    }
  }, [setStage, setLabeledImageCount])

  const progressPercent =
    labelingProgress.total > 0
      ? Math.round((labelingProgress.current / labelingProgress.total) * 100)
      : 0

  return (
    <div style={{ maxWidth: '600px', margin: '0 auto', textAlign: 'center' }}>
      <h2 style={{ fontSize: '20px', marginBottom: '24px' }}>阶段二：两段式打标进行中</h2>

      <div
        style={{
          background: '#fff',
          borderRadius: '8px',
          padding: '24px',
          marginBottom: '24px',
        }}
      >
        <GpuMonitor />
      </div>

      <div style={{ background: '#fff', borderRadius: '8px', padding: '24px' }}>
        <p style={{ fontSize: '16px', marginBottom: '16px', color: '#333' }}>
          当前阶段：
          <strong>
            {labelingProgress.phase === 'detection' ? 'YOLO-World 目标检测' : 'Moondream VQA 质检'}
          </strong>
        </p>

        <div
          style={{
            width: '100%',
            height: '24px',
            background: '#e0e0e0',
            borderRadius: '12px',
            overflow: 'hidden',
            marginBottom: '12px',
          }}
        >
          <div
            style={{
              width: `${progressPercent}%`,
              height: '100%',
              background: '#1976d2',
              transition: 'width 0.3s',
            }}
          />
        </div>

        <p style={{ fontSize: '14px', color: '#666' }}>
          {labelingProgress.current} / {labelingProgress.total} 张图片 ({progressPercent}%)
        </p>
      </div>

      <button
        onClick={() => {
          workerClient.cancel()
          setStage('upload')
        }}
        style={{
          marginTop: '24px',
          padding: '10px 24px',
          background: '#fff',
          border: '1px solid #d32f2f',
          color: '#d32f2f',
          borderRadius: '4px',
          cursor: 'pointer',
          fontSize: '15px',
        }}
      >
        取消打标
      </button>
    </div>
  )
}

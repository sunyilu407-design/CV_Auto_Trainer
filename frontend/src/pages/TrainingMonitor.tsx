import { useEffect } from 'react'
import { useTaskStore } from '../store/taskStore'
import { workerClient } from '../api/worker'
import { trainingApi } from '../api/backend'
import MetricsChart from '../components/MetricsChart'

export default function TrainingMonitor() {
  const { trainingProgress, trainConfig, setStage } = useTaskStore()

  useEffect(() => {
    if (trainConfig.trainMode === 'cloud') {
      // 云端训练：轮询后端状态
      const poll = setInterval(async () => {
        try {
          const status = await trainingApi.getStatus('current-task-id')
          useTaskStore.setState({
            trainingProgress: {
              state: status.state,
              currentEpoch: status.current_epoch,
              totalEpochs: status.total_epochs,
              currentMap: status.current_map,
              startedAt: new Date().toISOString(),
            },
          })
          if (status.state === 'done') {
            clearInterval(poll)
            setStage('delivery')
          }
        } catch {
          // ignore
        }
      }, 5000)
      return () => clearInterval(poll)
    } else {
      // 本地训练：WebSocket 推送已在 TrainConfig 启动
    }
  }, [trainConfig.trainMode, setStage])

  const handleCancel = async () => {
    if (trainConfig.trainMode === 'local') {
      workerClient.cancel()
    } else {
      await trainingApi.cancel('current-task-id')
    }
    setStage('train_config')
  }

  const progressPercent = trainingProgress
    ? Math.round((trainingProgress.currentEpoch / trainingProgress.totalEpochs) * 100)
    : 0

  return (
    <div style={{ maxWidth: '700px', margin: '0 auto' }}>
      <h2 style={{ fontSize: '20px', marginBottom: '20px' }}>
        阶段四：{trainConfig.trainMode === 'local' ? '本地训练' : '云端训练'}进行中
      </h2>

      <div style={{ background: '#fff', borderRadius: '8px', padding: '24px', marginBottom: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
          <span style={{ fontSize: '16px' }}>训练进度</span>
          <span style={{ fontSize: '16px', fontWeight: 600 }}>{progressPercent}%</span>
        </div>

        <div style={{ width: '100%', height: '20px', background: '#e0e0e0', borderRadius: '10px', overflow: 'hidden', marginBottom: '16px' }}>
          <div
            style={{
              width: `${progressPercent}%`,
              height: '100%',
              background: '#1976d2',
              transition: 'width 0.5s',
            }}
          />
        </div>

        <div style={{ display: 'flex', gap: '32px' }}>
          <div>
            <span style={{ fontSize: '12px', color: '#666' }}>当前 Epoch</span>
            <p style={{ fontSize: '20px', fontWeight: 600, margin: 0 }}>
              {trainingProgress?.currentEpoch ?? 0} / {trainingProgress?.totalEpochs ?? trainConfig.epochs}
            </p>
          </div>
          <div>
            <span style={{ fontSize: '12px', color: '#666' }}>mAP50</span>
            <p style={{ fontSize: '20px', fontWeight: 600, margin: 0 }}>
              {((trainingProgress?.currentMap ?? 0) * 100).toFixed(1)}%
            </p>
          </div>
          <div>
            <span style={{ fontSize: '12px', color: '#666' }}>训练模式</span>
            <p style={{ fontSize: '20px', fontWeight: 600, margin: 0 }}>
              {trainConfig.trainMode === 'local' ? '本地 GPU' : '云端'}
            </p>
          </div>
        </div>
      </div>

      <MetricsChart currentEpoch={trainingProgress?.currentEpoch ?? 0} currentMap={trainingProgress?.currentMap ?? 0} />

      <button
        onClick={handleCancel}
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
        取消训练
      </button>
    </div>
  )
}

import { useTaskStore } from '../store/taskStore'
import { useSettingsStore } from '../store/settingsStore'
import { trainingApi } from '../api/backend'
import { workerClient } from '../api/worker'

const MODELS = [
  { value: 'yolo11n.pt', label: 'YOLO11n（极限压缩，推荐嵌入式）' },
  { value: 'yolo11s.pt', label: 'YOLO11s（推荐，默认）' },
  { value: 'yolo11m.pt', label: 'YOLO11m（工控机/服务器）' },
  { value: 'yolo11l.pt', label: 'YOLO11l（高精度）' },
  { value: 'rtdetr-l.pt', label: 'RT-DETR-L（密集/遮挡场景）' },
]

export default function TrainConfig() {
  const { trainConfig, setTrainConfig, setStage, setTrainingProgress, setStage: setAppStage } = useTaskStore()
  const { settings } = useSettingsStore()

  const handleStartTraining = async () => {
    setStage('training')
    setTrainingProgress({
      state: 'starting',
      currentEpoch: 0,
      totalEpochs: trainConfig.epochs,
      currentMap: 0,
      startedAt: new Date().toISOString(),
    })

    if (trainConfig.trainMode === 'local') {
      // 本地训练：Worker 通过 WebSocket 启动
      workerClient.connect()
      workerClient.onMessage((msg) => {
        if (msg.type === 'training_progress') {
          setTrainingProgress({
            state: 'training',
            currentEpoch: msg.currentEpoch as number,
            totalEpochs: msg.totalEpochs as number,
            currentMap: msg.currentMap as number,
            startedAt: new Date().toISOString(),
          })
        }
        if (msg.type === 'training_complete') {
          setAppStage('delivery')
        }
        if (msg.type === 'training_error') {
          alert(`训练出错: ${msg.message}`)
        }
      })
      workerClient.startLocalTraining({
        dataset_dir: '/path/to/dataset',
        train_config: trainConfig as unknown as Record<string, unknown>,
      })
    } else {
      // 云端训练：调用后端 API
      try {
        await trainingApi.start({
          task_id: 'current-task-id',
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
          resume_last: false,
        })
      } catch (e) {
        alert(`启动失败: ${e}`)
        setStage('train_config')
      }
    }
  }

  return (
    <div style={{ maxWidth: '700px', margin: '0 auto' }}>
      <h2 style={{ fontSize: '20px', marginBottom: '20px' }}>阶段三：训练配置</h2>

      <div style={{ background: '#fff', borderRadius: '8px', padding: '20px', marginBottom: '20px' }}>
        {/* 训练模式选择 */}
        <div style={{ marginBottom: '24px' }}>
          <h3 style={{ fontSize: '16px', marginBottom: '12px' }}>训练模式</h3>
          <div style={{ display: 'flex', gap: '12px' }}>
            <label
              style={{
                flex: 1,
                padding: '12px',
                border: `2px solid ${trainConfig.trainMode === 'local' ? '#1976d2' : '#e0e0e0'}`,
                borderRadius: '8px',
                cursor: 'pointer',
                background: trainConfig.trainMode === 'local' ? '#e3f2fd' : '#fff',
              }}
            >
              <input
                type="radio"
                name="trainMode"
                value="local"
                checked={trainConfig.trainMode === 'local'}
                onChange={() => setTrainConfig({ trainMode: 'local' })}
                style={{ marginRight: '8px' }}
              />
              <strong>本地训练</strong>
              <p style={{ fontSize: '12px', color: '#666', margin: '4px 0 0 20px' }}>
                使用本机 GPU，无需上传数据，适合中小数据集
              </p>
            </label>
            <label
              style={{
                flex: 1,
                padding: '12px',
                border: `2px solid ${trainConfig.trainMode === 'cloud' ? '#1976d2' : '#e0e0e0'}`,
                borderRadius: '8px',
                cursor: 'pointer',
                background: trainConfig.trainMode === 'cloud' ? '#e3f2fd' : '#fff',
              }}
            >
              <input
                type="radio"
                name="trainMode"
                value="cloud"
                checked={trainConfig.trainMode === 'cloud'}
                onChange={() => setTrainConfig({ trainMode: 'cloud' })}
                style={{ marginRight: '8px' }}
              />
              <strong>云端训练</strong>
              <p style={{ fontSize: '12px', color: '#666', margin: '4px 0 0 20px' }}>
                使用云服务器（{settings.cloudProvider === 'autodl' ? 'AutoDL' : 'SSH 云服务器'}），适合大规模数据
              </p>
            </label>
          </div>
        </div>

        {/* 模型选择 */}
        <div style={{ marginBottom: '20px' }}>
          <h3 style={{ fontSize: '16px', marginBottom: '12px' }}>模型选择</h3>
          <select
            value={trainConfig.model}
            onChange={(e) => setTrainConfig({ model: e.target.value })}
            style={{
              width: '100%',
              padding: '8px 12px',
              border: '1px solid #ddd',
              borderRadius: '4px',
              fontSize: '14px',
            }}
          >
            {MODELS.map((m) => (
              <option key={m.value} value={m.value}>{m.label}</option>
            ))}
          </select>
        </div>

        {/* 超参数 */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
          <div>
            <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>Epochs（训练轮次）</label>
            <input
              type="number"
              min={10}
              max={500}
              value={trainConfig.epochs}
              onChange={(e) => setTrainConfig({ epochs: Number(e.target.value) })}
              style={{ width: '100%', padding: '6px 10px', border: '1px solid #ddd', borderRadius: '4px' }}
            />
          </div>
          <div>
            <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>Image Size（图像尺寸）</label>
            <select
              value={trainConfig.imgsz}
              onChange={(e) => setTrainConfig({ imgsz: Number(e.target.value) })}
              style={{ width: '100%', padding: '6px 10px', border: '1px solid #ddd', borderRadius: '4px' }}
            >
              {[416, 512, 640, 1280].map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
          <div>
            <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>学习率 lr0</label>
            <input
              type="number"
              min={0.0001}
              max={0.1}
              step={0.001}
              value={trainConfig.lr0}
              onChange={(e) => setTrainConfig({ lr0: Number(e.target.value) })}
              style={{ width: '100%', padding: '6px 10px', border: '1px solid #ddd', borderRadius: '4px' }}
            />
          </div>
          <div>
            <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>Patience（早停）</label>
            <input
              type="number"
              min={5}
              max={100}
              value={trainConfig.patience}
              onChange={(e) => setTrainConfig({ patience: Number(e.target.value) })}
              style={{ width: '100%', padding: '6px 10px', border: '1px solid #ddd', borderRadius: '4px' }}
            />
          </div>
          <div>
            <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>推理置信度</label>
            <input
              type="number"
              min={0.1}
              max={0.9}
              step={0.05}
              value={trainConfig.conf}
              onChange={(e) => setTrainConfig({ conf: Number(e.target.value) })}
              style={{ width: '100%', padding: '6px 10px', border: '1px solid #ddd', borderRadius: '4px' }}
            />
          </div>
          <div>
            <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>NMS IoU</label>
            <input
              type="number"
              min={0.3}
              max={0.9}
              step={0.05}
              value={trainConfig.iou}
              onChange={(e) => setTrainConfig({ iou: Number(e.target.value) })}
              style={{ width: '100%', padding: '6px 10px', border: '1px solid #ddd', borderRadius: '4px' }}
            />
          </div>
        </div>

        {/* 导出格式 */}
        <div style={{ marginBottom: '20px' }}>
          <h3 style={{ fontSize: '16px', marginBottom: '8px' }}>导出格式</h3>
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            {(['onnx', 'engine', 'coreml', 'openvino'] as const).map((fmt) => (
              <label key={fmt} style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={trainConfig.exportFormats.includes(fmt)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setTrainConfig({ exportFormats: [...trainConfig.exportFormats, fmt] })
                    } else {
                      setTrainConfig({ exportFormats: trainConfig.exportFormats.filter((f) => f !== fmt) })
                    }
                  }}
                />
                <span style={{ fontSize: '14px' }}>{fmt.toUpperCase()}</span>
              </label>
            ))}
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '12px' }}>
        <button
          onClick={handleStartTraining}
          style={{
            padding: '12px 28px',
            background: '#1976d2',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '16px',
            fontWeight: 600,
          }}
        >
          {trainConfig.trainMode === 'local' ? '开始本地训练' : '开始云端训练'}
        </button>
        <button
          onClick={() => setStage('review')}
          style={{
            padding: '12px 24px',
            background: '#fff',
            border: '1px solid #ccc',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '15px',
          }}
        >
          返回
        </button>
      </div>
    </div>
  )
}

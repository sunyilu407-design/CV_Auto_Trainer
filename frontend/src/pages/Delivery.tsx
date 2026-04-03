import { useTaskStore } from '../store/taskStore'
import { filesApi } from '../api/backend'

export default function Delivery() {
  const { trainConfig, reset } = useTaskStore()

  const handleDownload = (filename: string) => {
    const taskId = 'current-task-id'
    const url = filesApi.downloadArtifact(taskId, filename)
    window.open(url, '_blank')
  }

  return (
    <div style={{ maxWidth: '700px', margin: '0 auto' }}>
      <h2 style={{ fontSize: '20px', marginBottom: '20px' }}>训练完成 - 模型交付</h2>

      <div
        style={{
          background: '#e8f5e9',
          border: '1px solid #4caf50',
          borderRadius: '8px',
          padding: '16px 20px',
          marginBottom: '24px',
        }}
      >
        <p style={{ fontSize: '16px', color: '#2e7d32', margin: 0 }}>
          恭喜！模型训练完成，以下是训练产物：
        </p>
      </div>

      <div style={{ background: '#fff', borderRadius: '8px', padding: '20px', marginBottom: '24px' }}>
        <h3 style={{ fontSize: '16px', marginBottom: '16px' }}>模型文件</h3>
        {[
          { key: 'best.pt', label: '最佳权重（best.pt）', desc: '验证集上 mAP 最高的模型' },
          { key: 'last.pt', label: '最终权重（last.pt）', desc: '最后一个 epoch 的权重' },
        ].map(({ key, label, desc }) => (
          <div
            key={key}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '12px 0',
              borderBottom: '1px solid #eee',
            }}
          >
            <div>
              <p style={{ fontWeight: 600, margin: '0 0 4px', fontSize: '14px' }}>{label}</p>
              <p style={{ fontSize: '12px', color: '#666', margin: 0 }}>{desc}</p>
            </div>
            <button
              onClick={() => handleDownload(key)}
              style={{
                padding: '6px 16px',
                background: '#1976d2',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '13px',
              }}
            >
              下载
            </button>
          </div>
        ))}
      </div>

      {trainConfig.exportFormats.length > 0 && (
        <div style={{ background: '#fff', borderRadius: '8px', padding: '20px', marginBottom: '24px' }}>
          <h3 style={{ fontSize: '16px', marginBottom: '16px' }}>导出格式</h3>
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            {trainConfig.exportFormats.map((fmt) => (
              <button
                key={fmt}
                onClick={() => handleDownload(`model.${fmt}`)}
                style={{
                  padding: '8px 20px',
                  background: '#fff',
                  border: '1px solid #1976d2',
                  color: '#1976d2',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '14px',
                }}
              >
                .{fmt.toUpperCase()} 下载
              </button>
            ))}
          </div>
        </div>
      )}

      <div style={{ background: '#fff', borderRadius: '8px', padding: '20px', marginBottom: '24px' }}>
        <h3 style={{ fontSize: '16px', marginBottom: '16px' }}>训练报告</h3>
        <button
          onClick={() => handleDownload('results.csv')}
          style={{
            padding: '8px 20px',
            background: '#fff',
            border: '1px solid #666',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px',
          }}
        >
          下载 results.csv
        </button>
      </div>

      <div style={{ display: 'flex', gap: '12px' }}>
        <button
          onClick={reset}
          style={{
            padding: '10px 24px',
            background: '#1976d2',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '15px',
          }}
        >
          开始新任务
        </button>
        <button
          onClick={() => window.location.reload()}
          style={{
            padding: '10px 24px',
            background: '#fff',
            border: '1px solid #ccc',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '15px',
          }}
        >
          刷新页面
        </button>
      </div>
    </div>
  )
}

import { useTaskStore } from '../store/taskStore'

export default function IntentConfirm() {
  const { vlmResult, updateVLMClass, setStage } = useTaskStore()

  if (!vlmResult) {
    return <div>无 VLM 解析结果，请先上传样板图</div>
  }

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto' }}>
      <h2 style={{ fontSize: '20px', marginBottom: '20px' }}>阶段一：确认检测意图</h2>

      <p style={{ color: '#666', marginBottom: '16px' }}>
        以下是 VLM 解析出的检测类别，您可手动微调每个 class 的 prompt：
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginBottom: '24px' }}>
        {vlmResult.classes.map((cls, index) => (
          <div
            key={index}
            style={{
              background: '#fff',
              border: '1px solid #e0e0e0',
              borderRadius: '8px',
              padding: '16px',
            }}
          >
            <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginBottom: '12px' }}>
              <span
                style={{
                  background: '#1976d2',
                  color: '#fff',
                  padding: '4px 10px',
                  borderRadius: '4px',
                  fontWeight: 600,
                  fontSize: '14px',
                }}
              >
                Class {index}
              </span>
              <input
                value={cls.class_name}
                onChange={(e) => updateVLMClass(index, { class_name: e.target.value })}
                placeholder="class_name (英文小写下划线)"
                style={{
                  flex: 1,
                  padding: '6px 10px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  fontSize: '14px',
                }}
              />
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div>
                <label style={{ fontSize: '12px', color: '#666' }}>Prompt（英文视觉描述）</label>
                <textarea
                  value={cls.prompt}
                  onChange={(e) => updateVLMClass(index, { prompt: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '6px 10px',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    fontSize: '13px',
                    resize: 'vertical',
                    boxSizing: 'border-box',
                    fontFamily: 'monospace',
                  }}
                  rows={2}
                />
              </div>
              <div>
                <label style={{ fontSize: '12px', color: '#666' }}>Negative Prompt</label>
                <input
                  value={cls.negative_prompt}
                  onChange={(e) => updateVLMClass(index, { negative_prompt: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '6px 10px',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    fontSize: '13px',
                  }}
                />
              </div>
              <div>
                <label style={{ fontSize: '12px', color: '#666' }}>Color Hint（可为 null）</label>
                <input
                  value={cls.color_hint ?? ''}
                  onChange={(e) => updateVLMClass(index, { color_hint: e.target.value || null })}
                  style={{
                    width: '100%',
                    padding: '6px 10px',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    fontSize: '13px',
                  }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', gap: '12px' }}>
        <button
          onClick={() => setStage('labeling')}
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
          开始打标
        </button>
        <button
          onClick={() => setStage('upload')}
          style={{
            padding: '10px 24px',
            background: '#fff',
            border: '1px solid #ccc',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '15px',
          }}
        >
          返回修改
        </button>
      </div>
    </div>
  )
}

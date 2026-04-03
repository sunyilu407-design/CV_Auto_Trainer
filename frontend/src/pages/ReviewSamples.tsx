import { useTaskStore } from '../store/taskStore'

export default function ReviewSamples() {
  const { qualityReport, splitStats, setStage } = useTaskStore()

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto' }}>
      <h2 style={{ fontSize: '20px', marginBottom: '20px' }}>阶段三：数据质量评估</h2>

      {qualityReport && (
        <div style={{ background: '#fff', borderRadius: '8px', padding: '20px', marginBottom: '24px' }}>
          <h3 style={{ fontSize: '16px', marginBottom: '16px' }}>数据集统计</h3>
          <p style={{ fontSize: '14px', marginBottom: '8px' }}>总图片数：{qualityReport.totalImages}</p>
          <p style={{ fontSize: '14px', marginBottom: '16px' }}>平均每张 bbox 数：{qualityReport.avgBoxesPerImage.toFixed(1)}</p>

          {qualityReport.warnings.length > 0 && (
            <div style={{ background: '#fff3e0', borderRadius: '4px', padding: '12px', marginBottom: '16px' }}>
              {qualityReport.warnings.map((w, i) => (
                <p key={i} style={{ fontSize: '13px', color: '#e65100', margin: '4px 0' }}>{w}</p>
              ))}
            </div>
          )}

          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
            <thead>
              <tr style={{ background: '#f5f5f5' }}>
                <th style={{ padding: '8px', textAlign: 'left' }}>类别</th>
                <th style={{ padding: '8px', textAlign: 'right' }}>bbox 总数</th>
                <th style={{ padding: '8px', textAlign: 'right' }}>平均/图</th>
              </tr>
            </thead>
            <tbody>
              {qualityReport.classDistribution.map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '8px' }}>{d.className}</td>
                  <td style={{ padding: '8px', textAlign: 'right' }}>{d.boxCount}</td>
                  <td style={{ padding: '8px', textAlign: 'right' }}>{d.avgBoxesPerImage.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div style={{ background: '#fff', borderRadius: '8px', padding: '20px', marginBottom: '24px' }}>
        <h3 style={{ fontSize: '16px', marginBottom: '16px' }}>数据集分割（8:1:1）</h3>
        <div style={{ display: 'flex', gap: '24px' }}>
          <div><span style={{ fontSize: '24px', fontWeight: 600 }}>{splitStats.train}</span><br /><span style={{ fontSize: '12px', color: '#666' }}>训练集</span></div>
          <div><span style={{ fontSize: '24px', fontWeight: 600 }}>{splitStats.val}</span><br /><span style={{ fontSize: '12px', color: '#666' }}>验证集</span></div>
          <div><span style={{ fontSize: '24px', fontWeight: 600 }}>{splitStats.test}</span><br /><span style={{ fontSize: '12px', color: '#666' }}>测试集</span></div>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '12px' }}>
        <button
          onClick={() => setStage('train_config')}
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
          确认并前往训练配置
        </button>
        <button
          onClick={() => setStage('augment')}
          style={{
            padding: '10px 24px',
            background: '#fff',
            border: '1px solid #ccc',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '15px',
          }}
        >
          返回修改增强
        </button>
      </div>
    </div>
  )
}

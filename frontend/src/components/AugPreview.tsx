interface Props {
  strength: 'light' | 'medium' | 'heavy'
}

export default function AugPreview({ strength }: Props) {
  const descriptions: Record<string, string[]> = {
    light: ['水平翻转', '亮度对比度调整', '小幅旋转 (±5°)', '饱和度变化'],
    medium: ['水平翻转 + 旋转 (±15°)', '亮度/对比度/Gamma', '高斯噪声', '运动模糊', 'CLAHE'],
    heavy: ['所有中等变换', '天气模拟（雨/雾）', '遮挡模拟 Cutout', ' Mosaic 拼接'],
  }

  return (
    <div style={{ background: '#fff', borderRadius: '8px', padding: '16px' }}>
      <h3 style={{ fontSize: '16px', marginBottom: '12px' }}>增强效果预览</h3>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr 1fr',
          gap: '8px',
          marginBottom: '16px',
        }}
      >
        {['原图', '翻转', '增强'].map((label, i) => (
          <div
            key={i}
            style={{
              background: '#f5f5f5',
              borderRadius: '4px',
              height: '80px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '12px',
              color: '#999',
              border: i === 0 ? '2px solid #1976d2' : '1px solid #ddd',
            }}
          >
            {label}
          </div>
        ))}
      </div>

      <h4 style={{ fontSize: '14px', marginBottom: '8px' }}>增强内容（{strength === 'light' ? '轻' : strength === 'medium' ? '中' : '重'}度）</h4>
      <ul style={{ fontSize: '13px', color: '#666', paddingLeft: '16px', margin: 0 }}>
        {(descriptions[strength] || []).map((d, i) => (
          <li key={i} style={{ marginBottom: '4px' }}>{d}</li>
        ))}
      </ul>
    </div>
  )
}

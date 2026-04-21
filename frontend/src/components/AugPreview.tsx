interface Props {
  strength: 'light' | 'medium' | 'heavy'
}

export default function AugPreview({ strength }: Props) {
  const descriptions: Record<string, string[]> = {
    light: ['水平翻转', '亮度对比度调整', '小幅旋转 (±5°)', '饱和度变化'],
    medium: ['水平翻转 + 旋转 (±15°)', '亮度/对比度/Gamma', '高斯噪声', '运动模糊', 'CLAHE'],
    heavy: ['所有中等变换', '天气模拟（雨/雾）', '遮挡模拟 Cutout', 'Mosaic 拼接'],
  }

  const strengthColor: Record<string, string> = {
    light: 'var(--develop-blue)',
    medium: '#f59e0b',
    heavy: 'var(--preview-pink)',
  }

  const strengthBg: Record<string, string> = {
    light: 'rgba(10,114,239,0.06)',
    medium: 'rgba(245,158,11,0.06)',
    heavy: 'rgba(222,29,141,0.06)',
  }

  return (
    <div className="card-section" style={{ padding: 0, overflow: 'hidden' }}>
      {/* Header */}
      <div
        style={{
          padding: '14px 20px',
          borderBottom: '1px solid var(--gray-100)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <h3 style={{ fontSize: 14, fontWeight: 600, margin: 0, letterSpacing: '-0.2px' }}>增强效果预览</h3>
        <span
          style={{
            fontSize: 11,
            fontWeight: 600,
            padding: '3px 8px',
            borderRadius: 4,
            background: strengthBg[strength],
            color: strengthColor[strength],
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
          }}
        >
          {strength === 'light' ? '轻' : strength === 'medium' ? '中' : '重'}度
        </span>
      </div>

      {/* Preview Grid */}
      <div style={{ padding: '16px 20px' }}>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr 1fr',
            gap: 8,
            marginBottom: 16,
          }}
        >
          {[
            { label: '原图', accent: true },
            { label: '翻转', accent: false },
            { label: '增强', accent: false },
          ].map(({ label, accent }, i) => (
            <div
              key={i}
              style={{
                background: 'var(--gray-50)',
                borderRadius: 6,
                height: 72,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: accent ? `1.5px solid ${strengthColor[strength]}` : '1px solid var(--gray-100)',
                position: 'relative',
                overflow: 'hidden',
              }}
            >
              {i === 1 && (
                <svg
                  style={{ position: 'absolute', opacity: 0.3 }}
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke={strengthColor[strength]}
                  strokeWidth="1.5"
                >
                  <polyline points="17 1 21 5 17 9"/>
                  <path d="M3 11V9a4 4 0 014-4h14"/>
                  <polyline points="7 23 3 19 7 15"/>
                  <path d="M21 13v2a4 4 0 01-4 4H3"/>
                </svg>
              )}
              {i === 2 && (
                <svg
                  style={{ position: 'absolute', opacity: 0.3 }}
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke={strengthColor[strength]}
                  strokeWidth="1.5"
                >
                  <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                  <path d="M2 17l10 5 10-5"/>
                  <path d="M2 12l10 5 10-5"/>
                </svg>
              )}
              <span
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  color: accent ? strengthColor[strength] : 'var(--gray-400)',
                  letterSpacing: '0.02em',
                }}
              >
                {label}
              </span>
            </div>
          ))}
        </div>

        {/* Augmentation list */}
        <div
          style={{
            padding: '12px 14px',
            background: strengthBg[strength],
            borderRadius: 6,
            border: `1px solid ${strengthColor[strength]}20`,
          }}
        >
          <p
            style={{
              fontSize: 11,
              fontWeight: 600,
              color: strengthColor[strength],
              margin: '0 0 8px',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
            }}
          >
            增强内容
          </p>
          <ul
            style={{
              fontSize: 12,
              color: 'var(--gray-600)',
              padding: 0,
              margin: 0,
              listStyle: 'none',
              display: 'flex',
              flexDirection: 'column',
              gap: 4,
            }}
          >
            {(descriptions[strength] || []).map((d, i) => (
              <li key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <div
                  style={{
                    width: 4,
                    height: 4,
                    borderRadius: '50%',
                    background: strengthColor[strength],
                    flexShrink: 0,
                  }}
                />
                {d}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  )
}

import { useState } from 'react'
import { useTaskStore } from '../store/taskStore'
import { workerClient } from '../api/worker'
import AugPreview from '../components/AugPreview'

export default function AugmentConfig() {
  const { augConfig, setAugConfig, setStage, setTotalImageCount } = useTaskStore()
  const [augmenting, setAugmenting] = useState(false)

  const handleStartAug = () => {
    setAugmenting(true)
    workerClient.connect()
    workerClient.onMessage((msg) => {
      if (msg.type === 'progress' && msg.stage === 'augmentation') {
        useTaskStore.setState({
          totalImageCount: Math.round((msg.current as number) / (msg.total as number) * augConfig.targetCount),
        })
      }
      if (msg.type === 'stage_complete' && msg.stage === 'augmentation') {
        setTotalImageCount((msg.result as { total: number }).total)
        setAugmenting(false)
        setStage('review')
      }
    })
    workerClient.startAugmentation({
      src_image_dir: '/path/to/labeled/images',
      src_label_dir: '/path/to/labeled/labels',
      output_image_dir: '/path/to/aug/images',
      output_label_dir: '/path/to/aug/labels',
      target_count: augConfig.targetCount,
      strength: augConfig.strength,
      enabled: augConfig.enabled,
    })
  }

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto' }}>
      <h2 style={{ fontSize: '20px', marginBottom: '20px' }}>阶段二点五：数据增强配置</h2>

      <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
        {/* 左侧配置 */}
        <div style={{ flex: '1 1 400px', background: '#fff', borderRadius: '8px', padding: '20px' }}>
          <h3 style={{ fontSize: '16px', marginBottom: '16px' }}>增强配置</h3>

          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', fontSize: '14px', marginBottom: '6px' }}>
              目标总图片数：{augConfig.targetCount}
            </label>
            <input
              type="range"
              min="10"
              max="5000"
              step="10"
              value={augConfig.targetCount}
              onChange={(e) => setAugConfig({ targetCount: Number(e.target.value) })}
              style={{ width: '100%' }}
            />
            <span style={{ fontSize: '12px', color: '#666' }}>当前：{augConfig.targetCount} 张</span>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', fontSize: '14px', marginBottom: '6px' }}>增强强度</label>
            <div style={{ display: 'flex', gap: '8px' }}>
              {(['light', 'medium', 'heavy'] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => setAugConfig({ strength: s })}
                  style={{
                    flex: 1,
                    padding: '8px',
                    background: augConfig.strength === s ? '#1976d2' : '#fff',
                    color: augConfig.strength === s ? '#fff' : '#333',
                    border: '1px solid #ccc',
                    borderRadius: '4px',
                    cursor: 'pointer',
                  }}
                >
                  {s === 'light' ? '轻' : s === 'medium' ? '中' : '重'}
                </button>
              ))}
            </div>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', fontSize: '14px', marginBottom: '8px' }}>变换类型</label>
            {([
              { key: 'geometric', label: '几何变换（翻转/旋转/缩放）' },
              { key: 'color', label: '色彩扰动（亮度/饱和度）' },
              { key: 'noise', label: '噪声与模糊' },
              { key: 'weather', label: '天气模拟（雨/雾）' },
              { key: 'occlusion', label: '遮挡模拟（Cutout/Mosaic）' },
            ] as const).map(({ key, label }) => (
              <div key={key} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                <input
                  type="checkbox"
                  checked={augConfig.enabled[key]}
                  onChange={(e) =>
                    setAugConfig({ enabled: { ...augConfig.enabled, [key]: e.target.checked } })
                  }
                />
                <span style={{ fontSize: '14px' }}>{label}</span>
              </div>
            ))}
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={augConfig.deleteOriginalImages}
                onChange={(e) => setAugConfig({ deleteOriginalImages: e.target.checked })}
              />
              <span style={{ fontSize: '14px' }}>增强完成后删除原图</span>
            </label>
          </div>
        </div>

        {/* 右侧预览 */}
        <div style={{ flex: '1 1 300px' }}>
          <AugPreview strength={augConfig.strength} />
        </div>
      </div>

      <div style={{ marginTop: '24px', display: 'flex', gap: '12px' }}>
        <button
          onClick={handleStartAug}
          disabled={augmenting}
          style={{
            padding: '10px 24px',
            background: augmenting ? '#ccc' : '#1976d2',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: augmenting ? 'not-allowed' : 'pointer',
            fontSize: '15px',
          }}
        >
          {augmenting ? '增强中...' : '开始增强'}
        </button>
        <button
          onClick={() => setStage('review')}
          style={{
            padding: '10px 24px',
            background: '#fff',
            border: '1px solid #ccc',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '15px',
          }}
        >
          跳过增强
        </button>
        <button
          onClick={() => setStage('labeling')}
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

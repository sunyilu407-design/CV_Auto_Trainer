import { useState } from 'react'
import { useTaskStore } from '../store/taskStore'
import { workerClient } from '../api/worker'
import AugPreview from '../components/AugPreview'

export default function AugmentConfig() {
  const { taskId, augConfig, setAugConfig, setStage, setTotalImageCount } = useTaskStore()
  const [augmenting, setAugmenting] = useState(false)

  const handleStartAug = () => {
    if (!taskId) {
      alert('缺少任务 ID，请重新从上传阶段开始')
      return
    }

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
      src_image_dir: `../backend/uploads/${taskId}/labeled_images`,
      src_label_dir: `../backend/uploads/${taskId}/labels`,
      output_image_dir: `../backend/uploads/${taskId}/dataset/images/train`,
      output_label_dir: `../backend/uploads/${taskId}/dataset/labels/train`,
      target_count: augConfig.targetCount,
      strength: augConfig.strength,
      enabled: augConfig.enabled,
    })
  }

  return (
    <div>
      {/* Page Header */}
      <div className="page-header" style={{ marginBottom: 32 }}>
        <div className="flex items-center gap-3 mb-4">
          <div className="badge" style={{ background: 'var(--preview-pink)', color: '#fff', fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Stage 2.5</div>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: 'var(--preview-pink)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="23 4 23 10 17 10"/>
              <path d="M20.49 15a9 9 0 11-2.12-9.36L23 10"/>
            </svg>
          </div>
        </div>
        <h1 className="page-title">数据增强配置</h1>
        <p className="page-subtitle">Albumentations 离线增强，零 API 成本，GPU 端侧执行</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 24 }}>
        {/* Left: Config */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Target count */}
          <div className="card-section">
            <div className="flex justify-between items-center mb-4">
              <h3 style={{ fontSize: 15, fontWeight: 600 }}>目标图片数量</h3>
              <span style={{ fontSize: 20, fontWeight: 700, color: 'var(--preview-pink)', letterSpacing: '-1px' }}>
                {augConfig.targetCount}
                <span style={{ fontSize: 12, color: 'var(--gray-400)', fontWeight: 400, marginLeft: 4 }}>张</span>
              </span>
            </div>
            <input
              type="range"
              min="10"
              max="5000"
              step="10"
              value={augConfig.targetCount}
              onChange={(e) => setAugConfig({ targetCount: Number(e.target.value) })}
              style={{ width: '100%', accentColor: 'var(--preview-pink)' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
              <span style={{ fontSize: 11, color: 'var(--gray-400)' }}>10</span>
              <span style={{ fontSize: 11, color: 'var(--gray-400)' }}>5000</span>
            </div>
          </div>

          {/* Strength */}
          <div className="card-section">
            <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 12 }}>增强强度</h3>
            <div style={{ display: 'flex', gap: 8 }}>
              {(['light', 'medium', 'heavy'] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => setAugConfig({ strength: s })}
                  style={{
                    flex: 1,
                    padding: '10px 8px',
                    borderRadius: 8,
                    border: `2px solid ${augConfig.strength === s ? 'var(--preview-pink)' : 'var(--gray-100)'}`,
                    background: augConfig.strength === s ? 'rgba(222,29,141,0.06)' : 'var(--white)',
                    color: augConfig.strength === s ? 'var(--preview-pink)' : 'var(--gray-600)',
                    fontWeight: augConfig.strength === s ? 600 : 400,
                    fontSize: 13,
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                  }}
                >
                  {s === 'light' ? '轻度' : s === 'medium' ? '中度' : '重度'}
                </button>
              ))}
            </div>
          </div>

          {/* Transform types */}
          <div className="card-section">
            <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 12 }}>变换类型</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {([
                { key: 'geometric', label: '几何变换', sub: '翻转 / 旋转 / 缩放', icon: '↔' },
                { key: 'color', label: '色彩扰动', sub: '亮度 / 饱和度 / Gamma', icon: '◑' },
                { key: 'noise', label: '噪声与模糊', sub: '高斯噪声 / 运动模糊', icon: '◎' },
                { key: 'weather', label: '天气模拟', sub: '雨 / 雾 / 光照', icon: '☁' },
                { key: 'occlusion', label: '遮挡模拟', sub: 'Cutout / Mosaic', icon: '▦' },
              ] as const).map(({ key, label, sub }) => (
                <label
                  key={key}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    padding: '10px 12px',
                    borderRadius: 8,
                    border: `1px solid ${augConfig.enabled[key] ? 'rgba(222,29,141,0.2)' : 'var(--gray-100)'}`,
                    background: augConfig.enabled[key] ? 'rgba(222,29,141,0.04)' : 'var(--gray-50)',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={augConfig.enabled[key]}
                    onChange={(e) => setAugConfig({ enabled: { ...augConfig.enabled, [key]: e.target.checked } })}
                    style={{ accentColor: 'var(--preview-pink)' }}
                  />
                  <div>
                    <p style={{ fontSize: 13, fontWeight: 500, color: augConfig.enabled[key] ? 'var(--preview-pink)' : 'var(--gray-700)' }}>{label}</p>
                    <p style={{ fontSize: 11, color: 'var(--gray-400)' }}>{sub}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Delete original */}
          <div className="card-section" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <p style={{ fontSize: 14, fontWeight: 500 }}>增强后删除原图</p>
              <p style={{ fontSize: 12, color: 'var(--gray-400)' }}>节省存储空间，增强图替代原图</p>
            </div>
            <label style={{ position: 'relative', display: 'inline-flex', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={augConfig.deleteOriginalImages}
                onChange={(e) => setAugConfig({ deleteOriginalImages: e.target.checked })}
                style={{ width: 0, height: 0, opacity: 0 }}
              />
              <div
                style={{
                  width: 40,
                  height: 22,
                  borderRadius: 11,
                  background: augConfig.deleteOriginalImages ? 'var(--preview-pink)' : 'var(--gray-200)',
                  position: 'relative',
                  transition: 'all 0.2s ease',
                }}
              >
                <div
                  style={{
                    width: 16,
                    height: 16,
                    borderRadius: '50%',
                    background: '#fff',
                    position: 'absolute',
                    top: 3,
                    left: augConfig.deleteOriginalImages ? 21 : 3,
                    transition: 'left 0.2s ease',
                    boxShadow: '0 1px 3px rgba(0,0,0,0.15)',
                  }}
                />
              </div>
            </label>
          </div>
        </div>

        {/* Right: Preview */}
        <div>
          <AugPreview strength={augConfig.strength} />
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-3 mt-8">
        <button
          className="btn btn-primary"
          onClick={handleStartAug}
          disabled={augmenting}
          style={{
            padding: '10px 24px',
            background: 'var(--preview-pink)',
            fontWeight: 600,
          }}
        >
          {augmenting ? (
            <>
              <div className="spinner" />
              增强中...
            </>
          ) : (
            <>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
              开始增强
            </>
          )}
        </button>
        <button className="btn btn-secondary" onClick={() => setStage('review')}>
          跳过增强
        </button>
        <button className="btn btn-ghost" onClick={() => setStage('labeling')}>
          返回修改
        </button>
      </div>
    </div>
  )
}

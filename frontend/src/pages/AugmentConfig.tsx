import { useState } from 'react'
import { useTaskStore } from '../store/taskStore'
import { workerClient } from '../api/worker'
import { trainingApi } from '../api/backend'
import AugPreview from '../components/AugPreview'

export default function AugmentConfig() {
  const { taskId, vlmResult, augConfig, setAugConfig, setStage, setTotalImageCount, setSplitStats, setQualityReport, setWasAugmented, labeledImageCount } = useTaskStore()
  const [augmenting, setAugmenting] = useState(false)
  const [preparing, setPreparing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const prepareDatasetAndGoToReview = async (augmentedPaths?: { images: string; labels: string }) => {
    if (!taskId) return
    setPreparing(true)
    setError(null)
    try {
      const classNames = (vlmResult?.classes ?? []).map((c) => c.class_name)
      const result = await trainingApi.prepareDataset({
        task_id: taskId,
        class_names: classNames,
        labeled_images_dir_override: augmentedPaths?.images,
        labels_dir_override: augmentedPaths?.labels,
      })
      const qualityReport = result.quality_report
      setSplitStats(result.split_stats)
      setQualityReport({
        totalImages: qualityReport.total_images,
        classDistribution: qualityReport.class_distribution.map((c) => ({
          className: c.class_name,
          boxCount: c.box_count,
          avgBoxesPerImage: c.avg_boxes_per_image,
        })),
        avgBoxesPerImage: qualityReport.avg_boxes_per_image,
        warnings: qualityReport.warnings,
      })
      setTotalImageCount(result.split_stats.train + result.split_stats.val + result.split_stats.test)
      setStage('review')
    } catch (e) {
      setError(`数据集整理失败: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setPreparing(false)
    }
  }

  const handleStartAug = () => {
    if (!taskId) {
      alert('缺少任务 ID，请重新从上传阶段开始')
      return
    }
    if (labeledImageCount <= 0) {
      setError('当前没有可用于增强的有效标注图片。请返回修改监测对象提示词/阈值，或上传已有 YOLO 标注后重新打标。')
      return
    }

    setAugmenting(true)
    setError(null)
    workerClient.connect()
    workerClient.onMessage((msg) => {
      if (msg.type === 'progress' && msg.stage === 'augmentation') {
        useTaskStore.setState({
          totalImageCount: Math.round((msg.current as number) / (msg.total as number) * augConfig.targetCount),
        })
      }
      if (msg.type === 'stage_complete' && msg.stage === 'augmentation') {
        const result = msg.result as { total: number }
        setTotalImageCount(result.total)
        setWasAugmented(true)
        setAugmenting(false)
        // 先获取增强数据的质量报告（基于标注前输出目录），用于 Review 页面展示
        trainingApi.augQualityReport({
          task_id: taskId,
          augmented_images_dir: `../backend/uploads/${taskId}/labeled_images_aug`,
          augmented_labels_dir: `../backend/uploads/${taskId}/labels_aug`,
          class_names: (vlmResult?.classes ?? []).map((c) => c.class_name),
        }).then((report) => {
          const qr = report.data?.quality_report
          if (qr) {
            setQualityReport({
              totalImages: qr.total_images,
              classDistribution: qr.class_distribution.map((c) => ({
                className: c.class_name,
                boxCount: c.box_count,
                avgBoxesPerImage: c.avg_boxes_per_image,
              })),
              avgBoxesPerImage: qr.avg_boxes_per_image,
              warnings: qr.warnings,
            })
          }
        }).catch(() => {
          // 非致命：静默失败，Review 页面会用分割后数据兜底
        }).finally(() => {
          // 传完整增强目录（含原始+增强图片），让后端重新分层分割为 train/val/test
          prepareDatasetAndGoToReview({
            images: `../backend/uploads/${taskId}/labeled_images_aug`,
            labels: `../backend/uploads/${taskId}/labels_aug`,
          })
        })
      }
      if (msg.type === 'error') {
        setAugmenting(false)
        setError(`增强出错: ${String(msg.message ?? '未知错误')}`)
      }
    })
    workerClient.startAugmentation({
      src_image_dir: `../backend/uploads/${taskId}/labeled_images`,
      src_label_dir: `../backend/uploads/${taskId}/labels`,
      output_image_dir: `../backend/uploads/${taskId}/labeled_images_aug`,
      output_label_dir: `../backend/uploads/${taskId}/labels_aug`,
      target_count: augConfig.targetCount,
      strength: augConfig.strength,
      enabled: augConfig.enabled,
      delete_original: augConfig.deleteOriginalImages,
      min_visibility: augConfig.minVisibility,
      max_per_image: augConfig.maxPerImage,
    })
  }

  const handleSkipAndGoToReview = () => {
    if (labeledImageCount <= 0) {
      setError('当前没有可整理的数据集：标注图片数量为 0。请先完成至少 1 张有效图片标注。')
      return
    }
    setWasAugmented(false)
    prepareDatasetAndGoToReview({
      images: `../backend/uploads/${taskId}/labeled_images`,
      labels: `../backend/uploads/${taskId}/labels`,
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

        {/* Advanced Parameters */}
      <div className="card-section" style={{ marginBottom: 16 }}>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          style={{
            display: 'flex', alignItems: 'center', gap: 8,
            background: 'none', border: 'none', cursor: 'pointer',
            padding: 0, fontSize: 13, fontWeight: 600, color: 'var(--gray-600)',
          }}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
            style={{ transform: showAdvanced ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}>
            <polyline points="9 18 15 12 9 6"/>
          </svg>
          高级参数
        </button>

        {showAdvanced && (
          <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div className="form-group">
              <label className="form-label">bbox 最小可见比例</label>
              <input
                type="range"
                min="0.1"
                max="0.7"
                step="0.05"
                value={augConfig.minVisibility}
                onChange={(e) => setAugConfig({ minVisibility: Number(e.target.value) })}
                style={{ width: '100%', accentColor: 'var(--preview-pink)' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                <span style={{ fontSize: 11, color: 'var(--gray-400)' }}>0.1（保留更多框）</span>
                <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--preview-pink)' }}>{augConfig.minVisibility.toFixed(2)}</span>
                <span style={{ fontSize: 11, color: 'var(--gray-400)' }}>0.7（过滤掉更多）</span>
              </div>
              <p style={{ fontSize: 11, color: 'var(--gray-400)', marginTop: 4 }}>
                增强后 bbox 被裁剪超过此比例会被丢弃。值越低小目标保留越多。
              </p>
            </div>

            <div className="form-group">
              <label className="form-label">单图最多增强数</label>
              <input
                type="range"
                min="1"
                max="20"
                step="1"
                value={augConfig.maxPerImage}
                onChange={(e) => setAugConfig({ maxPerImage: Number(e.target.value) })}
                style={{ width: '100%', accentColor: 'var(--preview-pink)' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                <span style={{ fontSize: 11, color: 'var(--gray-400)' }}>1</span>
                <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--preview-pink)' }}>{augConfig.maxPerImage}</span>
                <span style={{ fontSize: 11, color: 'var(--gray-400)' }}>20</span>
              </div>
              <p style={{ fontSize: 11, color: 'var(--gray-400)', marginTop: 4 }}>
                每张原图最多生成多少张增强图，防止单张图过度增强。
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Right: Preview */}
        <div>
          <AugPreview strength={augConfig.strength} />
        </div>
      </div>

      {/* Actions */}
      {error && (
        <div style={{ marginBottom: 16, padding: '12px 16px', background: 'rgba(255,91,79,0.08)', border: '1px solid rgba(255,91,79,0.2)', borderRadius: 8, fontSize: 13, color: 'var(--ship-red)' }}>
          {error}
        </div>
      )}
      <div className="flex gap-3 mt-8">
        <button
          className="btn btn-primary"
          onClick={handleStartAug}
          disabled={augmenting || preparing}
          style={{
            padding: '10px 24px',
            background: augmenting || preparing ? 'var(--gray-300)' : 'var(--preview-pink)',
            fontWeight: 600,
            cursor: augmenting || preparing ? 'not-allowed' : 'pointer',
          }}
        >
          {augmenting || preparing ? (
            <>
              <div className="spinner" />
              {augmenting ? '增强中...' : '整理数据集...'}
            </>
          ) : (
            <>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
              开始增强
            </>
          )}
        </button>
        <button className="btn btn-secondary" onClick={handleSkipAndGoToReview} disabled={preparing}>
          跳过增强
        </button>
        <button className="btn btn-ghost" onClick={() => setStage('labeling')}>
          返回修改
        </button>
      </div>
    </div>
  )
}

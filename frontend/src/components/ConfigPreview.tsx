import { useState } from 'react'
import { useTaskStore, NegotiatedConfig } from '../store/taskStore'
import { negotiateApi } from '../api/backend'

type Tab = 'config' | 'preview' | 'requirements'

interface PreviewImage {
  image_name: string
  image_base64: string
  detections: Array<{ class_name: string; confidence: number; bbox: number[] }>
  detection_count: number
}

export default function ConfigPreview() {
  const { taskId, conversationId, negotiatedConfig } = useTaskStore()
  const [activeTab, setActiveTab] = useState<Tab>('config')
  const [previewImages, setPreviewImages] = useState<PreviewImage[]>([])
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState<string | null>(null)

  async function handleRunPreview() {
    if (!taskId) return
    setPreviewLoading(true)
    setPreviewError(null)
    try {
      const resp = await negotiateApi.preview({
        task_id: taskId,
        conversation_id: conversationId || undefined,
      })
      setPreviewImages((resp as any).results || [])
      setActiveTab('preview')
    } catch (err: any) {
      setPreviewError(err?.message || '预览失败')
    } finally {
      setPreviewLoading(false)
    }
  }

  const tabs: { key: Tab; label: string }[] = [
    { key: 'config', label: '检测配置' },
    { key: 'preview', label: `预览结果${previewImages.length ? ` (${previewImages.reduce((s, p) => s + p.detection_count, 0)})` : ''}` },
    { key: 'requirements', label: '完整需求' },
  ]

  return (
    <div
      style={{
        background: '#fff',
        borderRadius: 10,
        border: '1px solid var(--gray-100)',
        overflow: 'hidden',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Tabs */}
      <div
        style={{
          display: 'flex',
          borderBottom: '1px solid var(--gray-100)',
        }}
      >
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            style={{
              flex: 1,
              padding: '10px 12px',
              border: 'none',
              background: 'transparent',
              fontSize: 12,
              fontWeight: activeTab === tab.key ? 700 : 500,
              color: activeTab === tab.key ? '#0a72ef' : 'var(--gray-500)',
              borderBottom: activeTab === tab.key ? '2px solid #0a72ef' : '2px solid transparent',
              cursor: 'pointer',
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px 16px' }}>
        {!negotiatedConfig ? (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              color: 'var(--gray-400)',
              fontSize: 12,
            }}
          >
            等待 AI 生成配置...
          </div>
        ) : activeTab === 'config' ? (
          <ConfigTab config={negotiatedConfig} onRunPreview={handleRunPreview} previewLoading={previewLoading} />
        ) : activeTab === 'preview' ? (
          <PreviewTab images={previewImages} loading={previewLoading} error={previewError} onRetry={handleRunPreview} />
        ) : (
          <RequirementsTab config={negotiatedConfig} />
        )}
      </div>
    </div>
  )
}

function ConfigTab({ config, onRunPreview, previewLoading }: { config: NegotiatedConfig; onRunPreview: () => void; previewLoading: boolean }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {/* Preview button */}
      <button
        onClick={onRunPreview}
        disabled={previewLoading}
        style={{
          padding: '8px 12px',
          borderRadius: 6,
          border: '1px solid #0a72ef',
          background: previewLoading ? 'var(--gray-100)' : 'rgba(10,114,239,0.05)',
          color: previewLoading ? 'var(--gray-400)' : '#0a72ef',
          fontSize: 11,
          fontWeight: 600,
          cursor: previewLoading ? 'not-allowed' : 'pointer',
        }}
      >
        {previewLoading ? '检测中...' : '▶ 运行预览检测'}
      </button>

      {/* Classes */}
      {config.classes.map((cls, idx) => (
        <div
          key={idx}
          style={{
            padding: '10px 12px',
            background: 'var(--gray-50)',
            borderRadius: 8,
            border: '1px solid var(--gray-100)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                background: '#0a72ef',
                display: 'inline-block',
              }}
            />
            <span style={{ fontWeight: 700, fontSize: 13, color: 'var(--gray-800)' }}>
              {cls.display_name_zh || cls.class_name}
            </span>
            <span style={{ fontSize: 11, color: 'var(--gray-400)', fontFamily: 'monospace' }}>
              {cls.class_name}
            </span>
          </div>
          <div style={{ fontSize: 11, color: 'var(--gray-600)', lineHeight: '1.4' }}>
            <div><b>检测词:</b> {cls.prompt}</div>
            {(cls as any).prompt_aliases?.length > 0 && (
              <div><b>别名:</b> {(cls as any).prompt_aliases.join(', ')}</div>
            )}
            {cls.negative_prompt && (
              <div><b>排除:</b> {cls.display_negative_prompt_zh || cls.negative_prompt}</div>
            )}
            {cls.color_hint && (
              <div><b>颜色:</b> {cls.display_color_hint_zh || cls.color_hint}</div>
            )}
          </div>
        </div>
      ))}

      {/* Detection Rules */}
      {config.detection_rules && (
        <div
          style={{
            padding: '10px 12px',
            background: 'rgba(245,158,11,0.05)',
            borderRadius: 8,
            border: '1px solid rgba(245,158,11,0.2)',
          }}
        >
          <div style={{ fontSize: 11, fontWeight: 600, color: '#d97706', marginBottom: 4 }}>
            检测规则
          </div>
          <div style={{ fontSize: 11, color: 'var(--gray-600)' }}>
            <div>置信度阈值: {config.detection_rules.conf_threshold}</div>
            <div>IoU 阈值: {config.detection_rules.iou_threshold}</div>
            {config.detection_rules.post_filters?.length > 0 && (
              <div>
                后处理过滤: {config.detection_rules.post_filters.map((f) => `${f.type}=${f.value}`).join(', ')}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function RequirementsTab({ config }: { config: NegotiatedConfig }) {
  const hints = config.algorithm_hints
  if (!hints) {
    return <div style={{ color: 'var(--gray-400)', fontSize: 12 }}>暂无需求摘要</div>
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 12 }}>
      {hints.scenario_type && (
        <InfoRow label="场景类型" value={hints.scenario_type} />
      )}

      {hints.events && hints.events.length > 0 && (
        <div>
          <div style={{ fontWeight: 600, color: 'var(--gray-700)', marginBottom: 4 }}>事件/告警</div>
          {hints.events.map((e, i) => (
            <div key={i} style={{ color: 'var(--gray-600)', paddingLeft: 8 }}>
              • {e.name_zh}: {e.trigger}
            </div>
          ))}
        </div>
      )}

      {hints.regions && hints.regions.length > 0 && (
        <div>
          <div style={{ fontWeight: 600, color: 'var(--gray-700)', marginBottom: 4 }}>区域约束</div>
          {hints.regions.map((r, i) => (
            <div key={i} style={{ color: 'var(--gray-600)', paddingLeft: 8 }}>
              • {r.label}: {r.purpose}
            </div>
          ))}
        </div>
      )}

      <InfoRow label="需要跟踪" value={hints.needs_tracking ? '是' : '否'} />
      <InfoRow label="需要 OCR" value={hints.needs_ocr ? '是' : '否'} />
      {hints.performance_hint && (
        <InfoRow label="性能要求" value={hints.performance_hint} />
      )}
      {hints.suggested_pipeline_roles && hints.suggested_pipeline_roles.length > 0 && (
        <InfoRow label="Pipeline 角色" value={hints.suggested_pipeline_roles.join(', ')} />
      )}
    </div>
  )
}

function PreviewTab({
  images,
  loading,
  error,
  onRetry,
}: {
  images: PreviewImage[]
  loading: boolean
  error: string | null
  onRetry: () => void
}) {
  if (loading) {
    return <div style={{ textAlign: 'center', padding: 20, color: 'var(--gray-400)', fontSize: 12 }}>正在检测...</div>
  }

  if (error) {
    return (
      <div style={{ textAlign: 'center', padding: 20 }}>
        <div style={{ color: '#dc2626', fontSize: 12, marginBottom: 8 }}>{error}</div>
        <button onClick={onRetry} style={{ fontSize: 11, color: '#0a72ef', background: 'none', border: 'none', cursor: 'pointer', textDecoration: 'underline' }}>
          重试
        </button>
      </div>
    )
  }

  if (images.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: 20, color: 'var(--gray-400)', fontSize: 12 }}>
        点击"运行预览检测"查看效果
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {images.map((img, idx) => (
        <div key={idx} style={{ borderRadius: 8, border: '1px solid var(--gray-100)', overflow: 'hidden' }}>
          <div style={{ position: 'relative' }}>
            <img
              src={`data:image/jpeg;base64,${img.image_base64}`}
              alt={img.image_name}
              style={{ width: '100%', display: 'block' }}
            />
          </div>
          <div style={{ padding: '6px 10px', background: 'var(--gray-50)', fontSize: 11 }}>
            <span style={{ fontWeight: 600 }}>{img.image_name}</span>
            <span style={{ marginLeft: 8, color: img.detection_count > 0 ? '#059669' : 'var(--gray-400)' }}>
              {img.detection_count} 个检测
            </span>
            {img.detections.length > 0 && (
              <div style={{ marginTop: 4, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {img.detections.map((d, di) => (
                  <span
                    key={di}
                    style={{
                      padding: '1px 6px',
                      borderRadius: 4,
                      background: 'rgba(10,114,239,0.1)',
                      color: '#0a72ef',
                      fontSize: 10,
                    }}
                  >
                    {d.class_name} ({(d.confidence * 100).toFixed(0)}%)
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', gap: 8 }}>
      <span style={{ fontWeight: 600, color: 'var(--gray-700)', minWidth: 70 }}>{label}</span>
      <span style={{ color: 'var(--gray-600)' }}>{value}</span>
    </div>
  )
}

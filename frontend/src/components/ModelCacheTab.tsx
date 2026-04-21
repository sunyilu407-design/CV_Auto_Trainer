import { useEffect, useState } from 'react'
import { modelsApi, type CachedModelEntry } from '../api/backend'

const ROLE_TAG_LABELS: Record<string, string> = {
  primary_detector: '主检测器',
  secondary_detector: '辅助检测器',
  classifier: '分类器',
  feature_matcher: '特征匹配器',
  tracker: '跟踪器',
  rule_engine: '规则引擎',
}

function formatTime(ts: number): string {
  if (!ts) return '—'
  const d = new Date(ts * 1000)
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  const hh = String(d.getHours()).padStart(2, '0')
  const mm = String(d.getMinutes()).padStart(2, '0')
  return `${y}-${m}-${day} ${hh}:${mm}`
}

export default function ModelCacheTab() {
  const [entries, setEntries] = useState<CachedModelEntry[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const list = await modelsApi.listCache()
      setEntries(list)
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  const handleDelete = async (cacheId: string) => {
    setDeleting(cacheId)
    try {
      await modelsApi.deleteCache(cacheId, true)
      await load()
      setConfirmDeleteId(null)
    } catch (e) {
      alert(`删除失败：${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setDeleting(null)
    }
  }

  const totalSize = entries.reduce((acc, e) => acc + (e.weight_size_mb || 0), 0)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <p style={{ fontSize: 13, fontWeight: 600, margin: 0 }}>已训练模型缓存</p>
          <p style={{ fontSize: 11, color: 'var(--gray-500)', margin: '2px 0 0' }}>
            共 {entries.length} 个模型 · {totalSize.toFixed(1)} MB · 后续训练会自动复用匹配的模型
          </p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={load} disabled={loading}>
          {loading ? '刷新中...' : '刷新'}
        </button>
      </div>

      {error && (
        <div style={{ padding: 12, borderRadius: 8, background: '#fef2f2', color: '#991b1b', fontSize: 12 }}>
          {error}
        </div>
      )}

      {!loading && entries.length === 0 && !error && (
        <div
          style={{
            padding: '32px 20px',
            textAlign: 'center',
            borderRadius: 8,
            background: 'var(--gray-50)',
            color: 'var(--gray-500)',
            fontSize: 12,
          }}
        >
          暂无已缓存的训练模型。完成一次训练后，此处会自动记录并用于后续任务复用。
        </div>
      )}

      {entries.map((entry) => {
        const isConfirming = confirmDeleteId === entry.cache_id
        const isDeleting = deleting === entry.cache_id
        const roleTag = entry.tags.find((t) => t in ROLE_TAG_LABELS)
        return (
          <div
            key={entry.cache_id}
            style={{
              padding: '14px 16px',
              borderRadius: 10,
              background: entry.weight_exists ? 'var(--gray-50)' : '#fff8f0',
              border: `1px solid ${entry.weight_exists ? 'var(--gray-100)' : '#fde68a'}`,
              display: 'flex',
              flexDirection: 'column',
              gap: 10,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'flex-start' }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 4 }}>
                  {roleTag && (
                    <span className="badge badge-blue" style={{ fontSize: 10 }}>
                      {ROLE_TAG_LABELS[roleTag]}
                    </span>
                  )}
                  <strong
                    style={{
                      fontSize: 12,
                      fontFamily: 'var(--font-mono)',
                      color: 'var(--gray-900)',
                      wordBreak: 'break-all',
                    }}
                  >
                    {entry.cache_id}
                  </strong>
                  {!entry.weight_exists && (
                    <span className="badge badge-pink" style={{ fontSize: 10 }}>
                      权重缺失
                    </span>
                  )}
                  {entry.reuse_count > 0 && (
                    <span className="badge badge-green" style={{ fontSize: 10 }}>
                      已复用 {entry.reuse_count} 次
                    </span>
                  )}
                </div>
                <div style={{ fontSize: 11, color: 'var(--gray-500)', lineHeight: 1.7 }}>
                  <div>
                    基础模型：<code style={{ fontFamily: 'var(--font-mono)' }}>{entry.source_model_id}</code>
                  </div>
                  <div>
                    场景：{entry.scenario_type || '—'} · 类别：{entry.classes.join('、') || '—'}
                  </div>
                  <div>
                    mAP50：{entry.map50?.toFixed(3) ?? '—'} · 图片数：{entry.image_count} · 轮次：{entry.epochs_completed}
                  </div>
                  <div>
                    训练时间：{formatTime(entry.trained_at)}
                    {entry.weight_size_mb != null && <> · 大小：{entry.weight_size_mb} MB</>}
                  </div>
                </div>
              </div>

              {!isConfirming && (
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => setConfirmDeleteId(entry.cache_id)}
                  style={{ flexShrink: 0, color: 'var(--ship-red)' }}
                >
                  删除
                </button>
              )}
            </div>

            {isConfirming && (
              <div
                style={{
                  display: 'flex',
                  gap: 8,
                  alignItems: 'center',
                  padding: 10,
                  borderRadius: 6,
                  background: '#fff',
                  border: '1px solid #fecaca',
                }}
              >
                <span style={{ fontSize: 12, color: '#991b1b', flex: 1 }}>
                  确认删除？权重文件也会被一并移除。
                </span>
                <button
                  className="btn btn-sm"
                  onClick={() => setConfirmDeleteId(null)}
                  disabled={isDeleting}
                  style={{ padding: '6px 12px', background: 'var(--gray-100)' }}
                >
                  取消
                </button>
                <button
                  className="btn btn-sm"
                  onClick={() => handleDelete(entry.cache_id)}
                  disabled={isDeleting}
                  style={{ padding: '6px 12px', background: 'var(--ship-red)', color: '#fff' }}
                >
                  {isDeleting ? '删除中...' : '确认删除'}
                </button>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

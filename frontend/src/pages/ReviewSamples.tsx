import { useState } from 'react'
import { useTaskStore } from '../store/taskStore'
import { filesApi } from '../api/backend'

export default function ReviewSamples() {
  const { taskId, qualityReport, splitStats, setStage, wasAugmented } = useTaskStore()
  const [vlmQcLoading, setVlmQcLoading] = useState(false)
  const [vlmQcResult, setVlmQcResult] = useState<string | null>(null)
  const [vlmQcError, setVlmQcError] = useState<string | null>(null)

  const totalImages = qualityReport?.totalImages ?? 0
  const avgBoxes = qualityReport?.avgBoxesPerImage ?? 0
  const warnings = qualityReport?.warnings ?? []
  const classDist = qualityReport?.classDistribution ?? []

  const handleVlmQc = async () => {
    if (!taskId) return
    setVlmQcLoading(true)
    setVlmQcError(null)
    setVlmQcResult(null)
    try {
      const ann = await filesApi.checkExistingAnnotations(taskId)
      if (!ann.has_annotations) {
        setVlmQcError('当前没有标注数据，请先完成打标阶段')
        return
      }
      // 获取标注文件列表传给后端做 VLM 分析
      // 简化：直接在后端抽样读取标注并用 VLM 分析
      const resp = await fetch(`/api/training/qc-annotations/${taskId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${import.meta.env.VITE_AUTH_TOKEN || ''}`,
        },
      })
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: '请求失败' }))
        throw new Error(err.detail || 'VLM 质检请求失败')
      }
      const data = await resp.json()
      setVlmQcResult((data.data as { analysis: string }).analysis)
    } catch (e) {
      setVlmQcError(e instanceof Error ? e.message : String(e))
    } finally {
      setVlmQcLoading(false)
    }
  }

  return (
    <div>
      {/* Page Header */}
      <div className="page-header" style={{ marginBottom: 32 }}>
        <div className="flex items-center gap-3 mb-4">
          <div className="badge" style={{ background: 'var(--preview-pink)', color: '#fff', fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Stage 3</div>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: 'var(--preview-pink)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 11.08V12a10 10 0 11-5.93-9.14"/>
              <polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
          </div>
        </div>
        <h1 className="page-title">数据质量评估</h1>
        <p className="page-subtitle">
          {wasAugmented
            ? '已应用数据增强，统计基于增强后标注'
            : '查看数据集统计与类别分布，确认无误后进入训练阶段'}
        </p>
      </div>

      {/* VLM Annotation QC */}
      <div className="card-section" style={{ marginBottom: 24, background: 'rgba(10,114,239,0.03)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, marginBottom: 10 }}>
          <div>
            <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 2 }}>VLM 标注智能质检</h3>
            <p style={{ fontSize: 12, color: 'var(--gray-400)', margin: 0 }}>
              用视觉语言模型自动分析标注质量，识别框不准、漏标、错标等问题
            </p>
          </div>
          <button
            className="btn btn-secondary"
            onClick={handleVlmQc}
            disabled={vlmQcLoading}
            style={{ padding: '8px 16px', fontSize: 13, flexShrink: 0 }}
          >
            {vlmQcLoading ? (
              <><div className="spinner" style={{ width: 12, height: 12, borderWidth: 2 }} /> 分析中...</>
            ) : (
              <><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg> 智能质检</>
            )}
          </button>
        </div>
        {vlmQcError && (
          <div style={{ padding: '10px 14px', background: 'rgba(255,91,79,0.06)', border: '1px solid rgba(255,91,79,0.2)', borderRadius: 8, fontSize: 13, color: 'var(--ship-red)' }}>
            {vlmQcError}
          </div>
        )}
        {vlmQcResult && (
          <div style={{ padding: '14px 16px', background: 'rgba(16,185,129,0.06)', border: '1px solid rgba(16,185,129,0.2)', borderRadius: 8, fontSize: 13, color: 'var(--gray-700)', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
            {vlmQcResult}
          </div>
        )}
      </div>

      {qualityReport && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginBottom: 24 }}>
          {/* Stats */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
            {[
              { label: '总图片数', value: totalImages, accent: false },
              { label: 'bbox 总数', value: classDist.reduce((s, d) => s + d.boxCount, 0), accent: false },
              { label: '平均 bbox/图', value: avgBoxes.toFixed(1), accent: false },
            ].map((item) => (
              <div key={item.label} className="card-section" style={{ display: 'flex', flexDirection: 'column', gap: 4, padding: '16px 20px' }}>
                <span className="metric-label">{item.label}</span>
                <span className="metric-value" style={{ fontSize: 24 }}>{item.value}</span>
              </div>
            ))}
          </div>

          {/* Warnings */}
          {warnings.length > 0 && (
            <div style={{ padding: '12px 16px', background: '#fefce8', border: '1px solid #fef08a', borderRadius: 8 }}>
              <div className="flex items-center gap-2 mb-2">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ca8a04" strokeWidth="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                <span style={{ fontSize: 13, fontWeight: 600, color: '#ca8a04' }}>数据质量警告</span>
              </div>
              {warnings.map((w, i) => (
                <p key={i} style={{ fontSize: 12, color: '#92400e', margin: '2px 0 0 22px' }}>{w}</p>
              ))}
            </div>
          )}

          {/* Class Distribution Table */}
          {classDist.length > 0 && (
            <div className="card-section" style={{ padding: 0, overflow: 'hidden' }}>
              <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--gray-100)' }}>
                <h3 style={{ fontSize: 14, fontWeight: 600, margin: 0 }}>类别分布</h3>
              </div>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ background: 'var(--gray-50)' }}>
                    <th style={{ padding: '10px 20px', textAlign: 'left', fontSize: 12, fontWeight: 600, color: 'var(--gray-500)', borderBottom: '1px solid var(--gray-100)' }}>类别</th>
                    <th style={{ padding: '10px 20px', textAlign: 'right', fontSize: 12, fontWeight: 600, color: 'var(--gray-500)', borderBottom: '1px solid var(--gray-100)' }}>bbox 总数</th>
                    <th style={{ padding: '10px 20px', textAlign: 'right', fontSize: 12, fontWeight: 600, color: 'var(--gray-500)', borderBottom: '1px solid var(--gray-100)' }}>平均/图</th>
                  </tr>
                </thead>
                <tbody>
                  {classDist.map((d, i) => (
                    <tr key={i} style={{ borderBottom: i < classDist.length - 1 ? '1px solid var(--gray-100)' : 'none' }}>
                      <td style={{ padding: '10px 20px', fontSize: 13, fontFamily: 'var(--font-mono)', fontWeight: 500 }}>{d.className}</td>
                      <td style={{ padding: '10px 20px', textAlign: 'right', fontSize: 13, color: 'var(--gray-600)' }}>{d.boxCount}</td>
                      <td style={{ padding: '10px 20px', textAlign: 'right', fontSize: 13, color: 'var(--gray-600)' }}>{d.avgBoxesPerImage?.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Split Stats */}
      <div className="card-section" style={{ marginBottom: 32 }}>
        <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16 }}>数据集分割（8:1:1）</h3>
        <div style={{ display: 'flex', gap: 24 }}>
          {([
            { label: '训练集', value: splitStats.train, pct: Math.round(splitStats.train / (splitStats.train + splitStats.val + splitStats.test || 1) * 100) },
            { label: '验证集', value: splitStats.val, pct: Math.round(splitStats.val / (splitStats.train + splitStats.val + splitStats.test || 1) * 100) },
            { label: '测试集', value: splitStats.test, pct: Math.round(splitStats.test / (splitStats.train + splitStats.val + splitStats.test || 1) * 100) },
          ] as const).map((s) => (
            <div key={s.label} style={{ flex: 1 }}>
              <div className="progress-bar mb-2" style={{ height: 6 }}>
                <div className="progress-bar-fill" style={{ width: `${s.pct}%` }} />
              </div>
              <div className="flex justify-between items-center">
                <span style={{ fontSize: 12, color: 'var(--gray-500)' }}>{s.label}</span>
                <span style={{ fontSize: 18, fontWeight: 700, letterSpacing: '-0.5px' }}>{s.value}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        <button
          className="btn btn-primary"
          onClick={() => setStage('offline_validation')}
          style={{ padding: '10px 24px', fontWeight: 600 }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          确认并前往离线验证
        </button>
        <button className="btn btn-secondary" onClick={() => setStage('augment')} style={{ padding: '10px 20px' }}>
          返回修改增强
        </button>
      </div>
    </div>
  )
}

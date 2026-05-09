import { useEffect, useMemo, useState } from 'react'
import { useTaskStore } from '../store/taskStore'
import { filesApi, trainingApi, TrainingReport, taskApi, algorithmApi } from '../api/backend'
import TrainingHistory from '../components/TrainingHistory'

interface MultiModelArtifact {
  source: 'trained' | 'reuse' | 'failed'
  model_id: string
  role: string
  cache_id?: string
  weight_path?: string
  artifacts?: Record<string, string>
}

const ROLE_LABELS: Record<string, string> = {
  primary_detector: '主检测器',
  secondary_detector: '辅助检测器',
  classifier: '分类器',
  feature_matcher: '特征匹配器',
  tracker: '目标跟踪器',
  rule_engine: '规则引擎',
}

export default function Delivery() {
  const { taskId, trainConfig, artifacts, reset, setStage, setTrainConfig } = useTaskStore()
  const [packageReady, setPackageReady] = useState(false)
  const [packageLoading, setPackageLoading] = useState(false)
  const [artifactList, setArtifactList] = useState<Array<{ name: string; path: string; size: number }>>([])
  const [multiModelArtifacts, setMultiModelArtifacts] = useState<Record<string, MultiModelArtifact> | null>(null)
  const [report, setReport] = useState<TrainingReport | null>(null)
  const [reportLoading, setReportLoading] = useState(false)
  const [reportError, setReportError] = useState<string | null>(null)

  useEffect(() => {
    if (!taskId) {
      setArtifactList([])
      return
    }
    filesApi.getArtifacts(taskId).then(setArtifactList).catch(() => setArtifactList([]))
  }, [taskId, packageReady, artifacts])

  useEffect(() => {
    if (!taskId) return
    trainingApi.getStatus(taskId)
      .then((s) => setMultiModelArtifacts(s.multi_model_artifacts ?? null))
      .catch(() => setMultiModelArtifacts(null))
  }, [taskId, packageReady])

  // Auto-archive training version on delivery page mount
  useEffect(() => {
    if (!taskId) return
    trainingApi.archiveVersion(taskId).catch(() => {})
  }, [taskId])

  const availableArtifacts = useMemo(() => {
    const names = new Set<string>(Object.keys(artifacts))
    artifactList.forEach((item) => names.add(item.name))
    return names
  }, [artifactList, artifacts])

  const handleDownload = async (filename: string) => {
    if (!taskId) {
      return
    }
    try {
      const blob = await filesApi.downloadArtifact(taskId, filename)
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
    } catch (e) {
      alert(`下载失败: ${e}`)
    }
  }

  const handleExportPackage = async () => {
    if (!taskId) {
      return
    }
    setPackageLoading(true)
    try {
      await algorithmApi.exportPackage(taskId)
      setPackageReady(true)
    } catch (e) {
      alert(`导出算法工程包失败: ${e}`)
    } finally {
      setPackageLoading(false)
    }
  }

  const handleGetReport = async () => {
    if (!taskId) return
    setReportLoading(true)
    setReportError(null)
    try {
      const r = await trainingApi.getReport(taskId)
      setReport(r)
    } catch (e) {
      setReportError(e instanceof Error ? e.message : 'VLM 解读失败')
    } finally {
      setReportLoading(false)
    }
  }

  const weightFiles = [
    { key: 'best.pt', label: 'best.pt', desc: '最佳权重 — 验证集上 mAP 最高的模型', badge: 'Recommended' },
    { key: 'last.pt', label: 'last.pt', desc: '最终权重 — 最后一个 epoch 的模型', badge: null },
  ].filter(({ key }) => availableArtifacts.has(key))

  const exportFiles = trainConfig.exportFormats
    .map((fmt) => `model.${fmt}`)
    .filter((name) => availableArtifacts.has(name))

  const reportFiles = ['results.csv', 'confusion_matrix.png', 'PR_curve.png', 'F1_curve.png', 'results.png']
    .filter((name) => availableArtifacts.has(name))

  return (
    <div>
      {/* Page Header */}
      <div className="page-header" style={{ marginBottom: 32 }}>
        <div className="flex items-center gap-3 mb-4">
          <div style={{ width: 48, height: 48, borderRadius: 12, background: 'linear-gradient(135deg, #16a34a, #15803d)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 4px 12px rgba(22,163,74,0.3)' }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="20 6 9 17 4 12"/>
            </svg>
          </div>
          <div>
            <div className="badge badge-green mb-2" style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Training Complete</div>
            <h1 className="page-title" style={{ marginBottom: 0 }}>模型交付</h1>
          </div>
        </div>
        <p className="page-subtitle">恭喜！模型训练完成，下载所需格式的模型文件</p>
      </div>

      {/* Success Banner */}
      <div
        style={{
          background: 'linear-gradient(135deg, #f0fdf4, #dcfce7)',
          border: '1px solid #bbf7d0',
          borderRadius: 12,
          padding: '16px 20px',
          marginBottom: 24,
          display: 'flex',
          alignItems: 'center',
          gap: 12,
        }}
      >
        <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#16a34a', flexShrink: 0, animation: 'pulse 2s ease-in-out infinite' }} />
        <p style={{ fontSize: 14, color: '#15803d', margin: 0, fontWeight: 500 }}>
          模型训练已成功完成！以下是训练产物，可直接下载使用
        </p>
      </div>

      {/* Model Files */}
      <div className="card-section" style={{ marginBottom: 16 }}>
        <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-500)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 12 }}>模型权重</h3>
        {weightFiles.length === 0 && (
          <p style={{ fontSize: 12, color: 'var(--gray-500)', margin: 0 }}>
            当前还没有可下载的模型权重，请先完成云训练并等待产物回收。
          </p>
        )}
        {weightFiles.map(({ key, label, desc, badge }) => (
          <div
            key={key}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '14px 0',
              borderBottom: '1px solid var(--gray-100)',
            }}
          >
            <div className="flex items-center gap-3">
              <div style={{ width: 36, height: 36, borderRadius: 8, background: 'var(--gray-100)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--gray-500)" strokeWidth="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <p style={{ fontSize: 14, fontFamily: 'var(--font-mono)', fontWeight: 600, margin: 0 }}>{label}</p>
                  {badge && <span className="badge badge-green" style={{ fontSize: 10 }}>{badge}</span>}
                </div>
                <p style={{ fontSize: 12, color: 'var(--gray-400)', margin: '2px 0 0' }}>{desc}</p>
              </div>
            </div>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => handleDownload(key)}
              style={{ flexShrink: 0 }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              下载
            </button>
          </div>
        ))}
      </div>

      {multiModelArtifacts && Object.keys(multiModelArtifacts).length > 0 && (
        <div className="card-section" style={{ marginBottom: 16 }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-500)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 12 }}>多模型流水线</h3>
          <p style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 16, lineHeight: 1.6 }}>
            本方案由多个模型协同工作，生成算法工程包时会自动将它们按 <code style={{ background: 'var(--gray-100)', padding: '1px 6px', borderRadius: 3, fontFamily: 'var(--font-mono)', fontSize: 11 }}>models/&lt;step_id&gt;/</code> 分子目录打包。
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {Object.entries(multiModelArtifacts).map(([stepId, entry]) => (
              <div
                key={stepId}
                style={{
                  padding: '12px 14px',
                  borderRadius: 8,
                  background: entry.source === 'reuse' ? 'rgba(22,163,74,0.04)' : entry.source === 'failed' ? '#fff5f5' : 'var(--gray-50)',
                  border: entry.source === 'failed' ? '1px solid #fed7d7' : '1px solid var(--gray-100)',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 6 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                    <span className="badge badge-blue" style={{ fontSize: 10 }}>{ROLE_LABELS[entry.role] ?? entry.role}</span>
                    <strong style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color: 'var(--gray-900)' }}>{entry.model_id}</strong>
                    {entry.source === 'reuse' && <span className="badge badge-green" style={{ fontSize: 10 }}>复用已有模型</span>}
                    {entry.source === 'trained' && <span className="badge badge-dark" style={{ fontSize: 10 }}>本次训练</span>}
                    {entry.source === 'failed' && <span className="badge badge-pink" style={{ fontSize: 10 }}>训练失败</span>}
                  </div>
                  <span style={{ fontSize: 10, color: 'var(--gray-400)', fontFamily: 'var(--font-mono)' }}>{stepId}</span>
                </div>
                {entry.artifacts && Object.keys(entry.artifacts).length > 0 && (
                  <div style={{ fontSize: 11, color: 'var(--gray-500)', lineHeight: 1.7 }}>
                    产物：{Object.keys(entry.artifacts).join(', ')}
                  </div>
                )}
                {entry.source === 'reuse' && entry.cache_id && (
                  <div style={{ fontSize: 11, color: '#15803d', lineHeight: 1.7 }}>
                    缓存标识：<code style={{ fontFamily: 'var(--font-mono)' }}>{entry.cache_id}</code>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Export Formats */}
      {exportFiles.length > 0 && (
        <div className="card-section" style={{ marginBottom: 16 }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-500)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 12 }}>导出格式</h3>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {exportFiles.map((name) => (
              <button
                key={name}
                className="btn btn-secondary"
                onClick={() => handleDownload(name)}
                style={{ padding: '8px 16px', fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 600 }}
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                {name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Report */}
      <div className="card-section" style={{ marginBottom: 16 }}>
        <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-500)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 12 }}>训练报告</h3>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: reportFiles.length > 0 ? 16 : 0 }}>
          {reportFiles.length === 0 && !report && (
            <p style={{ fontSize: 12, color: 'var(--gray-500)', margin: 0 }}>
              当前没有训练报告文件。
            </p>
          )}
          {reportFiles.map((name) => (
            <button key={name} className="btn btn-secondary" onClick={() => handleDownload(name)} style={{ padding: '8px 16px' }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
              下载 {name}
            </button>
          ))}
        </div>
        {reportFiles.length > 0 && !report && (
          <div>
            <button
              className="btn btn-primary"
              onClick={handleGetReport}
              disabled={reportLoading}
              style={{ padding: '8px 20px', fontSize: 13 }}
            >
              {reportLoading ? (
                <>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ animation: 'spin 1s linear infinite' }}><path d="M21 12a9 9 0 11-6.219-8.56"/></svg>
                  AI 解读中...
                </>
              ) : (
                <>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
                  AI 解读训练报告
                </>
              )}
            </button>
            {reportError && <p style={{ fontSize: 12, color: '#dc2626', marginTop: 8 }}>{reportError}</p>}
          </div>
        )}
        {report && (
          <div style={{ marginTop: 8, padding: '16px 20px', borderRadius: 10, background: 'linear-gradient(135deg, #f0f9ff, #eff6ff)', border: '1px solid #bfdbfe' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
              <h4 style={{ fontSize: 14, fontWeight: 700, color: 'var(--gray-900)', margin: 0 }}>AI 训练报告解读</h4>
              {report.score != null && (
                <div style={{
                  padding: '4px 14px', borderRadius: 20, fontWeight: 700, fontSize: 13,
                  background: report.score >= 80 ? '#dcfce7' : report.score >= 60 ? '#fef9c3' : '#fee2e2',
                  color: report.score >= 80 ? '#15803d' : report.score >= 60 ? '#92400e' : '#dc2626',
                }}>
                  {report.score} 分
                </div>
              )}
            </div>
            {report.overall_assessment_zh && (
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--gray-500)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.05em' }}>总体评价</div>
                <p style={{ fontSize: 13, color: 'var(--gray-800)', lineHeight: 1.7, margin: 0, whiteSpace: 'pre-wrap' }}>{report.overall_assessment_zh}</p>
              </div>
            )}
            {report.convergence_zh && (
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--gray-500)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.05em' }}>收敛分析</div>
                <p style={{ fontSize: 13, color: 'var(--gray-800)', lineHeight: 1.7, margin: 0, whiteSpace: 'pre-wrap' }}>{report.convergence_zh}</p>
              </div>
            )}
            {report.class_performance_zh && (
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--gray-500)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.05em' }}>类别表现</div>
                <p style={{ fontSize: 13, color: 'var(--gray-800)', lineHeight: 1.7, margin: 0, whiteSpace: 'pre-wrap' }}>{report.class_performance_zh}</p>
              </div>
            )}
            {report.improvement_suggestions_zh && report.improvement_suggestions_zh.length > 0 && (
              <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--gray-500)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.05em' }}>改进建议</div>
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  {report.improvement_suggestions_zh.map((s, i) => (
                    <li key={i} style={{ fontSize: 13, color: 'var(--gray-800)', lineHeight: 1.8 }}>{s}</li>
                  ))}
                </ul>
              </div>
            )}
            {report.charts_analyzed && report.charts_analyzed.length > 0 && (
              <div style={{ fontSize: 11, color: 'var(--gray-400)', marginTop: 8 }}>
                已分析图表：{report.charts_analyzed.join(', ')}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="card-section" style={{ marginBottom: 32 }}>
        <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-500)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 12 }}>算法工程包</h3>
        <p style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 12, lineHeight: 1.6 }}>
          导出 `pipeline.json`、`manifest.json`、`README.md` 和运行入口脚本，用于后续集成与二次开发。
        </p>
        <div className="flex gap-3" style={{ marginBottom: packageReady ? 14 : 0 }}>
          <button className="btn btn-secondary" onClick={handleExportPackage} disabled={packageLoading || !taskId}>
            {packageLoading ? '导出中...' : '生成算法工程包'}
          </button>
        </div>
        {packageReady && (
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {['pipeline.json', 'manifest.json', 'README.md', 'run_pipeline.py', 'sample_input.json', 'sample_output.json'].filter((name) => availableArtifacts.has(name) || packageReady).map((name) => (
              <button
                key={name}
                className="btn btn-secondary"
                onClick={() => handleDownload(name)}
                style={{ padding: '8px 16px', fontFamily: 'var(--font-mono)', fontSize: 12 }}
              >
                下载 {name}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Training History */}
      <div style={{ marginBottom: 24 }}>
        <TrainingHistory />
      </div>

      {/* Actions */}
      <div className="flex gap-3" style={{ flexWrap: 'wrap' }}>
        <button
          className="btn btn-primary"
          onClick={() => {
            setTrainConfig({ incrementalMode: true, baseModelPath: null })
            setStage('upload')
          }}
          style={{ padding: '10px 24px', fontWeight: 600, borderColor: '#f59e0b', background: '#f59e0b' }}
          title="上传新的 badcase 图片，基于当前模型增量微调"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
          追加数据 & 增量训练
        </button>
        <button
          className="btn btn-primary"
          onClick={reset}
          style={{ padding: '10px 24px', fontWeight: 600 }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 11-2.12-9.36L23 10"/></svg>
          开始新任务
        </button>
        <button
          className="btn btn-secondary"
          onClick={async () => {
            if (!taskId) return
            try {
              const cloned = await taskApi.clone(taskId)
              alert(`任务已克隆：${cloned.name} (ID: ${cloned.id})`)
            } catch (e) {
              alert(`克隆失败: ${e}`)
            }
          }}
          disabled={!taskId}
          style={{ padding: '10px 20px' }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
          克隆任务
        </button>
        <button
          className="btn btn-secondary"
          onClick={async () => {
            if (!taskId) return
            try {
              const tpl = await taskApi.clone(taskId, { as_template: true })
              alert(`已保存为模板：${tpl.name} (ID: ${tpl.id})`)
            } catch (e) {
              alert(`保存模板失败: ${e}`)
            }
          }}
          disabled={!taskId}
          style={{ padding: '10px 20px' }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>
          保存为模板
        </button>
        <button className="btn btn-secondary" onClick={() => window.location.reload()} style={{ padding: '10px 20px' }}>
          刷新页面
        </button>
      </div>
    </div>
  )
}

import { useRef, useState } from 'react'
import { useTaskStore } from '../store/taskStore'
import { videoValidationApi, type VideoValidationResult } from '../api/backend'

export default function OfflineValidation() {
  const { taskId, setStage } = useTaskStore()
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [validating, setValidating] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [result, setResult] = useState<VideoValidationResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelected = (files: FileList | null) => {
    if (!files || files.length === 0) return
    setVideoFile(files[0])
    setResult(null)
    setError(null)
  }

  const handleValidate = async () => {
    if (!taskId || !videoFile) return
    setValidating(true)
    setError(null)
    setResult(null)
    setUploadProgress(0)
    try {
      const formData = new FormData()
      formData.append('video', videoFile)
      const res = await videoValidationApi.validateWithProgress(taskId, formData, (pct) => {
        setUploadProgress(pct)
      })
      setResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : '视频验证失败')
    } finally {
      setValidating(false)
    }
  }

  const handleSkip = () => setStage('train_config')
  const handleProceed = () => setStage('train_config')
  const handleBackToRevise = () => setStage('algorithm_plan')

  return (
    <div>
      <div className="page-header" style={{ marginBottom: 32 }}>
        <div className="flex items-center gap-3 mb-4">
          <div className="badge badge-pink" style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Stage 3.5</div>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: 'var(--gray-900)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
          </div>
        </div>
        <h1 className="page-title">离线视频验证</h1>
        <p className="page-subtitle">训练之前，先让 VLM 看一段真实现场视频，判断当前方案是否能覆盖你的场景。通过后再开始训练，避免白跑一遍。</p>
      </div>

      <div className="card-section" style={{ marginBottom: 24 }}>
        <h3 className="text-heading-sm" style={{ marginBottom: 12 }}>上传现场视频</h3>
        <p style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 16, lineHeight: 1.7 }}>
          请上传一段和部署现场类似的视频（10 秒到 5 分钟），系统会自动抽 8 帧发给 VLM 分析。
        </p>
        <input
          ref={fileInputRef}
          type="file"
          accept="video/mp4,video/avi,video/mov,video/mkv,video/webm"
          onChange={(e) => handleFileSelected(e.target.files)}
          style={{ display: 'none' }}
        />
        <div
          onClick={() => fileInputRef.current?.click()}
          style={{
            padding: videoFile ? '16px 20px' : '32px 20px',
            borderRadius: 8,
            border: videoFile ? '1px solid var(--develop-blue)' : '1px dashed var(--gray-200)',
            background: videoFile ? 'rgba(10,114,239,0.04)' : 'var(--gray-50)',
            cursor: 'pointer',
          }}
        >
          {videoFile ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-900)' }}>{videoFile.name}</div>
              <div style={{ fontSize: 11, color: 'var(--gray-400)' }}>{(videoFile.size / 1024 / 1024).toFixed(1)} MB · 点击可重新选择</div>
            </div>
          ) : (
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: 13, color: 'var(--gray-600)', margin: 0 }}>点击选择视频文件</p>
              <p style={{ fontSize: 11, color: 'var(--gray-400)', margin: '6px 0 0' }}>MP4/AVI/MOV · 建议 10 秒~5 分钟</p>
            </div>
          )}
        </div>

        <div className="flex gap-3" style={{ marginTop: 16 }}>
          <button
            className="btn btn-primary"
            onClick={handleValidate}
            disabled={!videoFile || validating}
            style={{ padding: '10px 20px', fontSize: 13, fontWeight: 600 }}
          >
            {validating ? (
              <>
                <div className="spinner" />
                {uploadProgress > 0 && uploadProgress < 100
                  ? `上传中 ${uploadProgress}%...`
                  : 'VLM 分析中（需要 30~60 秒）...'}
              </>
            ) : (
              '开始验证'
            )}
          </button>
          <button
            className="btn btn-secondary"
            onClick={handleSkip}
            disabled={validating}
            style={{ padding: '10px 20px', fontSize: 13 }}
          >
            跳过验证，直接训练
          </button>
        </div>
      </div>

      {error && (
        <div style={{ marginBottom: 16, padding: '12px 14px', background: '#fff5f5', borderRadius: 8, boxShadow: 'var(--shadow-border)', color: 'var(--ship-red)', fontSize: 13 }}>
          {error}
        </div>
      )}

      {result && (
        <div
          className="card-section"
          style={{
            marginBottom: 24,
            background:
              result.validation_passed === true
                ? 'rgba(22,163,74,0.04)'
                : result.validation_passed === false
                  ? '#fff5f5'
                  : 'var(--gray-50)',
            border:
              result.validation_passed === true
                ? '1px solid rgba(22,163,74,0.2)'
                : result.validation_passed === false
                  ? '1px solid #fed7d7'
                  : '1px solid var(--gray-100)',
            boxShadow: 'none',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
            <h3 className="text-heading-sm" style={{ margin: 0 }}>验证结果</h3>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              {result.validation_passed === true && <span className="badge badge-green">方案可行</span>}
              {result.validation_passed === false && <span className="badge badge-pink">需要调整</span>}
              {result.validation_passed === null && <span className="badge badge-dark">需要人工判断</span>}
              <span style={{ fontSize: 11, color: 'var(--gray-500)' }}>
                置信度 {Math.round(result.confidence * 100)}% · 分析 {result.frame_count_analyzed} 帧
              </span>
            </div>
          </div>

          <div style={{ fontSize: 13, color: 'var(--gray-700)', lineHeight: 1.7, whiteSpace: 'pre-wrap', marginBottom: 12 }}>
            {result.analysis_zh}
          </div>

          {result.suggestions_zh && result.suggestions_zh.length > 0 && (
            <div style={{ paddingTop: 12, borderTop: '1px solid var(--gray-100)' }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--gray-900)', marginBottom: 8 }}>建议</div>
              <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: 'var(--gray-600)', lineHeight: 1.7 }}>
                {result.suggestions_zh.map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      <div className="flex gap-3">
        {result?.validation_passed !== false && (
          <button
            className="btn btn-primary"
            onClick={handleProceed}
            disabled={validating}
            style={{ padding: '10px 24px', fontSize: 14, fontWeight: 600 }}
          >
            继续训练
          </button>
        )}
        {result?.validation_passed === false && (
          <button
            className="btn btn-primary"
            onClick={handleBackToRevise}
            style={{ padding: '10px 24px', fontSize: 14, fontWeight: 600 }}
          >
            返回修改方案
          </button>
        )}
        <button
          className="btn btn-secondary"
          onClick={() => setStage('review')}
          disabled={validating}
          style={{ padding: '10px 20px' }}
        >
          返回上一步
        </button>
      </div>
    </div>
  )
}

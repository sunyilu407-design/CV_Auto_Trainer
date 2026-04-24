import { useRef, useState } from 'react'
import { useTaskStore } from '../store/taskStore'
import { videoInferenceApi, type VideoInferenceFrame } from '../api/backend'

export default function VideoInferenceDemo() {
  const { taskId, artifacts, setStage } = useTaskStore()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [frames, setFrames] = useState<VideoInferenceFrame[]>([])
  const [currentIdx, setCurrentIdx] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const handleFileSelected = (files: FileList | null) => {
    if (!files || files.length === 0) return
    setVideoFile(files[0])
    setFrames([])
    setCurrentIdx(0)
    setError(null)
  }

  const handleRun = async () => {
    if (!taskId || !videoFile) return
    setRunning(true)
    setError(null)
    setFrames([])
    try {
      const result = await videoInferenceApi.run(taskId, videoFile, (pct) => setProgress(pct))
      setFrames(result.frames)
      setCurrentIdx(0)
    } catch (e) {
      setError(e instanceof Error ? e.message : '视频推理失败')
    } finally {
      setRunning(false)
    }
  }

  const currentFrame = frames[currentIdx]
  const bestWeight = artifacts?.['best.pt'] || artifacts?.bestMap || ''

  return (
    <div>
      {/* Page Header */}
      <div className="page-header" style={{ marginBottom: 32 }}>
        <div className="flex items-center gap-3 mb-4">
          <div className="badge badge-red" style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Demo</div>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: 'var(--ship-red)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
              <polygon points="23 7 16 12 23 5 21 2 16 12 21 22 23 17 16 22 8 13 16 8 3 17 8 12 3 17 8 12"/>
            </svg>
          </div>
        </div>
        <h1 className="page-title">视频推理演示</h1>
        <p className="page-subtitle">用训练好的模型在真实视频上推理，直观展示识别效果</p>
      </div>

      {/* Weight info */}
      {bestWeight && (
        <div className="card-section" style={{ marginBottom: 16, background: 'rgba(16,185,129,0.04)', border: '1px solid rgba(16,185,129,0.2)' }}>
          <div style={{ fontSize: 12, color: '#15803d', fontWeight: 600 }}>
            使用模型权重：{bestWeight}
          </div>
        </div>
      )}

      {/* Video selection */}
      <div className="card-section" style={{ marginBottom: 24 }}>
        <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 12 }}>上传测试视频</h3>
        <p style={{ fontSize: 12, color: 'var(--gray-400)', marginBottom: 16, lineHeight: 1.6 }}>
          选择一段和实际部署场景类似的视频，系统会用训练好的模型推理并标注检测结果。
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
            textAlign: 'center',
          }}
        >
          {videoFile ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-900)' }}>{videoFile.name}</div>
              <div style={{ fontSize: 11, color: 'var(--gray-400)' }}>{(videoFile.size / 1024 / 1024).toFixed(1)} MB · 点击重新选择</div>
            </div>
          ) : (
            <div>
              <div style={{ fontSize: 13, color: 'var(--gray-600)' }}>点击选择视频文件</div>
              <div style={{ fontSize: 11, color: 'var(--gray-400)', marginTop: 6 }}>MP4/AVI/MOV/MKV · 建议 10 秒到 2 分钟</div>
            </div>
          )}
        </div>

        {running && (
          <div style={{ marginTop: 16 }}>
            <div style={{ fontSize: 12, color: 'var(--gray-600)', marginBottom: 8 }}>
              {progress > 0 && progress < 100 ? `上传中 ${progress}%...` : '正在推理...'}
            </div>
            <div className="progress-bar" style={{ height: 4, borderRadius: 2 }}>
              <div className="progress-bar-fill animated" style={{ width: `${progress}%`, background: 'var(--develop-blue)' }} />
            </div>
          </div>
        )}

        {error && (
          <div style={{ marginTop: 12, padding: '10px 14px', background: 'rgba(255,91,79,0.06)', border: '1px solid rgba(255,91,79,0.2)', borderRadius: 8, fontSize: 13, color: 'var(--ship-red)' }}>
            {error}
          </div>
        )}

        <button
          className="btn btn-primary"
          onClick={handleRun}
          disabled={!videoFile || running}
          style={{ marginTop: 16, padding: '10px 24px', fontWeight: 600 }}
        >
          {running ? (
            <><div className="spinner" /> 推理中...</>
          ) : (
            <><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> 开始推理</>
          )}
        </button>
      </div>

      {/* Results */}
      {frames.length > 0 && (
        <div className="card-section">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <div>
              <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 2 }}>推理结果</h3>
              <p style={{ fontSize: 12, color: 'var(--gray-400)', margin: 0 }}>共 {frames.length} 帧 · {currentIdx + 1} / {frames.length}</p>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                onClick={() => setCurrentIdx(Math.max(0, currentIdx - 1))}
                disabled={currentIdx === 0}
                style={{ padding: '6px 12px', borderRadius: 6, border: '1px solid var(--gray-200)', background: 'white', cursor: 'pointer', fontSize: 12 }}
              >
                ‹ 上一帧
              </button>
              <button
                onClick={() => setCurrentIdx(Math.min(frames.length - 1, currentIdx + 1))}
                disabled={currentIdx === frames.length - 1}
                style={{ padding: '6px 12px', borderRadius: 6, border: '1px solid var(--gray-200)', background: 'white', cursor: 'pointer', fontSize: 12 }}
              >
                下一帧 ›
              </button>
            </div>
          </div>

          {currentFrame && (
            <div>
              <img
                src={`data:image/jpeg;base64,${currentFrame.frame_b64}`}
                alt={`帧 ${currentFrame.frame_idx}`}
                style={{ width: '100%', maxHeight: 480, objectFit: 'contain', borderRadius: 8, background: '#000' }}
              />
              <div style={{ marginTop: 12, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                <div style={{ fontSize: 11, color: 'var(--gray-500)' }}>
                  时间 {(currentFrame.timestamp_ms / 1000).toFixed(1)}s · 帧 #{currentFrame.frame_idx}
                </div>
                {currentFrame.source && (
                  <div style={{
                    fontSize: 11,
                    fontWeight: 600,
                    padding: '2px 8px',
                    borderRadius: 20,
                    background: currentFrame.source === 'keyframe' ? 'rgba(59,130,246,0.1)' : 'rgba(107,114,128,0.1)',
                    color: currentFrame.source === 'keyframe' ? '#2563eb' : '#6b7280',
                  }}>
                    {currentFrame.source === 'keyframe' ? '关键帧' : '均匀采样'}
                  </div>
                )}
                {currentFrame.detections.map((d, i) => (
                  <div key={i} style={{
                    display: 'inline-flex', alignItems: 'center', gap: 4,
                    padding: '4px 10px', borderRadius: 4,
                    background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.2)',
                    fontSize: 12, color: '#15803d',
                  }}>
                    <span style={{ fontWeight: 600 }}>{d.class}</span>
                    <span style={{ opacity: 0.7 }}>{(d.conf * 100).toFixed(0)}%</span>
                  </div>
                ))}
                {currentFrame.detections.length === 0 && (
                  <div style={{ fontSize: 12, color: 'var(--ship-red)', fontStyle: 'italic' }}>未检出目标</div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-3 mt-8">
        <button
          className="btn btn-primary"
          onClick={() => setStage('delivery')}
          style={{ padding: '10px 24px', fontWeight: 600 }}
        >
          查看交付物
        </button>
        <button
          className="btn btn-secondary"
          onClick={() => setStage('training')}
          style={{ padding: '10px 20px' }}
        >
          返回训练监控
        </button>
      </div>
    </div>
  )
}

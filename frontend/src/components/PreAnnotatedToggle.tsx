import { useEffect, useState } from 'react'
import { useTaskStore } from '../store/taskStore'
import { filesApi } from '../api/backend'

export default function PreAnnotatedToggle() {
  const { taskId, skipLabeling, setSkipLabeling } = useTaskStore()
  const [checking, setChecking] = useState(false)
  const [checkResult, setCheckResult] = useState<{
    has_annotations: boolean
    total_images: number
    annotated_images: number
    total_boxes: number
    detected_classes: number[]
    message: string
  } | null>(null)
  const [checkError, setCheckError] = useState<string | null>(null)

  useEffect(() => {
    if (!taskId) {
      setCheckResult(null)
      return
    }
    setChecking(true)
    setCheckError(null)
    filesApi.checkExistingAnnotations(taskId)
      .then((result) => {
        setCheckResult(result)
      })
      .catch(() => {
        // 非致命错误，可能是因为还没有上传图片
        setCheckError('未检测到预标注数据')
      })
      .finally(() => {
        setChecking(false)
      })
  }, [taskId])

  if (!taskId) return null

  if (checking) {
    return (
      <div
        className="card-section"
        style={{
          marginBottom: 16,
          padding: '12px 16px',
          background: 'var(--gray-50)',
          border: '1px solid var(--gray-100)',
          boxShadow: 'none',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 14, height: 14, border: '2px solid var(--gray-200)', borderTopColor: 'var(--develop-blue)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
          <span style={{ fontSize: 13, color: 'var(--gray-500)' }}>正在检测预标注数据...</span>
        </div>
      </div>
    )
  }

  const detected = checkResult?.has_annotations

  return (
    <div
      className="card-section"
      style={{
        marginBottom: 16,
        padding: '14px 16px',
        background: detected ? 'rgba(16,185,129,0.05)' : 'var(--gray-50)',
        border: `1px solid ${detected ? 'rgba(16,185,129,0.25)' : 'var(--gray-100)'}`,
        boxShadow: 'none',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
        {/* Icon */}
        <div
          style={{
            width: 36,
            height: 36,
            borderRadius: 8,
            background: detected ? 'rgba(16,185,129,0.12)' : 'var(--gray-100)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke={detected ? '#10b981' : 'var(--gray-400)'}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="16" y1="13" x2="8" y2="13" />
            <line x1="16" y1="17" x2="8" y2="17" />
            <polyline points="10 9 9 9 8 9" />
          </svg>
        </div>

        {/* Content */}
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 6 }}>
            <div>
              <p style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-900)', margin: 0 }}>
                我已用 LabelImg 标注好数据
              </p>
            <p style={{ fontSize: 11, color: 'var(--gray-500)', margin: '3px 0 0' }}>
              {checkError
                ? '上传标注好的图片后将自动检测 YOLO 标注文件'
                : checkResult
                  ? detected
                    ? `检测到 ${checkResult.annotated_images}/${checkResult.total_images} 张已标注图片，共 ${checkResult.total_boxes} 个框`
                    : '当前目录下未发现 YOLO 标注文件'
                  : ''}
            </p>
            </div>

            {/* Toggle */}
            <button
              onClick={() => setSkipLabeling(!skipLabeling)}
              style={{
                width: 44,
                height: 24,
                borderRadius: 12,
                border: 'none',
                background: skipLabeling ? '#10b981' : 'var(--gray-200)',
                cursor: 'pointer',
                padding: 2,
                flexShrink: 0,
                transition: 'background 0.2s ease',
                position: 'relative',
              }}
            >
              <div
                style={{
                  width: 20,
                  height: 20,
                  borderRadius: '50%',
                  background: '#fff',
                  transition: 'transform 0.2s ease',
                  transform: skipLabeling ? 'translateX(20px)' : 'translateX(0)',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.15)',
                }}
              />
            </button>
          </div>

          {/* Active state explanation */}
          {skipLabeling && (
            <div
              style={{
                marginTop: 10,
                padding: '8px 12px',
                borderRadius: 6,
                background: 'rgba(16,185,129,0.08)',
                border: '1px solid rgba(16,185,129,0.2)',
              }}
            >
              <p style={{ fontSize: 12, color: '#065f46', margin: 0, lineHeight: 1.6 }}>
                <strong>跳过自动打标</strong> — 系统将直接使用你上传的 YOLO .txt 标注文件，跳过 YOLO-World 检测和 Moondream VQA 质检两个阶段，节省大量时间。
              </p>
              {detected && (
                <p style={{ fontSize: 11, color: '#047857', margin: '4px 0 0', lineHeight: 1.6 }}>
                  检测到的类别索引：{checkResult!.detected_classes.join(', ')}。请确保类别数量与下方监测对象数量一致。
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

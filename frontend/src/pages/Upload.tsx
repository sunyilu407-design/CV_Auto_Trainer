import { useState, useRef, useCallback } from 'react'
import { useTaskStore } from '../store/taskStore'
import { vlmApi } from '../api/backend'
import AnnotationCanvas from '../components/AnnotationCanvas'

export default function Upload() {
  const { setStage, setVLMResult, sampleImages, setSampleImages, userDescription, setUserDescription } = useTaskStore()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeImageIndex, setActiveImageIndex] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFilesSelected = useCallback((files: FileList | null) => {
    if (!files) return
    const newFiles = Array.from(files).slice(0, 3 - sampleImages.length)
    setSampleImages([...sampleImages, ...newFiles])
  }, [sampleImages, setSampleImages])

  const handleParse = async () => {
    if (sampleImages.length === 0 || !userDescription.trim()) {
      setError('请上传样板图并输入描述')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const imagesBase64 = await Promise.all(
        sampleImages.map((f) =>
          new Promise<string>((resolve, reject) => {
            const reader = new FileReader()
            reader.onload = () => {
              const result = reader.result as string
              resolve(result.split(',')[1])
            }
            reader.onerror = reject
            reader.readAsDataURL(f)
          })
        )
      )

      const result = await vlmApi.parse(imagesBase64, userDescription)
      setVLMResult({
        classes: result.classes as never[],
        raw_vlm_response: '',
        confidence: result.confidence,
      } as never)
      setStage('intent_confirm')
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '解析失败')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: '900px', margin: '0 auto' }}>
      <h2 style={{ fontSize: '20px', marginBottom: '20px' }}>阶段一：上传样板图 + 描述</h2>

      <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
        {/* 图片上传区 */}
        <div style={{ flex: '1 1 400px' }}>
          <div
            style={{
              border: '2px dashed #ccc',
              borderRadius: '8px',
              padding: '24px',
              textAlign: 'center',
              minHeight: '300px',
              background: '#fff',
            }}
          >
            {sampleImages.length === 0 ? (
              <div>
                <p style={{ color: '#666', marginBottom: '12px' }}>
                  上传 1~3 张已手动画好框的样板图（JPG/PNG，单张 ≤ 10MB）
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png"
                  multiple
                  onChange={(e) => handleFilesSelected(e.target.files)}
                  style={{ display: 'none' }}
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  style={{
                    padding: '8px 20px',
                    background: '#1976d2',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                  }}
                >
                  选择图片
                </button>
              </div>
            ) : (
              <div>
                <div style={{ display: 'flex', gap: '8px', marginBottom: '12px', flexWrap: 'wrap' }}>
                  {sampleImages.map((img, i) => (
                    <div
                      key={i}
                      onClick={() => setActiveImageIndex(i)}
                      style={{
                        width: '80px',
                        height: '80px',
                        border: activeImageIndex === i ? '2px solid #1976d2' : '2px solid #ddd',
                        borderRadius: '4px',
                        overflow: 'hidden',
                        cursor: 'pointer',
                        position: 'relative',
                      }}
                    >
                      <img
                        src={URL.createObjectURL(img)}
                        alt={`样板图 ${i + 1}`}
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      />
                    </div>
                  ))}
                  {sampleImages.length < 3 && (
                    <>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/jpeg,image/png"
                        multiple
                        onChange={(e) => handleFilesSelected(e.target.files)}
                        style={{ display: 'none' }}
                      />
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        style={{
                          width: '80px',
                          height: '80px',
                          border: '2px dashed #ccc',
                          borderRadius: '4px',
                          background: '#f9f9f9',
                          cursor: 'pointer',
                          fontSize: '24px',
                          color: '#999',
                        }}
                      >
                        +
                      </button>
                    </>
                  )}
                </div>

                <AnnotationCanvas
                  imageUrl={URL.createObjectURL(sampleImages[activeImageIndex])}
                  onBoxesChange={() => {}}
                />

                <p style={{ fontSize: '12px', color: '#999', marginTop: '8px' }}>
                  在图上画出目标区域作为参考
                </p>
              </div>
            )}
          </div>
        </div>

        {/* 描述输入区 */}
        <div style={{ flex: '1 1 300px' }}>
          <textarea
            value={userDescription}
            onChange={(e) => setUserDescription(e.target.value)}
            placeholder="用口语描述你要检测的目标，例如：&#10;'把戴红帽子的人框出来，戴的叫 helmet，没戴的叫 no_helmet'"
            style={{
              width: '100%',
              height: '200px',
              padding: '12px',
              border: '1px solid #ddd',
              borderRadius: '4px',
              fontSize: '14px',
              resize: 'vertical',
              boxSizing: 'border-box',
            }}
          />

          {error && (
            <p style={{ color: '#d32f2f', fontSize: '14px', margin: '8px 0' }}>{error}</p>
          )}

          <button
            onClick={handleParse}
            disabled={loading}
            style={{
              marginTop: '12px',
              padding: '10px 24px',
              background: loading ? '#ccc' : '#1976d2',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: '15px',
            }}
          >
            {loading ? '解析中...' : '解析意图'}
          </button>

          <p style={{ fontSize: '12px', color: '#999', marginTop: '12px' }}>
            提示：样板图上画的框会帮助 VLM 理解你要检测的目标类型
          </p>
        </div>
      </div>
    </div>
  )
}

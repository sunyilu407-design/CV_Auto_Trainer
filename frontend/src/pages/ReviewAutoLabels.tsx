import { useState, useEffect, useCallback } from 'react'
import { useTaskStore } from '../store/taskStore'
import { filesApi } from '../api/backend'
import MultiClassAnnotationCanvas, {
  MCBox,
  ClassDef,
  classColor,
} from '../components/MultiClassAnnotationCanvas'

interface ReviewImage {
  name: string
  stem: string
}

type Decision = 'accept' | 'reject' | 'edit'

export default function ReviewAutoLabels() {
  const { taskId, vlmResult, setStage, setLabeledImageCount } = useTaskStore()

  const [images, setImages] = useState<ReviewImage[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [boxes, setBoxes] = useState<MCBox[]>([])
  const [decisions, setDecisions] = useState<Record<string, Decision>>({})
  const [loading, setLoading] = useState(true)
  const [imgNatSize, setImgNatSize] = useState<{ w: number; h: number } | null>(null)
  const [selectedClassIdx, setSelectedClassIdx] = useState(0)

  // Build class definitions
  const classes: ClassDef[] = (vlmResult?.classes || []).map((c, i) => ({
    name: c.display_name_zh || c.class_name,
    color: classColor(i),
  }))

  // Load review images list
  useEffect(() => {
    if (!taskId) return
    setLoading(true)
    // List images that have review labels
    fetch(`/api/files/${taskId}/review-labels`)
      .then((r) => r.json())
      .then((data) => {
        const list: ReviewImage[] = (data.data || []).map((name: string) => ({
          name,
          stem: name.replace(/\.[^.]+$/, ''),
        }))
        setImages(list)
      })
      .catch(() => setImages([]))
      .finally(() => setLoading(false))
  }, [taskId])

  const currentImage = images[currentIndex] || null

  // Build image URL
  const imageUrl = currentImage && taskId
    ? `/api/files/${taskId}/image/${encodeURIComponent(currentImage.name)}`
    : ''

  // Load review boxes for current image
  useEffect(() => {
    if (!taskId || !currentImage) {
      setBoxes([])
      return
    }
    fetch(`/api/files/${taskId}/review-labels/${encodeURIComponent(currentImage.stem)}`)
      .then((r) => r.json())
      .then((data) => {
        if (data.data && imgNatSize) {
          const loaded: MCBox[] = (data.data as Array<{ cls: number; cx: number; cy: number; w: number; h: number }>).map(
            (b) => ({
              classIndex: b.cls,
              x: (b.cx - b.w / 2) * imgNatSize.w,
              y: (b.cy - b.h / 2) * imgNatSize.h,
              width: b.w * imgNatSize.w,
              height: b.h * imgNatSize.h,
            }),
          )
          setBoxes(loaded)
        }
      })
      .catch(() => setBoxes([]))
  }, [taskId, currentImage, imgNatSize])

  const handleAccept = useCallback(() => {
    if (!currentImage || !taskId || !imgNatSize) return
    // Save accepted boxes as seed annotations
    const yoloBoxes = boxes.map((b) => ({
      class_index: b.classIndex,
      cx: (b.x + b.width / 2) / imgNatSize.w,
      cy: (b.y + b.height / 2) / imgNatSize.h,
      w: b.width / imgNatSize.w,
      h: b.height / imgNatSize.h,
    }))
    filesApi.saveSeedAnnotation(taskId, currentImage.name, yoloBoxes)
    setDecisions((prev) => ({ ...prev, [currentImage.stem]: 'accept' }))
    // Move to next
    if (currentIndex < images.length - 1) {
      setCurrentIndex(currentIndex + 1)
    }
  }, [currentImage, taskId, imgNatSize, boxes, currentIndex, images.length])

  const handleReject = useCallback(() => {
    if (!currentImage) return
    setDecisions((prev) => ({ ...prev, [currentImage.stem]: 'reject' }))
    if (currentIndex < images.length - 1) {
      setCurrentIndex(currentIndex + 1)
    }
  }, [currentImage, currentIndex, images.length])

  const handleFinishReview = useCallback(() => {
    const accepted = Object.values(decisions).filter((d) => d === 'accept').length
    const currentCount = useTaskStore.getState().labeledImageCount || 0
    setLabeledImageCount(currentCount + accepted)
    setStage('train_config')
  }, [decisions, setLabeledImageCount, setStage])

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'a' || e.key === 'A') handleAccept()
      if (e.key === 'r' || e.key === 'R') handleReject()
      if (e.key === 'ArrowRight' && currentIndex < images.length - 1) setCurrentIndex(currentIndex + 1)
      if (e.key === 'ArrowLeft' && currentIndex > 0) setCurrentIndex(currentIndex - 1)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [handleAccept, handleReject, currentIndex, images.length])

  if (loading) {
    return (
      <div style={{ padding: 40, textAlign: 'center', color: 'var(--gray-400)' }}>
        Loading review images...
      </div>
    )
  }

  if (images.length === 0) {
    return (
      <div style={{ padding: 40, textAlign: 'center' }}>
        <p style={{ fontSize: 14, color: 'var(--gray-500)', marginBottom: 16 }}>
          No low-confidence predictions to review.
        </p>
        <button className="btn btn-primary" onClick={() => setStage('train_config')}>
          Continue to Training
        </button>
      </div>
    )
  }

  const reviewedCount = Object.keys(decisions).length
  const acceptedCount = Object.values(decisions).filter((d) => d === 'accept').length
  const rejectedCount = Object.values(decisions).filter((d) => d === 'reject').length

  return (
    <div style={{ padding: '24px 32px' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div className="flex items-center gap-3 mb-2">
          <div className="badge" style={{ background: '#f59e0b', color: '#fff', fontSize: 11, fontWeight: 600 }}>Review</div>
          <h1 className="page-title" style={{ marginBottom: 0 }}>低置信框审核</h1>
        </div>
        <p className="page-subtitle">
          以下框是模型不太确定的检测结果，请逐张审核：接受 (A)、拒绝 (R)，或编辑后接受
        </p>
      </div>

      {/* Progress Bar */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--gray-500)', marginBottom: 4 }}>
          <span>{currentIndex + 1} / {images.length}</span>
          <span>
            Reviewed: {reviewedCount} | Accepted: {acceptedCount} | Rejected: {rejectedCount}
          </span>
        </div>
        <div style={{ height: 4, background: 'var(--gray-100)', borderRadius: 2 }}>
          <div
            style={{
              height: '100%',
              width: `${(reviewedCount / images.length) * 100}%`,
              background: 'var(--develop-blue)',
              borderRadius: 2,
              transition: 'width 0.2s',
            }}
          />
        </div>
      </div>

      {/* Main Content */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 240px', gap: 16 }}>
        {/* Canvas */}
        <div style={{ position: 'relative', background: '#111', borderRadius: 10, overflow: 'hidden', minHeight: 400 }}>
          {imageUrl && (
            <>
              <MultiClassAnnotationCanvas
                imageUrl={imageUrl}
                boxes={boxes}
                classes={classes}
                activeClassIndex={selectedClassIdx}
                onBoxesChange={setBoxes}
              />
              <img
                src={imageUrl}
                alt=""
                style={{ display: 'none' }}
                onLoad={(e) => setImgNatSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
                crossOrigin="anonymous"
              />
            </>
          )}
          {/* Decision overlay */}
          {currentImage && decisions[currentImage.stem] && (
            <div
              style={{
                position: 'absolute',
                top: 8,
                right: 8,
                padding: '4px 10px',
                borderRadius: 6,
                fontSize: 11,
                fontWeight: 700,
                background: decisions[currentImage.stem] === 'accept' ? '#dcfce7' : '#fef2f2',
                color: decisions[currentImage.stem] === 'accept' ? '#166534' : '#991b1b',
              }}
            >
              {decisions[currentImage.stem] === 'accept' ? 'ACCEPTED' : 'REJECTED'}
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {/* Class selector */}
          <div style={{ background: '#fff', borderRadius: 8, border: '1px solid var(--gray-100)', padding: 12 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--gray-500)', marginBottom: 8 }}>CLASS</div>
            {classes.map((cls, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedClassIdx(idx)}
                style={{
                  display: 'block',
                  width: '100%',
                  textAlign: 'left',
                  padding: '6px 8px',
                  borderRadius: 4,
                  border: 'none',
                  background: selectedClassIdx === idx ? `${cls.color}22` : 'transparent',
                  cursor: 'pointer',
                  fontSize: 12,
                  fontWeight: selectedClassIdx === idx ? 600 : 400,
                  color: selectedClassIdx === idx ? cls.color : 'var(--gray-600)',
                  marginBottom: 2,
                }}
              >
                <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: cls.color, marginRight: 6 }} />
                {cls.name}
              </button>
            ))}
          </div>

          {/* Action buttons */}
          <button
            className="btn btn-primary"
            onClick={handleAccept}
            style={{ width: '100%', padding: '10px 16px', background: '#16a34a', borderColor: '#16a34a' }}
          >
            ✓ Accept (A)
          </button>
          <button
            className="btn btn-secondary"
            onClick={handleReject}
            style={{ width: '100%', padding: '10px 16px', color: '#dc2626', borderColor: '#dc2626' }}
          >
            ✗ Reject (R)
          </button>

          {/* Navigation */}
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              className="btn btn-secondary"
              disabled={currentIndex === 0}
              onClick={() => setCurrentIndex(currentIndex - 1)}
              style={{ flex: 1, padding: '8px 0' }}
            >
              ← Prev
            </button>
            <button
              className="btn btn-secondary"
              disabled={currentIndex >= images.length - 1}
              onClick={() => setCurrentIndex(currentIndex + 1)}
              style={{ flex: 1, padding: '8px 0' }}
            >
              Next →
            </button>
          </div>

          {/* Finish */}
          <button
            className="btn btn-primary"
            onClick={handleFinishReview}
            disabled={reviewedCount < images.length * 0.5}
            style={{ width: '100%', padding: '10px 16px', marginTop: 'auto' }}
          >
            Complete Review ({reviewedCount}/{images.length})
          </button>
          <p style={{ fontSize: 10, color: 'var(--gray-400)', textAlign: 'center', margin: 0 }}>
            Review at least 50% to continue
          </p>
        </div>
      </div>
    </div>
  )
}

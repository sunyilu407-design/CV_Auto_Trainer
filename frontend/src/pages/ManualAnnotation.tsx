import { useState, useEffect, useCallback, useRef } from 'react'
import { useTaskStore } from '../store/taskStore'
import { filesApi } from '../api/backend'
import MultiClassAnnotationCanvas, {
  MCBox,
  ClassDef,
  classColor,
} from '../components/MultiClassAnnotationCanvas'

const MIN_SEED_IMAGES = 20

interface ImageItem {
  name: string
  hasAnnotation: boolean
}

type FilterMode = 'all' | 'annotated' | 'unannotated'

export default function ManualAnnotation() {
  const {
    taskId,
    vlmResult,
    setStage,
    setSnowballMode,
    setSeedAnnotatedCount,
  } = useTaskStore()

  const [imageList, setImageList] = useState<ImageItem[]>([])
  const [totalImages, setTotalImages] = useState(0)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [activeClassIndex, setActiveClassIndex] = useState(0)
  const [boxes, setBoxes] = useState<MCBox[]>([])
  const [selectedBoxIndex, setSelectedBoxIndex] = useState<number | null>(null)
  const [filter, setFilter] = useState<FilterMode>('all')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [annotatedCount, setAnnotatedCount] = useState(0)

  // Cache of annotations per image stem
  const annotationsCache = useRef<Record<string, MCBox[]>>({})

  const classes: ClassDef[] = (vlmResult?.classes || []).map((c, i) => ({
    name: c.display_name_zh || c.class_name,
    color: classColor(i),
  }))

  // Load image list and existing annotations
  useEffect(() => {
    if (!taskId) return
    ;(async () => {
      setLoading(true)
      try {
        const [imgResult, annResult] = await Promise.all([
          filesApi.listDatasetImages(taskId, 1, 10000),
          filesApi.getSeedAnnotations(taskId),
        ])

        const annotatedStems = new Set(Object.keys(annResult.annotations))
        const items: ImageItem[] = imgResult.images.map((img) => ({
          name: img.name,
          hasAnnotation: annotatedStems.has(img.name.replace(/\.[^.]+$/, '')),
        }))
        setImageList(items)
        setTotalImages(imgResult.total)
        setAnnotatedCount(annResult.annotated_count)
        setSeedAnnotatedCount(annResult.annotated_count)

        // Pre-populate cache from server
        for (const [stem, serverBoxes] of Object.entries(annResult.annotations)) {
          annotationsCache.current[stem] = serverBoxes.map((b) => ({
            classIndex: b.class_index,
            x: 0, y: 0, width: 0, height: 0,
            _cx: b.cx, _cy: b.cy, _w: b.w, _h: b.h,
          } as MCBox & { _cx: number; _cy: number; _w: number; _h: number }))
        }
      } catch (e) {
        console.error('Failed to load dataset images', e)
      } finally {
        setLoading(false)
      }
    })()
  }, [taskId])

  // Filtered list
  const filteredList = imageList.filter((img) => {
    if (filter === 'annotated') return img.hasAnnotation
    if (filter === 'unannotated') return !img.hasAnnotation
    return true
  })

  const currentImage = filteredList[currentIndex] || null
  const currentStem = currentImage ? currentImage.name.replace(/\.[^.]+$/, '') : ''

  // Build image URL
  const imageUrl = currentImage && taskId
    ? `/api/files/${taskId}/image/${encodeURIComponent(currentImage.name)}`
    : ''

  // Load boxes for current image (convert YOLO normalized → pixel when image loads)
  const [imgNatSize, setImgNatSize] = useState<{ w: number; h: number } | null>(null)

  // When current image changes, load its cached annotation
  useEffect(() => {
    if (!currentStem) {
      setBoxes([])
      return
    }
    const cached = annotationsCache.current[currentStem]
    if (cached && cached.length > 0 && (cached[0] as any)._cx !== undefined) {
      // Still in YOLO format, need image size to convert — defer to onImageLoad
      setBoxes([])
    } else if (cached) {
      setBoxes(cached)
    } else {
      setBoxes([])
    }
    setSelectedBoxIndex(null)
  }, [currentIndex, filter])

  // Convert YOLO-format cached boxes to pixel coords when image loads
  const handleImageLoad = useCallback(
    (img: HTMLImageElement) => {
      const natW = img.naturalWidth
      const natH = img.naturalHeight
      setImgNatSize({ w: natW, h: natH })

      if (!currentStem) return
      const cached = annotationsCache.current[currentStem]
      if (!cached || cached.length === 0) return

      if ((cached[0] as any)._cx !== undefined) {
        // Convert from YOLO normalized to pixel
        const converted = cached.map((b) => {
          const raw = b as any
          const cx = raw._cx * natW
          const cy = raw._cy * natH
          const bw = raw._w * natW
          const bh = raw._h * natH
          return {
            classIndex: b.classIndex,
            x: cx - bw / 2,
            y: cy - bh / 2,
            width: bw,
            height: bh,
          }
        })
        annotationsCache.current[currentStem] = converted
        setBoxes(converted)
      }
    },
    [currentStem],
  )

  // Save annotation to backend
  const saveCurrentAnnotation = useCallback(
    async (newBoxes: MCBox[]) => {
      if (!taskId || !currentImage || !imgNatSize) return
      annotationsCache.current[currentStem] = newBoxes

      // Convert pixel → YOLO normalized
      const yoloBoxes = newBoxes.map((b) => ({
        class_index: b.classIndex,
        cx: (b.x + b.width / 2) / imgNatSize.w,
        cy: (b.y + b.height / 2) / imgNatSize.h,
        w: b.width / imgNatSize.w,
        h: b.height / imgNatSize.h,
      }))

      setSaving(true)
      try {
        await filesApi.saveSeedAnnotation(taskId, currentImage.name, yoloBoxes)
        // Update annotation status in list
        const hadAnnotation = currentImage.hasAnnotation
        const hasNow = newBoxes.length > 0
        if (hadAnnotation !== hasNow) {
          setImageList((prev) =>
            prev.map((img) =>
              img.name === currentImage.name ? { ...img, hasAnnotation: hasNow } : img,
            ),
          )
          setAnnotatedCount((prev) => prev + (hasNow ? 1 : -1))
          setSeedAnnotatedCount(annotatedCount + (hasNow ? 1 : -1))
        }
      } catch (e) {
        console.error('Failed to save annotation', e)
      } finally {
        setSaving(false)
      }
    },
    [taskId, currentImage, currentStem, imgNatSize, annotatedCount],
  )

  const handleBoxesChange = useCallback(
    (newBoxes: MCBox[]) => {
      setBoxes(newBoxes)
      saveCurrentAnnotation(newBoxes)
    },
    [saveCurrentAnnotation],
  )

  // Navigation
  const goTo = (idx: number) => {
    if (idx >= 0 && idx < filteredList.length) setCurrentIndex(idx)
  }
  const goPrev = () => goTo(currentIndex - 1)
  const goNext = () => goTo(currentIndex + 1)
  const goSkip = () => goTo(currentIndex + 1)

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft') goPrev()
      else if (e.key === 'd' || e.key === 'D' || e.key === 'ArrowRight') goNext()
      else if (e.key === 's' || e.key === 'S') goSkip()
      else if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedBoxIndex != null) {
          const next = boxes.filter((_, i) => i !== selectedBoxIndex)
          handleBoxesChange(next)
          setSelectedBoxIndex(null)
        }
      }
      // Number keys to switch class
      const num = parseInt(e.key)
      if (num >= 1 && num <= classes.length) {
        setActiveClassIndex(num - 1)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [currentIndex, filteredList.length, selectedBoxIndex, boxes, classes.length])

  const canStartTraining = annotatedCount >= MIN_SEED_IMAGES

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 60, color: 'var(--gray-500)' }}>
        Loading dataset images...
      </div>
    )
  }

  return (
    <div>
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 20,
        }}
      >
        <div>
          <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700 }}>Manual Annotation</h2>
          <p style={{ margin: '4px 0 0', fontSize: 13, color: 'var(--gray-500)' }}>
            Annotated{' '}
            <strong style={{ color: annotatedCount >= MIN_SEED_IMAGES ? '#10b981' : '#f59e0b' }}>
              {annotatedCount}
            </strong>{' '}
            / {totalImages} images (min {MIN_SEED_IMAGES})
            {saving && <span style={{ marginLeft: 8, color: 'var(--gray-400)' }}>Saving...</span>}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => setStage('algorithm_plan')}
          >
            Back
          </button>
          <button
            className="btn btn-primary btn-sm"
            disabled={!canStartTraining}
            onClick={() => {
              setSnowballMode(true)
              setStage('seed_training')
            }}
            title={
              canStartTraining
                ? 'Start seed training'
                : `Need at least ${MIN_SEED_IMAGES} annotations`
            }
          >
            Start Training ({annotatedCount}/{MIN_SEED_IMAGES})
          </button>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 16, minHeight: 500 }}>
        {/* Left: Image list */}
        <div
          style={{
            width: 200,
            flexShrink: 0,
            background: '#fff',
            borderRadius: 10,
            border: '1px solid var(--gray-100)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}
        >
          {/* Filter tabs */}
          <div
            style={{
              display: 'flex',
              borderBottom: '1px solid var(--gray-100)',
              fontSize: 11,
              fontWeight: 600,
            }}
          >
            {(['all', 'annotated', 'unannotated'] as FilterMode[]).map((f) => (
              <button
                key={f}
                onClick={() => { setFilter(f); setCurrentIndex(0) }}
                style={{
                  flex: 1,
                  padding: '8px 0',
                  border: 'none',
                  background: filter === f ? 'var(--gray-50)' : 'transparent',
                  color: filter === f ? 'var(--gray-900)' : 'var(--gray-400)',
                  cursor: 'pointer',
                  borderBottom: filter === f ? '2px solid var(--gray-900)' : '2px solid transparent',
                }}
              >
                {f === 'all' ? 'All' : f === 'annotated' ? 'Done' : 'Todo'}
              </button>
            ))}
          </div>

          {/* Image thumbnails */}
          <div style={{ flex: 1, overflow: 'auto', padding: 4 }}>
            {filteredList.map((img, idx) => (
              <div
                key={img.name}
                onClick={() => setCurrentIndex(idx)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '6px 8px',
                  borderRadius: 6,
                  cursor: 'pointer',
                  background: idx === currentIndex ? 'var(--gray-100)' : 'transparent',
                  fontSize: 12,
                  color: 'var(--gray-700)',
                  transition: 'background 0.1s',
                }}
              >
                <span
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: img.hasAnnotation ? '#10b981' : 'var(--gray-200)',
                    flexShrink: 0,
                  }}
                />
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {img.name}
                </span>
              </div>
            ))}
            {filteredList.length === 0 && (
              <div style={{ padding: 16, textAlign: 'center', color: 'var(--gray-400)', fontSize: 12 }}>
                No images
              </div>
            )}
          </div>
        </div>

        {/* Right: Canvas + controls */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>
          {/* Class selector */}
          {classes.length > 1 && (
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {classes.map((cls, i) => (
                <button
                  key={i}
                  onClick={() => setActiveClassIndex(i)}
                  style={{
                    padding: '4px 12px',
                    borderRadius: 6,
                    border: `2px solid ${cls.color}`,
                    background: activeClassIndex === i ? cls.color : 'transparent',
                    color: activeClassIndex === i ? '#fff' : cls.color,
                    fontSize: 12,
                    fontWeight: 600,
                    cursor: 'pointer',
                    transition: 'all 0.15s',
                  }}
                >
                  [{i + 1}] {cls.name}
                </button>
              ))}
            </div>
          )}

          {/* Canvas */}
          {currentImage ? (
            <div style={{ flex: 1, position: 'relative' }}>
              <MultiClassAnnotationCanvas
                imageUrl={imageUrl}
                boxes={boxes}
                onBoxesChange={handleBoxesChange}
                classes={classes}
                activeClassIndex={activeClassIndex}
                selectedBoxIndex={selectedBoxIndex}
                onSelectBox={setSelectedBoxIndex}
              />
              {/* Hidden img to get natural size */}
              <img
                src={imageUrl}
                alt=""
                style={{ display: 'none' }}
                onLoad={(e) => handleImageLoad(e.currentTarget)}
                crossOrigin="anonymous"
              />
            </div>
          ) : (
            <div
              style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'var(--gray-50)',
                borderRadius: 10,
                color: 'var(--gray-400)',
                fontSize: 14,
              }}
            >
              Select an image from the list
            </div>
          )}

          {/* Navigation */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '8px 0',
            }}
          >
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                className="btn btn-secondary btn-sm"
                onClick={goPrev}
                disabled={currentIndex <= 0}
              >
                A Prev
              </button>
              <button
                className="btn btn-secondary btn-sm"
                onClick={goSkip}
                disabled={currentIndex >= filteredList.length - 1}
              >
                S Skip
              </button>
              <button
                className="btn btn-secondary btn-sm"
                onClick={goNext}
                disabled={currentIndex >= filteredList.length - 1}
              >
                D Next
              </button>
            </div>
            <span style={{ fontSize: 12, color: 'var(--gray-500)' }}>
              {filteredList.length > 0
                ? `${currentIndex + 1} / ${filteredList.length}`
                : '0 / 0'}{' '}
              | Boxes: {boxes.length}
            </span>
          </div>

          {/* Hints */}
          <div
            style={{
              padding: '8px 12px',
              background: 'var(--gray-50)',
              borderRadius: 8,
              fontSize: 11,
              color: 'var(--gray-500)',
              lineHeight: 1.6,
            }}
          >
            <strong>Shortcuts:</strong> A/D or Arrow keys = navigate | S = skip |
            Number keys = switch class | Right-click box to select | Delete = remove selected |
            Draw with left mouse drag
          </div>
        </div>
      </div>
    </div>
  )
}

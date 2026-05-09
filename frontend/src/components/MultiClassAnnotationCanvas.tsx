import { useRef, useEffect, useState, useCallback } from 'react'

export interface MCBox {
  classIndex: number
  x: number
  y: number
  width: number
  height: number
  confidence?: number
}

export interface ClassDef {
  name: string
  color: string
}

const DEFAULT_COLORS = [
  '#f59e0b', '#0a72ef', '#10b981', '#ef4444', '#8b5cf6',
  '#ec4899', '#06b6d4', '#f97316', '#14b8a6', '#6366f1',
]

export function classColor(index: number): string {
  return DEFAULT_COLORS[index % DEFAULT_COLORS.length]
}

interface Props {
  imageUrl: string
  boxes: MCBox[]
  onBoxesChange: (boxes: MCBox[]) => void
  classes: ClassDef[]
  activeClassIndex: number
  readOnly?: boolean
  selectedBoxIndex?: number | null
  onSelectBox?: (index: number | null) => void
}

export default function MultiClassAnnotationCanvas({
  imageUrl,
  boxes,
  onBoxesChange,
  classes,
  activeClassIndex,
  readOnly = false,
  selectedBoxIndex = null,
  onSelectBox,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const [drawing, setDrawing] = useState(false)
  const [startPos, setStartPos] = useState<{ x: number; y: number } | null>(null)

  const boxesRef = useRef(boxes)
  boxesRef.current = boxes

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      imageRef.current = img
      canvas.width = img.width
      canvas.height = img.height
      redraw(ctx, boxesRef.current, selectedBoxIndex)
    }
    img.src = imageUrl
  }, [imageUrl])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx || !imageRef.current) return
    redraw(ctx, boxes, selectedBoxIndex)
  }, [boxes, selectedBoxIndex])

  const redraw = useCallback(
    (ctx: CanvasRenderingContext2D, currentBoxes: MCBox[], selIdx: number | null) => {
      const img = imageRef.current
      if (!img) return

      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
      ctx.drawImage(img, 0, 0)

      currentBoxes.forEach((box, i) => {
        const color = classes[box.classIndex]?.color || classColor(box.classIndex)
        const isSelected = selIdx === i

        ctx.strokeStyle = color
        ctx.lineWidth = isSelected ? 3 : 2
        ctx.strokeRect(box.x, box.y, box.width, box.height)

        const alpha = isSelected ? 0.2 : 0.1
        ctx.fillStyle = color + Math.round(alpha * 255).toString(16).padStart(2, '0')
        ctx.fillRect(box.x, box.y, box.width, box.height)

        // Label badge
        const label = classes[box.classIndex]?.name || `cls_${box.classIndex}`
        const confStr = box.confidence != null ? ` ${(box.confidence * 100).toFixed(0)}%` : ''
        const text = label + confStr
        ctx.font = 'bold 11px Inter, sans-serif'
        const tw = ctx.measureText(text).width
        const badgeH = 18
        const badgeW = tw + 10
        const bx = box.x
        const by = box.y - badgeH > 0 ? box.y - badgeH : box.y

        ctx.fillStyle = color
        ctx.fillRect(bx, by, badgeW, badgeH)
        ctx.fillStyle = '#fff'
        ctx.textAlign = 'left'
        ctx.textBaseline = 'middle'
        ctx.fillText(text, bx + 5, by + badgeH / 2)

        // Selected indicator
        if (isSelected) {
          ctx.setLineDash([4, 3])
          ctx.strokeStyle = '#fff'
          ctx.lineWidth = 1
          ctx.strokeRect(box.x - 1, box.y - 1, box.width + 2, box.height + 2)
          ctx.setLineDash([])
        }
      })
    },
    [classes],
  )

  const getCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    }
  }

  const hitTest = (px: number, py: number): number | null => {
    for (let i = boxes.length - 1; i >= 0; i--) {
      const b = boxes[i]
      if (px >= b.x && px <= b.x + b.width && py >= b.y && py <= b.y + b.height) {
        return i
      }
    }
    return null
  }

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (readOnly) return
    // Right-click: select for deletion
    if (e.button === 2) {
      e.preventDefault()
      const pos = getCanvasCoords(e)
      const idx = hitTest(pos.x, pos.y)
      onSelectBox?.(idx)
      return
    }
    setDrawing(true)
    setStartPos(getCanvasCoords(e))
    onSelectBox?.(null)
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drawing || !startPos) return
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    const current = getCanvasCoords(e)

    redraw(ctx, boxes, selectedBoxIndex)
    const color = classes[activeClassIndex]?.color || classColor(activeClassIndex)
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.setLineDash([4, 3])
    ctx.strokeRect(startPos.x, startPos.y, current.x - startPos.x, current.y - startPos.y)
    ctx.setLineDash([])
    ctx.fillStyle = color + '1a'
    ctx.fillRect(startPos.x, startPos.y, current.x - startPos.x, current.y - startPos.y)
  }

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drawing || !startPos) return
    setDrawing(false)
    const current = getCanvasCoords(e)
    const newBox: MCBox = {
      classIndex: activeClassIndex,
      x: Math.min(startPos.x, current.x),
      y: Math.min(startPos.y, current.y),
      width: Math.abs(current.x - startPos.x),
      height: Math.abs(current.y - startPos.y),
    }

    if (newBox.width > 5 && newBox.height > 5) {
      onBoxesChange([...boxes, newBox])
    }
    setStartPos(null)
  }

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault()
  }

  return (
    <div style={{ position: 'relative' }}>
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => setDrawing(false)}
        onContextMenu={handleContextMenu}
        style={{
          maxWidth: '100%',
          borderRadius: 8,
          border: '1px solid var(--gray-200)',
          cursor: readOnly ? 'default' : 'crosshair',
          display: 'block',
        }}
      />
      {/* Toolbar overlay */}
      {!readOnly && (
        <div
          style={{
            position: 'absolute',
            top: 8,
            right: 8,
            display: 'flex',
            gap: 6,
          }}
        >
          {selectedBoxIndex != null && (
            <button
              onClick={() => {
                const next = boxes.filter((_, i) => i !== selectedBoxIndex)
                onBoxesChange(next)
                onSelectBox?.(null)
              }}
              style={{
                padding: '4px 10px',
                fontSize: 12,
                fontWeight: 500,
                background: 'rgba(239,68,68,0.85)',
                backdropFilter: 'blur(8px)',
                color: '#fff',
                border: 'none',
                borderRadius: 6,
                cursor: 'pointer',
              }}
            >
              Delete
            </button>
          )}
          {boxes.length > 0 && (
            <button
              onClick={() => {
                onBoxesChange([])
                onSelectBox?.(null)
              }}
              style={{
                padding: '4px 10px',
                fontSize: 12,
                fontWeight: 500,
                background: 'rgba(23,23,23,0.75)',
                backdropFilter: 'blur(8px)',
                color: '#fff',
                border: 'none',
                borderRadius: 6,
                cursor: 'pointer',
              }}
            >
              Clear All
            </button>
          )}
        </div>
      )}
    </div>
  )
}

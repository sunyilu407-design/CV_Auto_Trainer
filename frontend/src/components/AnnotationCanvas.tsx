import { useRef, useEffect, useState, useCallback } from 'react'

interface Box {
  x: number
  y: number
  width: number
  height: number
}

interface Props {
  imageUrl: string
  onBoxesChange?: (boxes: Box[]) => void
  initialBoxes?: Box[]
  readOnly?: boolean
}

export default function AnnotationCanvas({ imageUrl, onBoxesChange, initialBoxes = [], readOnly = false }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const [boxes, setBoxes] = useState<Box[]>(initialBoxes)
  const [drawing, setDrawing] = useState(false)
  const [startPos, setStartPos] = useState<{ x: number; y: number } | null>(null)

  useEffect(() => {
    setBoxes(initialBoxes)
  }, [initialBoxes])

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
      redraw(ctx, boxes)
    }
    img.src = imageUrl
  }, [imageUrl])

  const redraw = useCallback((ctx: CanvasRenderingContext2D, currentBoxes: Box[]) => {
    const img = imageRef.current
    if (!img) return

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    ctx.drawImage(img, 0, 0)

    currentBoxes.forEach((box, i) => {
      const isFirst = i === 0
      ctx.strokeStyle = isFirst ? '#f59e0b' : 'var(--develop-blue)'
      ctx.lineWidth = 2
      ctx.strokeRect(box.x, box.y, box.width, box.height)

      ctx.fillStyle = isFirst ? 'rgba(245,158,11,0.12)' : 'rgba(10,114,239,0.12)'
      ctx.fillRect(box.x, box.y, box.width, box.height)

      // Class number badge
      ctx.fillStyle = isFirst ? '#f59e0b' : 'var(--develop-blue)'
      ctx.beginPath()
      ctx.arc(box.x + 10, box.y + 10, 8, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = '#fff'
      ctx.font = 'bold 9px Inter, sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(String(i + 1), box.x + 10, box.y + 10)
    })
  }, [])

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

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (readOnly) return
    setDrawing(true)
    setStartPos(getCanvasCoords(e))
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drawing || !startPos) return
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    const current = getCanvasCoords(e)

    redraw(ctx, boxes)
    ctx.strokeStyle = '#f59e0b'
    ctx.lineWidth = 2
    ctx.setLineDash([4, 3])
    ctx.strokeRect(
      startPos.x,
      startPos.y,
      current.x - startPos.x,
      current.y - startPos.y
    )
    ctx.setLineDash([])

    ctx.fillStyle = 'rgba(245,158,11,0.12)'
    ctx.fillRect(startPos.x, startPos.y, current.x - startPos.x, current.y - startPos.y)
  }

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drawing || !startPos) return
    setDrawing(false)
    const current = getCanvasCoords(e)
    const newBox: Box = {
      x: Math.min(startPos.x, current.x),
      y: Math.min(startPos.y, current.y),
      width: Math.abs(current.x - startPos.x),
      height: Math.abs(current.y - startPos.y),
    }

    if (newBox.width > 5 && newBox.height > 5) {
      const newBoxes = [...boxes, newBox]
      setBoxes(newBoxes)
      onBoxesChange?.(newBoxes)
    }
    setStartPos(null)
  }

  return (
    <div style={{ position: 'relative' }}>
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => setDrawing(false)}
        style={{
          maxWidth: '100%',
          borderRadius: 8,
          border: '1px solid var(--gray-200)',
          cursor: readOnly ? 'default' : 'crosshair',
          display: 'block',
        }}
      />
      {boxes.length > 0 && !readOnly && (
        <button
          onClick={() => {
            setBoxes([])
            onBoxesChange?.([])
          }}
          style={{
            position: 'absolute',
            top: 10,
            right: 10,
            padding: '5px 10px',
            fontSize: 12,
            fontWeight: 500,
            background: 'rgba(23,23,23,0.75)',
            backdropFilter: 'blur(8px)',
            color: '#fff',
            border: 'none',
            borderRadius: 6,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            transition: 'background 0.15s ease',
          }}
          onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(23,23,23,0.9)' }}
          onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(23,23,23,0.75)' }}
        >
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18"/>
            <line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
          清除所有框
        </button>
      )}
    </div>
  )
}

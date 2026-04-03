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
      ctx.strokeStyle = i === 0 ? '#ff9800' : '#1976d2'
      ctx.lineWidth = 2
      ctx.strokeRect(box.x, box.y, box.width, box.height)

      ctx.fillStyle = i === 0 ? 'rgba(255,152,0,0.15)' : 'rgba(25,118,210,0.15)'
      ctx.fillRect(box.x, box.y, box.width, box.height)
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
    ctx.strokeStyle = '#ff9800'
    ctx.lineWidth = 2
    ctx.strokeRect(
      startPos.x,
      startPos.y,
      current.x - startPos.x,
      current.y - startPos.y
    )
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
          border: '1px solid #ddd',
          borderRadius: '4px',
          cursor: readOnly ? 'default' : 'crosshair',
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
            top: '8px',
            right: '8px',
            padding: '4px 8px',
            fontSize: '12px',
            background: 'rgba(0,0,0,0.6)',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          清除所有框
        </button>
      )}
    </div>
  )
}

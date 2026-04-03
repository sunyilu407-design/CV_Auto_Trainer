import { useEffect, useRef } from 'react'

interface Props {
  currentEpoch: number
  currentMap: number
}

export default function MetricsChart({ currentEpoch, currentMap }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const dataRef = useRef<{ epoch: number; map: number }[]>([])

  useEffect(() => {
    if (currentEpoch > 0) {
      dataRef.current.push({ epoch: currentEpoch, map: currentMap })
      if (dataRef.current.length > 100) {
        dataRef.current.shift()
      }
    }
  }, [currentEpoch, currentMap])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const data = dataRef.current
    canvas.width = canvas.offsetWidth * 2
    canvas.height = canvas.offsetHeight * 2
    ctx.scale(2, 2)

    const w = canvas.offsetWidth
    const h = canvas.offsetHeight
    const padding = 30

    ctx.clearRect(0, 0, w, h)

    // Grid
    ctx.strokeStyle = '#f0f0f0'
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = padding + ((h - padding * 2) * i) / 4
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(w - padding, y)
      ctx.stroke()
      ctx.fillStyle = '#999'
      ctx.font = '10px sans-serif'
      ctx.fillText(`${((4 - i) / 4 * 100).toFixed(0)}%`, 5, y + 3)
    }

    if (data.length < 2) return

    const maxEpoch = Math.max(...data.map((d) => d.epoch), 1)
    const xScale = (w - padding * 2) / maxEpoch

    // Draw line
    ctx.strokeStyle = '#1976d2'
    ctx.lineWidth = 2
    ctx.beginPath()
    data.forEach((d, i) => {
      const x = padding + d.epoch * xScale
      const y = padding + (h - padding * 2) * (1 - d.map)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // X axis label
    ctx.fillStyle = '#999'
    ctx.font = '10px sans-serif'
    ctx.fillText('Epoch', w / 2 - 15, h - 5)
  }, [currentEpoch, currentMap])

  return (
    <div
      style={{
        background: '#fff',
        borderRadius: '8px',
        padding: '16px',
        height: '200px',
      }}
    >
      <h3 style={{ fontSize: '14px', marginBottom: '12px' }}>mAP50 曲线</h3>
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: 'calc(100% - 30px)' }}
      />
    </div>
  )
}

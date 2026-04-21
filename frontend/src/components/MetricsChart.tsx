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
    const padding = 36

    ctx.clearRect(0, 0, w, h)

    // Grid lines
    ctx.strokeStyle = 'var(--gray-100)'
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = padding + ((h - padding * 2) * i) / 4
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(w - padding, y)
      ctx.stroke()
    }

    // Y-axis labels
    ctx.fillStyle = 'var(--gray-400)'
    ctx.font = '11px Inter, sans-serif'
    ctx.textAlign = 'left'
    for (let i = 0; i <= 4; i++) {
      const y = padding + ((h - padding * 2) * i) / 4
      ctx.fillText(`${((4 - i) / 4 * 100).toFixed(0)}%`, 4, y + 4)
    }

    if (data.length < 2) return

    const maxEpoch = Math.max(...data.map((d) => d.epoch), 1)
    const xScale = (w - padding * 2) / maxEpoch

    // Gradient fill under line
    const gradient = ctx.createLinearGradient(0, padding, 0, h - padding)
    gradient.addColorStop(0, 'rgba(10, 114, 239, 0.15)')
    gradient.addColorStop(1, 'rgba(10, 114, 239, 0)')

    ctx.beginPath()
    data.forEach((d, i) => {
      const x = padding + d.epoch * xScale
      const y = padding + (h - padding * 2) * (1 - d.map)
      if (i === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })
    const lastX = padding + data[data.length - 1].epoch * xScale
    ctx.lineTo(lastX, h - padding)
    ctx.lineTo(padding, h - padding)
    ctx.closePath()
    ctx.fillStyle = gradient
    ctx.fill()

    // Draw line
    ctx.strokeStyle = 'var(--develop-blue)'
    ctx.lineWidth = 2
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.beginPath()
    data.forEach((d, i) => {
      const x = padding + d.epoch * xScale
      const y = padding + (h - padding * 2) * (1 - d.map)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // Latest point dot
    const latest = data[data.length - 1]
    const lx = padding + latest.epoch * xScale
    const ly = padding + (h - padding * 2) * (1 - latest.map)
    ctx.beginPath()
    ctx.arc(lx, ly, 3, 0, Math.PI * 2)
    ctx.fillStyle = 'var(--develop-blue)'
    ctx.fill()
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 1.5
    ctx.stroke()

    // X axis label
    ctx.fillStyle = 'var(--gray-400)'
    ctx.font = '11px Inter, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('Epoch', w / 2, h - 6)
  }, [currentEpoch, currentMap])

  return (
    <div
      className="card-section"
      style={{ padding: '16px 20px' }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h3 style={{ fontSize: 14, fontWeight: 600, margin: 0, letterSpacing: '-0.2px' }}>mAP50 曲线</h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--develop-blue)' }} />
          <span style={{ fontSize: 12, color: 'var(--gray-400)', fontVariantNumeric: 'tabular-nums' }}>
            {currentMap > 0 ? `${(currentMap * 100).toFixed(1)}%` : '—'}
          </span>
        </div>
      </div>
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '160px' }}
      />
    </div>
  )
}

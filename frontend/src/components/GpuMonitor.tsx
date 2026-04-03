import { useEffect, useState } from 'react'

interface GpuInfo {
  available: boolean
  name?: string
  totalMemoryGB?: number
  usedMemoryGB?: number
  freeMemoryGB?: number
}

export default function GpuMonitor() {
  const [gpu, setGpu] = useState<GpuInfo | null>(null)

  useEffect(() => {
    const fetchGpu = async () => {
      try {
        const res = await fetch('http://localhost:7860/gpu-info')
        const data = await res.json()
        setGpu(data)
      } catch {
        setGpu({ available: false })
      }
    }

    fetchGpu()
    const interval = setInterval(fetchGpu, 5000)
    return () => clearInterval(interval)
  }, [])

  if (!gpu || !gpu.available) {
    return (
      <div style={{ textAlign: 'center', color: '#999', padding: '16px' }}>
        未检测到 GPU 或 Worker 未启动
      </div>
    )
  }

  const usedPercent = gpu.totalMemoryGB
    ? Math.round((gpu.usedMemoryGB! / gpu.totalMemoryGB!) * 100)
    : 0

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
        <span style={{ fontSize: '14px', fontWeight: 600 }}>{gpu.name}</span>
        <span style={{ fontSize: '14px', color: '#666' }}>
          {gpu.freeMemoryGB?.toFixed(1)} GB 可用 / {gpu.totalMemoryGB?.toFixed(1)} GB 总计
        </span>
      </div>
      <div
        style={{
          width: '100%',
          height: '16px',
          background: '#e0e0e0',
          borderRadius: '8px',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${usedPercent}%`,
            height: '100%',
            background: usedPercent > 90 ? '#d32f2f' : usedPercent > 70 ? '#ff9800' : '#4caf50',
            transition: 'width 0.5s',
          }}
        />
      </div>
      <p style={{ fontSize: '12px', color: '#666', marginTop: '4px', textAlign: 'right' }}>
        已用 {gpu.usedMemoryGB?.toFixed(1)} GB ({usedPercent}%)
      </p>
    </div>
  )
}

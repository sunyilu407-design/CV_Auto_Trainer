import { useEffect, useState } from 'react'
import { WORKER_HTTP_BASE } from '../api/worker'

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
        const res = await fetch(`${WORKER_HTTP_BASE}/gpu-info`)
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
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 8,
          padding: '24px 20px',
          background: 'var(--gray-50)',
          borderRadius: 8,
          border: '1px solid var(--gray-100)',
        }}
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--gray-400)" strokeWidth="1.5">
          <rect x="2" y="3" width="20" height="14" rx="2"/>
          <line x1="8" y1="21" x2="16" y2="21"/>
          <line x1="12" y1="17" x2="12" y2="21"/>
        </svg>
        <span style={{ fontSize: 13, color: 'var(--gray-400)' }}>未检测到 GPU 或 Worker 未启动</span>
      </div>
    )
  }

  const usedPercent = gpu.totalMemoryGB
    ? Math.round((gpu.usedMemoryGB! / gpu.totalMemoryGB!) * 100)
    : 0

  const memColor = usedPercent > 90 ? 'var(--ship-red)' : usedPercent > 70 ? '#f59e0b' : 'var(--develop-blue)'

  return (
    <div
      style={{
        padding: '16px 20px',
        background: 'var(--gray-50)',
        borderRadius: 8,
        border: '1px solid var(--gray-100)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div
            style={{
              width: 28,
              height: 28,
              borderRadius: 6,
              background: `${memColor}15`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={memColor} strokeWidth="2">
              <rect x="2" y="3" width="20" height="14" rx="2"/>
              <line x1="8" y1="21" x2="16" y2="21"/>
              <line x1="12" y1="17" x2="12" y2="21"/>
            </svg>
          </div>
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-700)' }}>{gpu.name}</span>
        </div>
        <span style={{ fontSize: 12, color: 'var(--gray-400)' }}>
          {gpu.freeMemoryGB?.toFixed(1)} GB{' '}
          <span style={{ color: 'var(--gray-300)' }}>/</span>{' '}
          {gpu.totalMemoryGB?.toFixed(1)} GB
        </span>
      </div>

      {/* Progress bar */}
      <div
        style={{
          height: 6,
          borderRadius: 3,
          background: 'var(--gray-200)',
          overflow: 'hidden',
          marginBottom: 8,
        }}
      >
        <div
          style={{
            width: `${usedPercent}%`,
            height: '100%',
            background: memColor,
            borderRadius: 3,
            transition: 'width 0.5s ease',
          }}
        />
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 12, color: 'var(--gray-400)' }}>
          已用 {gpu.usedMemoryGB?.toFixed(1)} GB
        </span>
        <span
          style={{
            fontSize: 12,
            fontWeight: 600,
            color: memColor,
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          {usedPercent}%
        </span>
      </div>
    </div>
  )
}

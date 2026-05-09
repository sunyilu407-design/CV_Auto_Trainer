import { useEffect, useState } from 'react'
import { trainingApi, TrainingVersionEntry } from '../api/backend'
import { useTaskStore } from '../store/taskStore'

function mapColor(v: number): string {
  if (v >= 0.8) return '#10b981'
  if (v >= 0.6) return '#f59e0b'
  return '#ef4444'
}

function DeltaBadge({ current, previous }: { current: number; previous: number }) {
  const diff = current - previous
  if (Math.abs(diff) < 0.001) return null
  const pct = (diff * 100).toFixed(1)
  const up = diff > 0
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 2,
        padding: '1px 5px',
        borderRadius: 4,
        fontSize: 10,
        fontWeight: 600,
        background: up ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)',
        color: up ? '#059669' : '#dc2626',
        marginLeft: 4,
      }}
    >
      {up ? '↑' : '↓'}{up ? '+' : ''}{pct}
    </span>
  )
}

function VersionChart({ history }: { history: TrainingVersionEntry[] }) {
  const withMap = history.filter((e) => e.map50 != null)
  if (withMap.length < 1) return null

  const maxMap = Math.max(...withMap.map((e) => e.map50!), 0.5)
  const chartH = 120
  const barW = Math.min(48, Math.floor(400 / Math.max(withMap.length, 1)))

  return (
    <div style={{ padding: '12px 16px' }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--gray-500)', marginBottom: 8 }}>
        mAP50 VERSION COMPARISON
      </div>
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-end',
          gap: 6,
          height: chartH + 24,
          borderBottom: '1px solid var(--gray-100)',
          paddingBottom: 0,
        }}
      >
        {withMap.map((entry, idx) => {
          const pct = (entry.map50! / maxMap) * 100
          const prevMap = idx > 0 ? withMap[idx - 1].map50! : null
          return (
            <div
              key={entry.version}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 2,
                flex: `0 0 ${barW}px`,
              }}
            >
              {/* Value label */}
              <span style={{ fontSize: 10, fontWeight: 600, color: mapColor(entry.map50!) }}>
                {(entry.map50! * 100).toFixed(1)}%
              </span>
              {/* Delta */}
              {prevMap != null && (
                <DeltaBadge current={entry.map50!} previous={prevMap} />
              )}
              {/* Bar */}
              <div
                style={{
                  width: barW - 8,
                  height: `${Math.max(pct, 4)}%`,
                  maxHeight: chartH,
                  background: `linear-gradient(180deg, ${mapColor(entry.map50!)} 0%, ${mapColor(entry.map50!)}88 100%)`,
                  borderRadius: '4px 4px 0 0',
                  transition: 'height 0.4s ease',
                  position: 'relative',
                }}
              />
              {/* Version label */}
              <span style={{ fontSize: 10, color: 'var(--gray-500)', fontWeight: 500, marginTop: 2 }}>
                v{entry.version}
              </span>
            </div>
          )
        })}
      </div>
      {/* Summary line */}
      {withMap.length >= 2 && (() => {
        const first = withMap[0].map50!
        const last = withMap[withMap.length - 1].map50!
        const totalDiff = last - first
        const totalPct = (totalDiff * 100).toFixed(1)
        return (
          <div style={{ fontSize: 11, color: 'var(--gray-500)', marginTop: 8, textAlign: 'center' }}>
            v{withMap[0].version} → v{withMap[withMap.length - 1].version}:
            <span style={{ fontWeight: 600, color: totalDiff >= 0 ? '#059669' : '#dc2626', marginLeft: 4 }}>
              {totalDiff >= 0 ? '+' : ''}{totalPct}% mAP50
            </span>
            {' '}across {withMap.length} versions
          </div>
        )
      })()}
    </div>
  )
}

export default function TrainingHistory() {
  const { taskId } = useTaskStore()
  const [history, setHistory] = useState<TrainingVersionEntry[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!taskId) return
    setLoading(true)
    trainingApi
      .getTrainingHistory(taskId)
      .then(setHistory)
      .catch(() => setHistory([]))
      .finally(() => setLoading(false))
  }, [taskId])

  if (loading) {
    return (
      <div style={{ padding: 16, color: 'var(--gray-400)', fontSize: 13 }}>
        Loading history...
      </div>
    )
  }

  if (history.length === 0) {
    return (
      <div style={{ padding: 16, color: 'var(--gray-400)', fontSize: 13 }}>
        No training history yet.
      </div>
    )
  }

  return (
    <div
      style={{
        background: '#fff',
        borderRadius: 10,
        border: '1px solid var(--gray-100)',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          padding: '12px 16px',
          borderBottom: '1px solid var(--gray-100)',
          fontSize: 13,
          fontWeight: 600,
          color: 'var(--gray-700)',
        }}
      >
        Training History
      </div>

      {/* Version comparison chart */}
      <VersionChart history={history} />

      {/* Table */}
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ background: 'var(--gray-50)' }}>
            <th style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600 }}>Version</th>
            <th style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600 }}>Images</th>
            <th style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600 }}>mAP50</th>
            <th style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600 }}>Δ</th>
            <th style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600 }}>New</th>
            <th style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600 }}>Date</th>
            <th style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600 }}>Model</th>
          </tr>
        </thead>
        <tbody>
          {history.map((entry, idx) => {
            const prevMap = idx > 0 ? history[idx - 1].map50 : null
            return (
              <tr
                key={entry.version}
                style={{ borderBottom: '1px solid var(--gray-50)' }}
              >
                <td style={{ padding: '8px 12px', fontWeight: 700, color: 'var(--gray-900)' }}>
                  v{entry.version}
                </td>
                <td style={{ padding: '8px 12px', textAlign: 'right', color: 'var(--gray-600)' }}>
                  {entry.images ?? '-'}
                </td>
                <td style={{ padding: '8px 12px', textAlign: 'right' }}>
                  {entry.map50 != null ? (
                    <span style={{ fontWeight: 600, color: mapColor(entry.map50) }}>
                      {(entry.map50 * 100).toFixed(1)}%
                    </span>
                  ) : (
                    '-'
                  )}
                </td>
                <td style={{ padding: '8px 12px', textAlign: 'right' }}>
                  {entry.map50 != null && prevMap != null ? (
                    <DeltaBadge current={entry.map50} previous={prevMap} />
                  ) : (
                    <span style={{ color: 'var(--gray-300)', fontSize: 10 }}>—</span>
                  )}
                </td>
                <td style={{ padding: '8px 12px', textAlign: 'right', color: 'var(--gray-500)' }}>
                  {entry.new_images ? `+${entry.new_images}` : '-'}
                </td>
                <td style={{ padding: '8px 12px', color: 'var(--gray-500)' }}>
                  {entry.archived_at
                    ? new Date(entry.archived_at).toLocaleDateString()
                    : '-'}
                </td>
                <td style={{ padding: '8px 12px', textAlign: 'center' }}>
                  {entry.has_model ? (
                    <span
                      style={{
                        display: 'inline-block',
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        background: '#10b981',
                      }}
                    />
                  ) : (
                    <span style={{ color: 'var(--gray-300)' }}>—</span>
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

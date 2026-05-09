import { useEffect, useState } from 'react'
import { fetchModelStatus, ModelPrepState } from '../api/worker'

function formatSize(bytes: number) {
  if (!bytes) return '0 B'
  const gb = bytes / 1024 / 1024 / 1024
  if (gb >= 1) return `${gb.toFixed(2)} GB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function StatusPill({ installed, warning }: { installed: boolean; warning?: boolean }) {
  const color = warning ? '#f59e0b' : installed ? 'var(--success-green)' : 'var(--ship-red)'
  const label = warning ? '未完成' : installed ? '已安装' : '未安装'

  return (
    <span
      style={{
        padding: '3px 8px',
        borderRadius: 999,
        background: `${color}14`,
        color,
        fontSize: 11,
        fontWeight: 600,
      }}
    >
      {label}
    </span>
  )
}

function ModelRow({ name, detail, size, installed, warning }: { name: string; detail: string; size: string; installed: boolean; warning?: boolean }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, padding: '10px 0', borderBottom: '1px solid var(--gray-100)' }}>
      <div style={{ minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--gray-700)' }}>{name}</span>
          <StatusPill installed={installed} warning={warning} />
        </div>
        <div style={{ fontSize: 11, color: 'var(--gray-400)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{detail}</div>
      </div>
      <div style={{ fontSize: 12, color: 'var(--gray-500)', flexShrink: 0 }}>{size}</div>
    </div>
  )
}

export default function ModelStatus() {
  const [state, setState] = useState<ModelPrepState | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        setState(await fetchModelStatus())
        setFailed(false)
      } catch {
        setFailed(true)
      }
    }

    fetchStatus()
    const interval = setInterval(fetchStatus, 10000)
    return () => clearInterval(interval)
  }, [])

  if (failed || !state?.status) {
    return (
      <div style={{ padding: '14px 16px', background: 'var(--gray-50)', borderRadius: 8, border: '1px solid var(--gray-100)', fontSize: 13, color: 'var(--gray-400)' }}>
        Worker 未启动，暂时无法读取模型安装状态
      </div>
    )
  }

  const status = state.status
  const moondreamWarning = status.moondream.incomplete_files.length > 0
  const moondreamDetail = moondreamWarning
    ? `${status.moondream.incomplete_files.length} 个未完成缓存文件`
    : status.moondream.installed
      ? `${status.moondream.complete_snapshots?.length ?? 1} 个完整快照`
      : `未发现完整快照（当前 ${status.moondream.snapshot_count ?? 0} 个快照）`

  return (
    <div style={{ padding: '6px 16px 4px', background: 'var(--gray-50)', borderRadius: 8, border: '1px solid var(--gray-100)' }}>
      <ModelRow
        name="YOLO-World"
        detail={status.yolo_world.selected_path}
        size={formatSize(status.yolo_world.worker_path.size_bytes || status.yolo_world.cwd_path.size_bytes)}
        installed={status.yolo_world.installed}
      />
      <ModelRow
        name="CLIP ViT-B-32"
        detail={status.clip.cache.path}
        size={formatSize(status.clip.cache.size_bytes)}
        installed={status.clip.installed}
      />
      <ModelRow
        name="Moondream2"
        detail={moondreamDetail}
        size={formatSize(status.moondream.cache.size_bytes)}
        installed={status.moondream.installed}
        warning={moondreamWarning}
      />
    </div>
  )
}

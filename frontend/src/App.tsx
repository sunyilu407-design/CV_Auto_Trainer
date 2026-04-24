import { useState, useEffect } from 'react'
import './styles/globals.css'
import Upload from './pages/Upload'
import IntentConfirm from './pages/IntentConfirm'
import AlgorithmPlan from './pages/AlgorithmPlan'
import LabelingProgress from './pages/LabelingProgress'
import AugmentConfig from './pages/AugmentConfig'
import ReviewSamples from './pages/ReviewSamples'
import OfflineValidation from './pages/OfflineValidation'
import TrainConfig from './pages/TrainConfig'
import TrainingMonitor from './pages/TrainingMonitor'
import Delivery from './pages/Delivery'
import VideoInferenceDemo from './pages/VideoInferenceDemo'
import SettingsPanel from './components/SettingsPanel'
import Login from './pages/Login'
import { useTaskStore, Stage } from './store/taskStore'
import { useSettingsStore } from './store/settingsStore'
import { useAuthStore } from './store/authStore'

const WORKFLOW_STAGES: { key: Stage; label: string; accent: string }[] = [
  { key: 'upload', label: '需求录入', accent: '#0a72ef' },
  { key: 'intent_confirm', label: '需求协商', accent: '#0a72ef' },
  { key: 'algorithm_plan', label: '能力草案', accent: '#0a72ef' },
  { key: 'labeling', label: '数据准备', accent: '#0a72ef' },
  { key: 'augment', label: '样本增强', accent: '#de1d8d' },
  { key: 'review', label: '质量复核', accent: '#de1d8d' },
  { key: 'offline_validation', label: '离线验证', accent: '#de1d8d' },
  { key: 'train_config', label: '训练配置', accent: '#de1d8d' },
  { key: 'training', label: '训练执行', accent: '#de1d8d' },
  { key: 'video_inference', label: '效果演示', accent: '#ff5b4f' },
  { key: 'delivery', label: '算法交付', accent: '#ff5b4f' },
]

function App() {
  const { stage } = useTaskStore()
  const [showSettings, setShowSettings] = useState(false)
  const { loadSettings } = useSettingsStore()
  const { isLoggedIn, user, logout } = useAuthStore()

  useEffect(() => {
    if (isLoggedIn) {
      loadSettings()
    }
  }, [isLoggedIn])

  const activeIndex = WORKFLOW_STAGES.findIndex((s) => s.key === stage)

  // Not logged in — show login page
  if (!isLoggedIn) {
    return <Login />
  }

  const handleLogout = async () => {
    const token = useAuthStore.getState().token
    if (token) {
      try {
        await fetch('/api/auth/logout', {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}` },
        })
      } catch {
        // ignore
      }
    }
    logout()
  }

  return (
    <div style={{ minHeight: '100vh', background: 'var(--gray-50)' }}>
      {/* ── Header ── */}
      <header
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 100,
          background: 'rgba(255, 255, 255, 0.85)',
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
          borderBottom: '1px solid var(--gray-100)',
        }}
      >
        {/* Top bar */}
        <div
          style={{
            maxWidth: 1100,
            margin: '0 auto',
            padding: '0 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            height: 52,
          }}
        >
          {/* Logo */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <rect width="24" height="24" rx="6" fill="#171717" />
              <path d="M6 12L10 16L18 8" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span
              style={{
                fontSize: 14,
                fontWeight: 600,
                letterSpacing: '-0.5px',
                color: 'var(--gray-900)',
              }}
            >
              CV Auto Trainer
            </span>
          </div>

          {/* Right side */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            {/* User badge */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '4px 10px',
                background: 'var(--gray-100)',
                borderRadius: 6,
                fontSize: 12,
                fontWeight: 500,
                color: 'var(--gray-600)',
              }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2" />
                <circle cx="12" cy="7" r="4" />
              </svg>
              {user?.username}
              {user?.role === 'admin' && (
                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 600,
                    padding: '1px 5px',
                    background: 'var(--gray-900)',
                    color: '#fff',
                    borderRadius: 4,
                    letterSpacing: '0.03em',
                  }}
                >
                  ADMIN
                </span>
              )}
            </div>

            {/* Settings */}
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => setShowSettings(true)}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="3" />
                <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z" />
              </svg>
              设置
            </button>

            {/* Logout */}
            <button
              className="btn btn-sm"
              style={{
                background: 'transparent',
                color: 'var(--gray-400)',
                border: 'none',
                padding: '6px 8px',
                fontSize: 12,
              }}
              onClick={handleLogout}
              title="退出登录"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4" />
                <polyline points="16 17 21 12 16 7" />
                <line x1="21" y1="12" x2="9" y2="12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Workflow Pipeline */}
        <div
          style={{
            maxWidth: 1100,
            margin: '0 auto',
            padding: '0 24px 12px',
            overflowX: 'auto',
          }}
        >
          <nav className="workflow-nav" style={{ display: 'inline-flex', minWidth: '100%' }}>
            {WORKFLOW_STAGES.map((s, i) => {
              const isActive = s.key === stage
              const isCompleted = activeIndex > i
              const isLast = i === WORKFLOW_STAGES.length - 1

              return (
                <div key={s.key} className="flex items-center">
                  <div
                    className={`workflow-step ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}
                    style={{ fontWeight: isActive ? 600 : 400 }}
                  >
                    <div
                      className="workflow-step-dot"
                      style={{
                        width: 6,
                        height: 6,
                        borderRadius: '50%',
                        background: isActive ? '#fff' : isCompleted ? s.accent : 'var(--gray-300)',
                        boxShadow: isActive ? '0 0 0 2px rgba(255,255,255,0.3)' : 'none',
                        flexShrink: 0,
                        transition: 'all 0.2s ease',
                      }}
                    />
                    {s.label}
                  </div>
                  {!isLast && (
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke={isCompleted ? 'var(--gray-300)' : 'var(--gray-200)'}
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      style={{ flexShrink: 0, margin: '0 2px' }}
                    >
                      <path d="M9 18l6-6-6-6" />
                    </svg>
                  )}
                </div>
              )
            })}
          </nav>
        </div>
      </header>

      {/* ── Main Content ── */}
      <main style={{ padding: '32px 24px 64px' }}>
        <div style={{ maxWidth: 960, margin: '0 auto' }}>
          <div key={stage} className="animate-fade-in">
            {stage === 'upload' && <Upload />}
            {stage === 'intent_confirm' && <IntentConfirm />}
            {stage === 'algorithm_plan' && <AlgorithmPlan />}
            {stage === 'labeling' && <LabelingProgress />}
            {stage === 'augment' && <AugmentConfig />}
            {stage === 'review' && <ReviewSamples />}
            {stage === 'offline_validation' && <OfflineValidation />}
            {stage === 'train_config' && <TrainConfig />}
            {stage === 'training' && <TrainingMonitor />}
            {stage === 'video_inference' && <VideoInferenceDemo />}
            {stage === 'delivery' && <Delivery />}
          </div>
        </div>
      </main>

      {/* ── Settings Modal ── */}
      {showSettings && (
        <SettingsPanel onClose={() => setShowSettings(false)} />
      )}
    </div>
  )
}

export default App

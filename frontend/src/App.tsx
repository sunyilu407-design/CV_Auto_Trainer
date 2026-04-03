import { useState } from 'react'
import Upload from './pages/Upload'
import IntentConfirm from './pages/IntentConfirm'
import LabelingProgress from './pages/LabelingProgress'
import AugmentConfig from './pages/AugmentConfig'
import ReviewSamples from './pages/ReviewSamples'
import TrainConfig from './pages/TrainConfig'
import TrainingMonitor from './pages/TrainingMonitor'
import Delivery from './pages/Delivery'
import SettingsPanel from './components/SettingsPanel'
import { useTaskStore } from './store/taskStore'

function App() {
  const { stage } = useTaskStore()
  const [showSettings, setShowSettings] = useState(false)

  const renderPage = () => {
    switch (stage) {
      case 'upload':
        return <Upload />
      case 'intent_confirm':
        return <IntentConfirm />
      case 'labeling':
        return <LabelingProgress />
      case 'augment':
        return <AugmentConfig />
      case 'review':
        return <ReviewSamples />
      case 'train_config':
        return <TrainConfig />
      case 'training':
        return <TrainingMonitor />
      case 'delivery':
        return <Delivery />
      default:
        return <Upload />
    }
  }

  return (
    <div style={{ minHeight: '100vh', background: '#f5f5f5' }}>
      <header style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '12px 24px',
        background: '#fff',
        borderBottom: '1px solid #e0e0e0',
      }}>
        <h1 style={{ fontSize: '18px', fontWeight: 600, margin: 0 }}>
          CV 自动化训练中台
        </h1>
        <button
          onClick={() => setShowSettings(true)}
          style={{
            padding: '6px 16px',
            background: '#fff',
            border: '1px solid #d0d0d0',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px',
          }}
        >
          设置
        </button>
      </header>

      <main style={{ padding: '24px' }}>
        {renderPage()}
      </main>

      {showSettings && (
        <SettingsPanel onClose={() => setShowSettings(false)} />
      )}
    </div>
  )
}

export default App

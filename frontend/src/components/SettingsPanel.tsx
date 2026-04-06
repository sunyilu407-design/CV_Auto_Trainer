import { useState } from 'react'
import { useSettingsStore, VlmProvider, VlmApiFormat } from '../store/settingsStore'
import { vlmApi } from '../api/backend'

interface Props {
  onClose: () => void
}

// Provider 默认配置（使用各厂家最新多模态模型，2026年4月）
const PROVIDER_DEFAULTS: Record<VlmProvider, { baseUrl: string; apiFormat: VlmApiFormat; model: string }> = {
  openai: { baseUrl: 'https://api.openai.com/v1', apiFormat: 'openai', model: 'gpt-4.1' },
  kimi: { baseUrl: 'https://api.moonshot.cn/v1', apiFormat: 'openai', model: 'kimi-k2.5' },
  minimax: { baseUrl: 'https://api.minimax.chat/v1', apiFormat: 'openai', model: 'MiniMax-M2.7' },
  zhipu: { baseUrl: 'https://open.bigmodel.cn/api/paas/v4', apiFormat: 'openai', model: 'glm-4v-plus' },
  gemini: { baseUrl: 'https://generativelanguage.googleapis.com/v1beta', apiFormat: 'gemini', model: 'gemini-2.5-flash' },
  claude: { baseUrl: 'https://api.anthropic.com/v1', apiFormat: 'anthropic', model: 'claude-sonnet-4-6' },
  custom: { baseUrl: '', apiFormat: 'openai', model: '' },
}

export default function SettingsPanel({ onClose }: Props) {
  const { settings, setSettings, saveSettings } = useSettingsStore()
  const [activeTab, setActiveTab] = useState<'vlm' | 'cloud' | 'general'>('vlm')
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null)

  const handleProviderChange = (provider: VlmProvider) => {
    const defaults = PROVIDER_DEFAULTS[provider]
    setSettings({
      vlmProvider: provider,
      vlmBaseUrl: defaults.baseUrl,
      vlmApiFormat: defaults.apiFormat,
      vlmModel: defaults.model,
    })
    setTestResult(null)
  }

  const handleTest = async () => {
    setTesting(true)
    setTestResult(null)
    try {
      const result = await vlmApi.test({
        provider: settings.vlmProvider,
        base_url: settings.vlmBaseUrl,
        api_key: settings.vlmApiKey,
        api_format: settings.vlmApiFormat,
        model: settings.vlmModel,
      })
      setTestResult(result)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : '测试失败'
      setTestResult({ success: false, message: msg })
    } finally {
      setTesting(false)
    }
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      await saveSettings(settings)
      onClose()
    } finally {
      setSaving(false)
    }
  }

  const showCustomFields = settings.vlmProvider === 'custom' || settings.vlmProvider === 'claude' || settings.vlmProvider === 'openai' || settings.vlmProvider === 'kimi' || settings.vlmProvider === 'minimax' || settings.vlmProvider === 'zhipu' || settings.vlmProvider === 'gemini'

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
    >
      <div
        style={{
          background: '#fff',
          borderRadius: '8px',
          width: '600px',
          maxHeight: '80vh',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <div
          style={{
            padding: '16px 20px',
            borderBottom: '1px solid #e0e0e0',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <h2 style={{ fontSize: '18px', margin: 0 }}>设置</h2>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              fontSize: '20px',
              cursor: 'pointer',
              color: '#666',
            }}
          >
            ×
          </button>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', borderBottom: '1px solid #e0e0e0' }}>
          {([
            { key: 'vlm', label: 'VLM' },
            { key: 'cloud', label: '云端训练' },
            { key: 'general', label: '全局' },
          ] as const).map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key)}
              style={{
                flex: 1,
                padding: '10px',
                background: 'none',
                border: 'none',
                borderBottom: activeTab === key ? '2px solid #1976d2' : '2px solid transparent',
                color: activeTab === key ? '#1976d2' : '#666',
                cursor: 'pointer',
                fontSize: '14px',
              }}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div style={{ padding: '20px', overflowY: 'auto', flex: 1 }}>
          {activeTab === 'vlm' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div>
                <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>VLM Provider</label>
                <select
                  value={settings.vlmProvider}
                  onChange={(e) => handleProviderChange(e.target.value as VlmProvider)}
                  style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px' }}
                >
                  <option value="openai">OpenAI (GPT-4o)</option>
                  <option value="kimi">Kimi (moonshot-v1-32k)</option>
                  <option value="minimax">MiniMax (minimax-v-01)</option>
                  <option value="zhipu">智谱 GLM-4V</option>
                  <option value="gemini">Google Gemini</option>
                  <option value="claude">Anthropic Claude</option>
                  <option value="custom">自定义 / 中转代理</option>
                </select>
              </div>

              {showCustomFields && (
                <>
                  <div>
                    <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>API 格式</label>
                    <select
                      value={settings.vlmApiFormat}
                      onChange={(e) => setSettings({ vlmApiFormat: e.target.value as VlmApiFormat })}
                      style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px' }}
                    >
                      <option value="openai">OpenAI 兼容 (/v1/chat/completions)</option>
                      <option value="anthropic">Anthropic (/v1/messages)</option>
                      <option value="gemini">Google Gemini</option>
                    </select>
                  </div>
                  <div>
                    <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>模型名称</label>
                    <input
                      value={settings.vlmModel}
                      onChange={(e) => setSettings({ vlmModel: e.target.value })}
                      placeholder="输入模型名称，如 gpt-4o-2024-11-20"
                      style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px', boxSizing: 'border-box' }}
                    />
                  </div>
                </>
              )}

              <div>
                <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>Base URL</label>
                <input
                  value={settings.vlmBaseUrl}
                  onChange={(e) => setSettings({ vlmBaseUrl: e.target.value })}
                  placeholder="https://api.openai.com/v1"
                  style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px', boxSizing: 'border-box' }}
                />
              </div>
              <div>
                <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>API Key</label>
                <input
                  type="password"
                  value={settings.vlmApiKey}
                  onChange={(e) => setSettings({ vlmApiKey: e.target.value })}
                  placeholder="sk-..."
                  style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px', boxSizing: 'border-box' }}
                />
              </div>

              {/* 测试按钮和结果 */}
              <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                <button
                  onClick={handleTest}
                  disabled={testing || !settings.vlmApiKey}
                  style={{
                    padding: '8px 20px',
                    background: testing ? '#ccc' : '#4caf50',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: testing || !settings.vlmApiKey ? 'not-allowed' : 'pointer',
                    fontSize: '14px',
                  }}
                >
                  {testing ? '测试中...' : '测试连接'}
                </button>
                {testResult && (
                  <span
                    style={{
                      color: testResult.success ? '#4caf50' : '#f44336',
                      fontSize: '14px',
                    }}
                  >
                    {testResult.message}
                  </span>
                )}
              </div>
            </div>
          )}

          {activeTab === 'cloud' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div>
                <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>云端提供商</label>
                <select
                  value={settings.cloudProvider}
                  onChange={(e) => setSettings({ cloudProvider: e.target.value as typeof settings.cloudProvider })}
                  style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px' }}
                >
                  <option value="generic">通用 SSH（阿里云/腾讯云/AWS/自有服务器）</option>
                  <option value="autodl">AutoDL（自动创建/销毁 GPU 实例）</option>
                </select>
              </div>

              {settings.cloudProvider === 'generic' ? (
                <>
                  <div style={{ display: 'flex', gap: '12px' }}>
                    <div style={{ flex: 2 }}>
                      <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>SSH Host</label>
                      <input
                        value={settings.sshHost}
                        onChange={(e) => setSettings({ sshHost: e.target.value })}
                        placeholder="123.45.67.89"
                        style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px', boxSizing: 'border-box' }}
                      />
                    </div>
                    <div style={{ flex: 1 }}>
                      <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>端口</label>
                      <input
                        type="number"
                        value={settings.sshPort}
                        onChange={(e) => setSettings({ sshPort: Number(e.target.value) })}
                        style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px', boxSizing: 'border-box' }}
                      />
                    </div>
                  </div>
                  <div>
                    <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>SSH 用户名</label>
                    <input
                      value={settings.sshUsername}
                      onChange={(e) => setSettings({ sshUsername: e.target.value })}
                      placeholder="root"
                      style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px', boxSizing: 'border-box' }}
                    />
                  </div>
                  <div>
                    <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>SSH 密码</label>
                    <input
                      type="password"
                      value={settings.sshPassword}
                      onChange={(e) => setSettings({ sshPassword: e.target.value })}
                      placeholder="密码或私钥二选一"
                      style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px', boxSizing: 'border-box' }}
                    />
                  </div>
                  <div>
                    <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>SSH 私钥路径</label>
                    <input
                      value={settings.sshPrivateKeyPath}
                      onChange={(e) => setSettings({ sshPrivateKeyPath: e.target.value })}
                      placeholder="~/.ssh/id_rsa"
                      style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px', boxSizing: 'border-box' }}
                    />
                  </div>
                  <div>
                    <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>远程工作目录</label>
                    <input
                      value={settings.remoteWorkDir}
                      onChange={(e) => setSettings({ remoteWorkDir: e.target.value })}
                      placeholder="/root/workspace"
                      style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px', boxSizing: 'border-box' }}
                    />
                  </div>
                </>
              ) : (
                <div>
                  <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>AutoDL Token</label>
                  <input
                    type="password"
                    value={settings.autodlToken}
                    onChange={(e) => setSettings({ autodlToken: e.target.value })}
                    placeholder="AutoDL API Token"
                    style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px', boxSizing: 'border-box' }}
                  />
                  <p style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
                    请在 AutoDL 控制台 → API Token 中获取
                  </p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'general' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div>
                <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>默认训练模式</label>
                <select
                  value={settings.defaultTrainMode}
                  onChange={(e) => setSettings({ defaultTrainMode: e.target.value as typeof settings.defaultTrainMode })}
                  style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px' }}
                >
                  <option value="local">本地训练</option>
                  <option value="cloud">云端训练</option>
                </select>
              </div>
              <div>
                <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>默认模型</label>
                <select
                  value={settings.defaultModel}
                  onChange={(e) => setSettings({ defaultModel: e.target.value })}
                  style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px' }}
                >
                  <option value="yolo11n.pt">YOLO11n</option>
                  <option value="yolo11s.pt">YOLO11s（默认）</option>
                  <option value="yolo11m.pt">YOLO11m</option>
                  <option value="yolo11l.pt">YOLO11l</option>
                  <option value="rtdetr-l.pt">RT-DETR-L</option>
                </select>
              </div>
              <div>
                <label style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>默认增强强度</label>
                <select
                  value={settings.defaultAugmentStrength}
                  onChange={(e) => setSettings({ defaultAugmentStrength: e.target.value as typeof settings.defaultAugmentStrength })}
                  style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px' }}
                >
                  <option value="light">轻度</option>
                  <option value="medium">中度（默认）</option>
                  <option value="heavy">重度</option>
                </select>
              </div>
              <div>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={settings.defaultDeleteOriginal}
                    onChange={(e) => setSettings({ defaultDeleteOriginal: e.target.checked })}
                  />
                  <span style={{ fontSize: '14px' }}>完成后默认删除原图</span>
                </label>
              </div>
            </div>
          )}
        </div>

        <div style={{ padding: '16px 20px', borderTop: '1px solid #e0e0e0', display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
          <button
            onClick={onClose}
            style={{
              padding: '8px 20px',
              background: '#fff',
              border: '1px solid #ccc',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '14px',
            }}
          >
            取消
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            style={{
              padding: '8px 20px',
              background: saving ? '#ccc' : '#1976d2',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              cursor: saving ? 'not-allowed' : 'pointer',
              fontSize: '14px',
            }}
          >
            {saving ? '保存中...' : '保存设置'}
          </button>
        </div>
      </div>
    </div>
  )
}

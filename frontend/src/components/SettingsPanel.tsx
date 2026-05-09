import { useState } from 'react'
import { useSettingsStore, VlmProvider, VlmApiFormat, ReasoningProvider } from '../store/settingsStore'
import { vlmApi, reasoningApi } from '../api/backend'
import ModelCacheTab from './ModelCacheTab'

interface Props {
  onClose: () => void
}

const PROVIDER_DEFAULTS: Record<VlmProvider, { baseUrl: string; apiFormat: VlmApiFormat; model: string }> = {
  openai: { baseUrl: 'https://api.openai.com/v1', apiFormat: 'openai', model: 'gpt-4.1' },
  kimi: { baseUrl: 'https://api.moonshot.cn/v1', apiFormat: 'openai', model: 'kimi-k2.5' },
  minimax: { baseUrl: 'https://api.minimax.chat/v1', apiFormat: 'openai', model: 'MiniMax-M2.7' },
  zhipu: { baseUrl: 'https://open.bigmodel.cn/api/paas/v4', apiFormat: 'openai', model: 'glm-4v-plus' },
  gemini: { baseUrl: 'https://generativelanguage.googleapis.com/v1beta', apiFormat: 'gemini', model: 'gemini-2.5-flash' },
  claude: { baseUrl: 'https://api.anthropic.com/v1', apiFormat: 'anthropic', model: 'claude-sonnet-4-6' },
  custom: { baseUrl: '', apiFormat: 'openai', model: '' },
}

const REASONING_PROVIDER_DEFAULTS: Record<ReasoningProvider, { baseUrl: string; model: string; label: string }> = {
  deepseek: { baseUrl: 'https://api.deepseek.com/v1', model: 'deepseek-reasoner', label: 'DeepSeek-R1（推荐 / 国内可达）' },
  openai: { baseUrl: 'https://api.openai.com/v1', model: 'o3-mini', label: 'OpenAI o3-mini' },
  kimi: { baseUrl: 'https://api.moonshot.cn/v1', model: 'kimi-thinking-preview', label: 'Kimi (k1.5 thinking)' },
  qwen: { baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1', model: 'qwq-plus', label: '通义千问 QwQ' },
  zhipu: { baseUrl: 'https://open.bigmodel.cn/api/paas/v4', model: 'glm-zero-preview', label: '智谱 GLM-Zero' },
  custom: { baseUrl: '', model: '', label: '自定义 / 中转代理' },
}

export default function SettingsPanel({ onClose }: Props) {
  const { settings, setSettings, saveSettings } = useSettingsStore()
  const [activeTab, setActiveTab] = useState<'vlm' | 'reasoning' | 'cloud' | 'general' | 'cache'>('vlm')
  const [saving, setSaving] = useState(false)
  const [saveSuccessOpen, setSaveSuccessOpen] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null)
  const [reasoningTesting, setReasoningTesting] = useState(false)
  const [reasoningTestResult, setReasoningTestResult] = useState<{ success: boolean; message: string } | null>(null)

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

  const handleReasoningProviderChange = (provider: ReasoningProvider) => {
    const defaults = REASONING_PROVIDER_DEFAULTS[provider]
    setSettings({
      reasoningProvider: provider,
      reasoningBaseUrl: defaults.baseUrl,
      reasoningModel: defaults.model,
    })
    setReasoningTestResult(null)
  }

  const handleReasoningTest = async () => {
    setReasoningTesting(true)
    setReasoningTestResult(null)
    try {
      const result = await reasoningApi.test({
        provider: settings.reasoningProvider,
        base_url: settings.reasoningBaseUrl,
        api_key: settings.reasoningApiKey,
        model: settings.reasoningModel,
      })
      setReasoningTestResult(result)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : '测试失败'
      setReasoningTestResult({ success: false, message: msg })
    } finally {
      setReasoningTesting(false)
    }
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      const result = await saveSettings(settings)
      if (result.success) {
        setTestResult(null)
        setSaveSuccessOpen(true)
      } else {
        setTestResult({ success: false, message: result.message })
      }
    } finally {
      setSaving(false)
    }
  }

  const tabs: { key: 'vlm' | 'reasoning' | 'cloud' | 'general' | 'cache'; label: string }[] = [
    { key: 'vlm', label: 'VLM' },
    { key: 'reasoning', label: '推理模型' },
    { key: 'cloud', label: '云端训练' },
    { key: 'general', label: '全局' },
    { key: 'cache', label: '模型缓存' },
  ]

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'hsla(0, 0%, 0%, 0.4)',
        backdropFilter: 'blur(4px)',
        WebkitBackdropFilter: 'blur(4px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        animation: 'fadeIn 0.15s ease-out',
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div
        style={{
          background: '#fff',
          borderRadius: 12,
          width: 560,
          maxHeight: '85vh',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
          boxShadow: '0 20px 60px rgba(0,0,0,0.15), 0 0 0 1px rgba(0,0,0,0.05)',
          animation: 'fadeInScale 0.2s ease-out',
        }}
      >
        {saveSuccessOpen && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              background: 'rgba(255,255,255,0.96)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 24,
              zIndex: 1,
            }}
          >
            <div
              style={{
                width: '100%',
                maxWidth: 360,
                padding: '28px 24px',
                borderRadius: 12,
                border: '1px solid var(--gray-100)',
                background: '#fff',
                boxShadow: '0 12px 32px rgba(0,0,0,0.08)',
                textAlign: 'center',
              }}
            >
              <div
                style={{
                  width: 52,
                  height: 52,
                  borderRadius: '50%',
                  margin: '0 auto 16px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'rgba(22,163,74,0.08)',
                  color: '#16a34a',
                }}
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              </div>
              <h3 style={{ margin: 0, fontSize: 18, fontWeight: 600, color: 'var(--gray-900)' }}>设置已保存</h3>
              <p style={{ margin: '8px 0 0', fontSize: 13, lineHeight: 1.6, color: 'var(--gray-400)' }}>
                当前配置已经成功写入，点击确定关闭设置面板。
              </p>
              <div style={{ marginTop: 22, display: 'flex', justifyContent: 'center' }}>
                <button
                  className="btn btn-primary"
                  onClick={onClose}
                  style={{ minWidth: 120, padding: '8px 20px' }}
                >
                  确定
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Header */}
        <div
          style={{
            padding: '16px 20px',
            borderBottom: '1px solid var(--gray-100)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <h2 style={{ fontSize: 15, fontWeight: 600, margin: 0, letterSpacing: '-0.3px' }}>设置</h2>
          <button
            onClick={onClose}
            style={{
              background: 'var(--gray-100)',
              border: 'none',
              borderRadius: 6,
              width: 28,
              height: 28,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              color: 'var(--gray-600)',
              fontSize: 16,
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--gray-200)' }}
            onMouseLeave={(e) => { e.currentTarget.style.background = 'var(--gray-100)' }}
          >
            ×
          </button>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', borderBottom: '1px solid var(--gray-100)', padding: '0 20px' }}>
          {tabs.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key)}
              style={{
                flex: 1,
                padding: '12px 8px',
                background: 'none',
                border: 'none',
                borderBottom: `2px solid ${activeTab === key ? 'var(--gray-900)' : 'transparent'}`,
                color: activeTab === key ? 'var(--gray-900)' : 'var(--gray-400)',
                fontWeight: activeTab === key ? 600 : 400,
                fontSize: 13,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                marginBottom: -1,
              }}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div style={{ padding: 20, overflowY: 'auto', flex: 1 }}>

          {/* ── VLM Tab ── */}
          {activeTab === 'vlm' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div className="form-group">
                <label className="form-label">VLM Provider</label>
                <select
                  className="input"
                  value={settings.vlmProvider}
                  onChange={(e) => handleProviderChange(e.target.value as VlmProvider)}
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

              {(settings.vlmProvider === 'custom' || settings.vlmProvider === 'claude' || settings.vlmProvider === 'openai' || settings.vlmProvider === 'kimi' || settings.vlmProvider === 'minimax' || settings.vlmProvider === 'zhipu' || settings.vlmProvider === 'gemini') && (
                <>
                  <div className="form-group">
                    <label className="form-label">API 格式</label>
                    <select
                      className="input"
                      value={settings.vlmApiFormat}
                      onChange={(e) => setSettings({ vlmApiFormat: e.target.value as VlmApiFormat })}
                    >
                      <option value="openai">OpenAI 兼容 (/v1/chat/completions)</option>
                      <option value="anthropic">Anthropic (/v1/messages)</option>
                      <option value="gemini">Google Gemini</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label className="form-label">模型名称</label>
                    <input
                      className="input"
                      value={settings.vlmModel}
                      onChange={(e) => setSettings({ vlmModel: e.target.value })}
                      placeholder="如 gpt-4o-2024-11-20"
                    />
                  </div>
                </>
              )}

              <div className="form-group">
                <label className="form-label">Base URL</label>
                <input
                  className="input"
                  value={settings.vlmBaseUrl}
                  onChange={(e) => setSettings({ vlmBaseUrl: e.target.value })}
                  placeholder="https://api.openai.com/v1"
                />
              </div>

              <div className="form-group">
                <label className="form-label">API Key</label>
                <input
                  type="password"
                  className="input"
                  value={settings.vlmApiKey}
                  onChange={(e) => setSettings({ vlmApiKey: e.target.value })}
                  placeholder="sk-..."
                />
              </div>

              {/* Test Connection */}
              <div
                style={{
                  display: 'flex',
                  gap: 12,
                  alignItems: 'center',
                  padding: '12px 14px',
                  background: 'var(--gray-50)',
                  borderRadius: 8,
                  border: '1px solid var(--gray-100)',
                }}
              >
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={handleTest}
                  disabled={testing || !settings.vlmApiKey}
                  style={{ flexShrink: 0 }}
                >
                  {testing ? (
                    <>
                      <div className="spinner" />
                      测试中...
                    </>
                  ) : (
                    <>
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                      测试连接
                    </>
                  )}
                </button>
                {testResult && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    {testResult.success ? (
                      <>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#16a34a" strokeWidth="2"><polyline points="20 6 9 17 4 12"/></svg>
                        <span style={{ fontSize: 12, color: '#16a34a', fontWeight: 500 }}>{testResult.message}</span>
                      </>
                    ) : (
                      <>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--ship-red)" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                        <span style={{ fontSize: 12, color: 'var(--ship-red)', fontWeight: 500 }}>{testResult.message}</span>
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── Reasoning Tab ── */}
          {activeTab === 'reasoning' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div
                style={{
                  padding: '10px 12px',
                  borderRadius: 8,
                  background: 'rgba(59,130,246,0.06)',
                  border: '1px solid rgba(59,130,246,0.18)',
                  fontSize: 12,
                  lineHeight: 1.7,
                  color: 'var(--gray-600)',
                }}
              >
                <strong style={{ color: 'var(--develop-blue)' }}>推理模型用途：</strong>在「需求确认 → 算法方案」之间，对 VLM 给出的类别词做二次归一化，把抽象/状态词替换成 CLIP/YOLO-World 友好的英文物体词，显著提升首轮自动打标召回。未配置时主流程不受影响（会跳过这一步）。
              </div>

              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '12px 14px',
                  background: 'var(--gray-50)',
                  borderRadius: 8,
                  border: '1px solid var(--gray-100)',
                }}
              >
                <div>
                  <p style={{ fontSize: 13, fontWeight: 500, margin: 0 }}>启用推理模型决策层</p>
                  <p style={{ fontSize: 11, color: 'var(--gray-400)', margin: '2px 0 0' }}>关闭后类别归一化将完全跳过</p>
                </div>
                <label style={{ cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={settings.reasoningEnabled}
                    onChange={(e) => setSettings({ reasoningEnabled: e.target.checked })}
                    style={{ width: 0, height: 0, opacity: 0, position: 'absolute' }}
                  />
                  <div
                    style={{
                      width: 36,
                      height: 20,
                      borderRadius: 10,
                      background: settings.reasoningEnabled ? 'var(--develop-blue)' : 'var(--gray-200)',
                      position: 'relative',
                      transition: 'all 0.2s ease',
                      cursor: 'pointer',
                    }}
                  >
                    <div
                      style={{
                        width: 14,
                        height: 14,
                        borderRadius: '50%',
                        background: '#fff',
                        position: 'absolute',
                        top: 3,
                        left: settings.reasoningEnabled ? 19 : 3,
                        transition: 'left 0.2s ease',
                        boxShadow: '0 1px 2px rgba(0,0,0,0.15)',
                      }}
                    />
                  </div>
                </label>
              </div>

              <div className="form-group">
                <label className="form-label">推理模型 Provider</label>
                <select
                  className="input"
                  value={settings.reasoningProvider}
                  onChange={(e) => handleReasoningProviderChange(e.target.value as ReasoningProvider)}
                  disabled={!settings.reasoningEnabled}
                >
                  {(Object.entries(REASONING_PROVIDER_DEFAULTS) as [ReasoningProvider, { label: string }][]).map(
                    ([k, v]) => (
                      <option key={k} value={k}>{v.label}</option>
                    ),
                  )}
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">模型名称</label>
                <input
                  className="input"
                  value={settings.reasoningModel}
                  onChange={(e) => setSettings({ reasoningModel: e.target.value })}
                  disabled={!settings.reasoningEnabled}
                  placeholder="如 deepseek-reasoner / o3-mini"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Base URL</label>
                <input
                  className="input"
                  value={settings.reasoningBaseUrl}
                  onChange={(e) => setSettings({ reasoningBaseUrl: e.target.value })}
                  disabled={!settings.reasoningEnabled}
                  placeholder="https://api.deepseek.com/v1"
                />
                <p style={{ fontSize: 11, color: 'var(--gray-400)', margin: '4px 0 0' }}>
                  须为 OpenAI 兼容的 chat/completions 端点（DeepSeek / OpenAI / Kimi / Qwen / 智谱 / 中转代理均可）
                </p>
              </div>

              <div className="form-group">
                <label className="form-label">API Key</label>
                <input
                  type="password"
                  className="input"
                  value={settings.reasoningApiKey}
                  onChange={(e) => setSettings({ reasoningApiKey: e.target.value })}
                  disabled={!settings.reasoningEnabled}
                  placeholder="sk-..."
                />
              </div>

              <div
                style={{
                  display: 'flex',
                  gap: 12,
                  alignItems: 'center',
                  padding: '12px 14px',
                  background: 'var(--gray-50)',
                  borderRadius: 8,
                  border: '1px solid var(--gray-100)',
                }}
              >
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={handleReasoningTest}
                  disabled={reasoningTesting || !settings.reasoningEnabled || !settings.reasoningApiKey}
                  style={{ flexShrink: 0 }}
                >
                  {reasoningTesting ? (
                    <>
                      <div className="spinner" />
                      测试中...
                    </>
                  ) : (
                    <>
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                      测试连接
                    </>
                  )}
                </button>
                {reasoningTestResult && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    {reasoningTestResult.success ? (
                      <>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#16a34a" strokeWidth="2"><polyline points="20 6 9 17 4 12"/></svg>
                        <span style={{ fontSize: 12, color: '#16a34a', fontWeight: 500 }}>{reasoningTestResult.message}</span>
                      </>
                    ) : (
                      <>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--ship-red)" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                        <span style={{ fontSize: 12, color: 'var(--ship-red)', fontWeight: 500 }}>{reasoningTestResult.message}</span>
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── Cloud Tab ── */}
          {activeTab === 'cloud' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div className="form-group">
                <label className="form-label">云端提供商</label>
                <select
                  className="input"
                  value={settings.cloudProvider}
                  onChange={(e) => setSettings({ cloudProvider: e.target.value as typeof settings.cloudProvider })}
                >
                  <option value="generic">通用 SSH（阿里云/腾讯云/AWS/自有服务器）</option>
                  <option value="autodl">AutoDL（自动创建/销毁 GPU 实例）</option>
                </select>
              </div>

              {settings.cloudProvider === 'generic' ? (
                <>
                  <div style={{ display: 'flex', gap: 12 }}>
                    <div style={{ flex: 2 }} className="form-group">
                      <label className="form-label">SSH Host</label>
                      <input className="input" value={settings.sshHost} onChange={(e) => setSettings({ sshHost: e.target.value })} placeholder="123.45.67.89" />
                    </div>
                    <div style={{ flex: 1 }} className="form-group">
                      <label className="form-label">端口</label>
                      <input type="number" className="input" value={settings.sshPort} onChange={(e) => setSettings({ sshPort: Number(e.target.value) })} />
                    </div>
                  </div>
                  <div className="form-group">
                    <label className="form-label">SSH 用户名</label>
                    <input className="input" value={settings.sshUsername} onChange={(e) => setSettings({ sshUsername: e.target.value })} placeholder="root" />
                  </div>
                  <div className="form-group">
                    <label className="form-label">SSH 密码</label>
                    <input type="password" className="input" value={settings.sshPassword} onChange={(e) => setSettings({ sshPassword: e.target.value })} placeholder="密码或私钥二选一" />
                  </div>
                  <div className="form-group">
                    <label className="form-label">SSH 私钥路径</label>
                    <input className="input" value={settings.sshPrivateKeyPath} onChange={(e) => setSettings({ sshPrivateKeyPath: e.target.value })} placeholder="~/.ssh/id_rsa" />
                  </div>
                  <div className="form-group">
                    <label className="form-label">远程工作目录</label>
                    <input className="input" value={settings.remoteWorkDir} onChange={(e) => setSettings({ remoteWorkDir: e.target.value })} placeholder="/root/workspace" />
                  </div>
                </>
              ) : (
                <div className="form-group">
                  <label className="form-label">AutoDL Token</label>
                  <input
                    type="password"
                    className="input"
                    value={settings.autodlToken}
                    onChange={(e) => setSettings({ autodlToken: e.target.value })}
                    placeholder="AutoDL API Token"
                  />
                  <p style={{ fontSize: 12, color: 'var(--gray-400)', marginTop: 4 }}>请在 AutoDL 控制台 → API Token 中获取</p>
                </div>
              )}
            </div>
          )}

          {/* ── General Tab ── */}
          {activeTab === 'general' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {[
                { key: 'defaultTrainMode', label: '默认训练模式', options: [{ value: 'local', label: '本地训练' }, { value: 'cloud', label: '云端训练' }] },
                { key: 'defaultModel', label: '默认模型', options: [{ value: 'yolo11n.pt', label: 'YOLO11n' }, { value: 'yolo11s.pt', label: 'YOLO11s（默认）' }, { value: 'yolo11m.pt', label: 'YOLO11m' }, { value: 'yolo11l.pt', label: 'YOLO11l' }, { value: 'rtdetr-l.pt', label: 'RT-DETR-L' }] },
                { key: 'defaultAugmentStrength', label: '默认增强强度', options: [{ value: 'light', label: '轻度' }, { value: 'medium', label: '中度（默认）' }, { value: 'heavy', label: '重度' }] },
              ].map(({ key, label, options }) => (
                <div key={key} className="form-group">
                  <label className="form-label">{label}</label>
                  <select
                    className="input"
                    value={(settings as unknown as Record<string, unknown>)[key] as string}
                    onChange={(e) => setSettings({ [key]: e.target.value } as unknown as typeof settings)}
                  >
                    {options.map((opt) => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                  </select>
                </div>
              ))}

              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '12px 14px',
                  background: 'var(--gray-50)',
                  borderRadius: 8,
                  border: '1px solid var(--gray-100)',
                }}
              >
                <div>
                  <p style={{ fontSize: 13, fontWeight: 500, margin: 0 }}>完成后默认删除原图</p>
                  <p style={{ fontSize: 11, color: 'var(--gray-400)', margin: '2px 0 0' }}>增强完成后自动清理原始图片</p>
                </div>
                <label style={{ cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={settings.defaultDeleteOriginal}
                    onChange={(e) => setSettings({ defaultDeleteOriginal: e.target.checked })}
                    style={{ width: 0, height: 0, opacity: 0, position: 'absolute' }}
                  />
                  <div
                    style={{
                      width: 36,
                      height: 20,
                      borderRadius: 10,
                      background: settings.defaultDeleteOriginal ? 'var(--develop-blue)' : 'var(--gray-200)',
                      position: 'relative',
                      transition: 'all 0.2s ease',
                      cursor: 'pointer',
                    }}
                  >
                    <div
                      style={{
                        width: 14,
                        height: 14,
                        borderRadius: '50%',
                        background: '#fff',
                        position: 'absolute',
                        top: 3,
                        left: settings.defaultDeleteOriginal ? 19 : 3,
                        transition: 'left 0.2s ease',
                        boxShadow: '0 1px 2px rgba(0,0,0,0.15)',
                      }}
                    />
                  </div>
                </label>
              </div>
            </div>
          )}
        </div>

          {/* ── Cache Tab ── */}
          {activeTab === 'cache' && <ModelCacheTab />}

        {/* Footer */}
        <div
          style={{
            padding: '14px 20px',
            borderTop: '1px solid var(--gray-100)',
            display: 'flex',
            gap: 10,
            justifyContent: 'flex-end',
            background: 'var(--gray-50)',
          }}
        >
          <button className="btn btn-secondary" onClick={onClose} style={{ padding: '8px 16px' }}>
            取消
          </button>
          <button
            className="btn btn-primary"
            onClick={handleSave}
            disabled={saving}
            style={{ padding: '8px 20px' }}
          >
            {saving ? (
              <>
                <div className="spinner" />
                保存中...
              </>
            ) : '保存设置'}
          </button>
        </div>
      </div>
    </div>
  )
}

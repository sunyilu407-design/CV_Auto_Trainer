import { create } from 'zustand'

export type CloudProvider = 'generic' | 'autodl'

export interface UserSettings {
  // VLM
  vlmProvider: 'openai' | 'kimi' | 'gemini'
  vlmBaseUrl: string
  vlmApiKey: string

  // 云端训练
  cloudProvider: CloudProvider

  // 通用 SSH
  sshHost: string
  sshPort: number
  sshUsername: string
  sshPassword: string
  sshPrivateKeyPath: string
  remoteWorkDir: string

  // AutoDL 专用
  autodlToken: string

  // 全局
  defaultModel: string
  defaultAugmentStrength: 'light' | 'medium' | 'heavy'
  defaultDeleteOriginal: boolean
  defaultGpuType: string
  defaultTrainMode: 'local' | 'cloud'
}

interface SettingsState {
  settings: UserSettings
  loading: boolean
  setSettings: (settings: Partial<UserSettings>) => void
  loadSettings: () => Promise<void>
  saveSettings: (settings: Partial<UserSettings>) => Promise<void>
}

const defaultSettings: UserSettings = {
  vlmProvider: 'openai',
  vlmBaseUrl: 'https://api.openai.com/v1',
  vlmApiKey: '',

  cloudProvider: 'generic',
  sshHost: '',
  sshPort: 22,
  sshUsername: 'root',
  sshPassword: '',
  sshPrivateKeyPath: '',
  remoteWorkDir: '/root/workspace',

  autodlToken: '',

  defaultModel: 'yolo11s.pt',
  defaultAugmentStrength: 'medium',
  defaultDeleteOriginal: false,
  defaultGpuType: 'RTX 4090',
  defaultTrainMode: 'local',
}

export const useSettingsStore = create<SettingsState>((set, get) => ({
  settings: defaultSettings,
  loading: false,

  setSettings: (updates) =>
    set((state) => ({ settings: { ...state.settings, ...updates } })),

  loadSettings: async () => {
    set({ loading: true })
    try {
      const res = await fetch('/api/settings')
      const data = await res.json()
      if (data.code === 0 && data.data) {
        set({ settings: { ...defaultSettings, ...data.data } })
      }
    } catch {
      // use defaults
    } finally {
      set({ loading: false })
    }
  },

  saveSettings: async (updates) => {
    const current = get().settings
    const merged = { ...current, ...updates }
    await fetch('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(merged),
    })
    set({ settings: merged })
  },
}))

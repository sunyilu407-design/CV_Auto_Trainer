import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { useAuthStore } from './authStore'

export type CloudProvider = 'generic' | 'autodl'
export type VlmProvider = 'openai' | 'kimi' | 'minimax' | 'zhipu' | 'gemini' | 'claude' | 'custom'
export type VlmApiFormat = 'openai' | 'anthropic' | 'gemini'
export type ReasoningProvider = 'deepseek' | 'openai' | 'kimi' | 'qwen' | 'zhipu' | 'custom'

export interface UserSettings {
  vlmProvider: VlmProvider
  vlmBaseUrl: string
  vlmApiKey: string
  vlmApiFormat: VlmApiFormat
  vlmModel: string

  reasoningEnabled: boolean
  reasoningProvider: ReasoningProvider
  reasoningBaseUrl: string
  reasoningApiKey: string
  reasoningModel: string

  cloudProvider: CloudProvider

  sshHost: string
  sshPort: number
  sshUsername: string
  sshPassword: string
  sshPrivateKeyPath: string
  remoteWorkDir: string

  autodlToken: string

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
  saveSettings: (settings: Partial<UserSettings>) => Promise<{ success: boolean; message: string }>
}

// Backend uses snake_case; frontend uses camelCase
const snakeToCamel: Record<string, string> = {
  vlm_provider: 'vlmProvider',
  vlm_base_url: 'vlmBaseUrl',
  vlm_api_key: 'vlmApiKey',
  vlm_api_format: 'vlmApiFormat',
  vlm_model: 'vlmModel',
  reasoning_enabled: 'reasoningEnabled',
  reasoning_provider: 'reasoningProvider',
  reasoning_base_url: 'reasoningBaseUrl',
  reasoning_api_key: 'reasoningApiKey',
  reasoning_model: 'reasoningModel',
  cloud_provider: 'cloudProvider',
  ssh_host: 'sshHost',
  ssh_port: 'sshPort',
  ssh_username: 'sshUsername',
  ssh_password: 'sshPassword',
  ssh_private_key_path: 'sshPrivateKeyPath',
  remote_work_dir: 'remoteWorkDir',
  autodl_token: 'autodlToken',
  default_model: 'defaultModel',
  default_augment_strength: 'defaultAugmentStrength',
  default_delete_original: 'defaultDeleteOriginal',
  default_gpu_type: 'defaultGpuType',
  default_train_mode: 'defaultTrainMode',
  vlm_temperature: 'vlmTemperature',
  vlm_top_p: 'vlmTopP',
  vlm_stop: 'vlmStop',
}

const camelToSnake: Record<string, string> = Object.fromEntries(
  Object.entries(snakeToCamel).map(([k, v]) => [v, k])
)

function mapKeys<T extends Record<string, unknown>>(obj: T, keyMap: Record<string, string>): Partial<T> {
  const result: Partial<T> = {}
  for (const [k, v] of Object.entries(obj)) {
    const mapped = keyMap[k] ?? k
    ;(result as Record<string, unknown>)[mapped] = v
  }
  return result
}

const defaultSettings: UserSettings = {
  vlmProvider: 'openai',
  vlmBaseUrl: 'https://api.openai.com/v1',
  vlmApiKey: '',
  vlmApiFormat: 'openai',
  vlmModel: '',

  reasoningEnabled: true,
  reasoningProvider: 'deepseek',
  reasoningBaseUrl: 'https://api.deepseek.com/v1',
  reasoningApiKey: '',
  reasoningModel: 'deepseek-reasoner',

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

function authHeaders(): HeadersInit {
  const token = useAuthStore.getState().token
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function parseJsonSafe(res: Response): Promise<Record<string, unknown> | null> {
  try {
    return await res.json()
  } catch {
    return null
  }
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      settings: defaultSettings,
      loading: false,

      setSettings: (updates) =>
        set((state) => ({ settings: { ...state.settings, ...updates } })),

      loadSettings: async () => {
        set({ loading: true })
        try {
          const res = await fetch('/api/settings', {
            headers: authHeaders(),
          })
          const json = await parseJsonSafe(res)

          if (res.status === 401 || json?.code === 401 || json?.detail === '未登录') {
            useAuthStore.getState().logout()
            return
          }

          if (json?.code === 0 && json.data) {
            const mapped = mapKeys(json.data as Record<string, unknown>, snakeToCamel)
            set({ settings: { ...defaultSettings, ...mapped } as UserSettings })
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
        const toSave = mapKeys(merged as unknown as Record<string, unknown>, camelToSnake)

        try {
          const res = await fetch('/api/settings', {
            method: 'PUT',
            headers: {
              'Content-Type': 'application/json',
              ...authHeaders(),
            },
            body: JSON.stringify(toSave),
          })
          const json = await parseJsonSafe(res)

          if (res.status === 401 || json?.code === 401 || json?.detail === '未登录') {
            useAuthStore.getState().logout()
            return { success: false, message: '登录状态已失效，请重新登录后再保存' }
          }

          if (json?.code === 0) {
            set({ settings: merged })
            return { success: true, message: '设置已保存' }
          } else {
            return { success: false, message: String(json?.msg || '保存失败') }
          }
        } catch {
          return { success: false, message: '网络错误，请稍后重试' }
        }
      },
    }),
    {
      name: 'cv-settings-storage', // localStorage key
      partialize: (state) => ({ settings: state.settings }),
    }
  )
)

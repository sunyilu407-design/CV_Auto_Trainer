import { create } from 'zustand'

interface AuthUser {
  username: string
  role: string
}

interface AuthState {
  token: string | null
  user: AuthUser | null
  isLoggedIn: boolean
  login: (token: string, user: AuthUser) => void
  logout: () => void
  loadAuth: () => boolean
}

export const useAuthStore = create<AuthState>((set, get) => ({
  token: null,
  user: null,
  isLoggedIn: false,

  login: (token: string, user: AuthUser) => {
    set({ token, user, isLoggedIn: true })
  },

  logout: () => {
    set({ token: null, user: null, isLoggedIn: false })
  },

  loadAuth: () => {
    const { token } = get()
    return !!token
  },
}))

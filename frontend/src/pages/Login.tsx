import { useState } from 'react'
import { useAuthStore } from '../store/authStore'

export default function Login() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { login } = useAuthStore()

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })
      const data = await res.json()

      if (data.code !== 0) {
        setError(data.msg || '登录失败')
        setLoading(false)
        return
      }

      login(data.data.token, {
        username: data.data.username,
        role: data.data.role,
      })
    } catch {
      setError('网络错误，请稍后重试')
      setLoading(false)
    }
  }

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--gray-50)',
        padding: '24px',
      }}
    >
      <div
        className="fadeInScale"
        style={{
          width: '100%',
          maxWidth: 380,
        }}
      >
        {/* Logo */}
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <svg
            width="40"
            height="40"
            viewBox="0 0 40 40"
            fill="none"
            style={{ margin: '0 auto 12px' }}
          >
            <rect width="40" height="40" rx="10" fill="var(--gray-900)" />
            <path
              d="M10 28L20 12L30 28H10Z"
              fill="white"
              opacity="0.9"
            />
            <circle cx="20" cy="22" r="3" fill="var(--gray-900)" />
          </svg>
          <h1
            style={{
              fontSize: 20,
              fontWeight: 700,
              letterSpacing: '-0.4px',
              color: 'var(--gray-900)',
              margin: '0 0 4px',
            }}
          >
            CV Auto Trainer
          </h1>
          <p style={{ fontSize: 13, color: 'var(--gray-400)', margin: 0 }}>
            登录以继续
          </p>
        </div>

        {/* Card */}
        <div
          className="card"
          style={{
            padding: '28px 24px',
            boxShadow: 'var(--shadow-card)',
          }}
        >
          <form onSubmit={handleLogin}>
            <div className="form-group">
              <label className="form-label">用户名</label>
              <input
                className="input"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="输入管理员用户名"
                autoComplete="username"
                required
              />
            </div>

            <div className="form-group" style={{ marginTop: 16 }}>
              <label className="form-label">密码</label>
              <input
                className="input"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                autoComplete="current-password"
                required
              />
            </div>

            {error && (
              <div
                className="fadeIn"
                style={{
                  marginTop: 16,
                  padding: '10px 14px',
                  background: 'rgba(255,91,79,0.08)',
                  border: '1px solid rgba(255,91,79,0.2)',
                  borderRadius: 8,
                  fontSize: 13,
                  color: 'var(--ship-red)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                {error}
              </div>
            )}

            <button
              type="submit"
              className="btn btn-primary"
              style={{
                width: '100%',
                marginTop: 20,
                height: 40,
                fontSize: 14,
                fontWeight: 600,
              }}
              disabled={loading}
            >
              {loading ? (
                <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                  <svg className="spin" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 12a9 9 0 11-6.219-8.56" />
                  </svg>
                  登录中...
                </span>
              ) : (
                '登录'
              )}
            </button>
          </form>
        </div>

        <p
          style={{
            textAlign: 'center',
            fontSize: 12,
            color: 'var(--gray-300)',
            marginTop: 20,
          }}
        >
          管理员账户由后端环境变量初始化
        </p>
      </div>
    </div>
  )
}

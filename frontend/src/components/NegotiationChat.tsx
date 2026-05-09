import { useState, useRef, useEffect } from 'react'
import { useTaskStore, NegotiationMessage } from '../store/taskStore'
import { negotiateApi } from '../api/backend'

export default function NegotiationChat() {
  const {
    taskId,
    conversationId,
    negotiationMessages,
    negotiationConverged,
    setConversationId,
    addNegotiationMessage,
    setNegotiatedConfig,
    setNegotiationConverged,
    resetNegotiation,
  } = useTaskStore()

  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [negotiationMessages])

  // 首次进入时生成开场白（延迟一帧，等 restore 先执行）
  useEffect(() => {
    if (!taskId || loading) return
    const timer = setTimeout(() => {
      // 再次检查，避免和 restore 竞争
      const msgs = useTaskStore.getState().negotiationMessages
      if (msgs.length === 0) {
        handleSend('', true)
      }
    }, 100)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskId])

  async function handleSend(message?: string, isInitial = false) {
    const text = message ?? input.trim()
    if (!text && !isInitial) return
    if (!taskId) return

    setError(null)
    setLoading(true)

    // 用户消息
    if (text && !isInitial) {
      const userMsg: NegotiationMessage = {
        role: 'user',
        content: text,
        timestamp: Date.now(),
      }
      addNegotiationMessage(userMsg)
      setInput('')
    }

    try {
      const resp = await negotiateApi.chat({
        task_id: taskId,
        message: isInitial ? '__INIT__' : text,
        conversation_id: conversationId,
        include_initial: isInitial || negotiationMessages.length === 0,
      })

      // 更新 conversation ID
      if (resp.conversation_id && resp.conversation_id !== conversationId) {
        setConversationId(resp.conversation_id)
      }

      // AI 回复
      const aiMsg: NegotiationMessage = {
        role: 'assistant',
        content: resp.reply,
        timestamp: Date.now(),
        metadata: {
          should_preview: resp.should_preview,
          config_updated: resp.updated_config !== null,
          converged: resp.convergence.converged,
        },
      }
      addNegotiationMessage(aiMsg)

      // 更新配置
      if (resp.updated_config) {
        setNegotiatedConfig(resp.updated_config as any)
      }

      // 更新收敛状态
      setNegotiationConverged(resp.convergence.converged)
    } catch (err: any) {
      setError(err?.message || '对话请求失败，请重试')
    } finally {
      setLoading(false)
    }
  }

  async function handleReset() {
    if (!taskId) return
    if (!confirm('确定要重新开始对话吗？当前对话内容将被清除。')) return
    setLoading(true)
    try {
      await negotiateApi.reset(taskId)
      resetNegotiation()
      // 重新触发开场白
      setTimeout(() => handleSend('', true), 200)
    } catch (err: any) {
      setError(err?.message || '重置失败')
      setLoading(false)
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        background: '#fff',
        borderRadius: 10,
        border: '1px solid var(--gray-100)',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: '12px 16px',
          borderBottom: '1px solid var(--gray-100)',
          fontSize: 13,
          fontWeight: 600,
          color: 'var(--gray-700)',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}
      >
        <span style={{ fontSize: 16 }}>💬</span>
        需求确认助手
        {negotiationMessages.length > 0 && (
          <button
            onClick={handleReset}
            disabled={loading}
            style={{
              marginLeft: 'auto',
              padding: '2px 10px',
              borderRadius: 6,
              border: '1px solid var(--gray-200)',
              background: '#fff',
              color: 'var(--gray-500)',
              fontSize: 11,
              cursor: loading ? 'not-allowed' : 'pointer',
            }}
            title="清除当前对话，重新开始"
          >
            重新开始
          </button>
        )}
        {negotiationConverged && (
          <span
            style={{
              marginLeft: 'auto',
              fontSize: 11,
              padding: '2px 8px',
              borderRadius: 10,
              background: 'rgba(16,185,129,0.1)',
              color: '#059669',
              fontWeight: 600,
            }}
          >
            ✓ 需求已明确
          </span>
        )}
      </div>

      {/* Messages */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '16px',
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
        }}
      >
        {negotiationMessages.map((msg, idx) => (
          <MessageBubble key={idx} message={msg} />
        ))}

        {loading && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '8px 12px',
              color: 'var(--gray-400)',
              fontSize: 12,
            }}
          >
            <span className="loading-dots">AI 思考中</span>
            <span style={{ animation: 'pulse 1.5s ease-in-out infinite' }}>...</span>
          </div>
        )}

        {error && (
          <div
            style={{
              padding: '8px 12px',
              background: 'rgba(239,68,68,0.05)',
              borderRadius: 8,
              color: '#dc2626',
              fontSize: 12,
            }}
          >
            {error}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div
        style={{
          padding: '12px 16px',
          borderTop: '1px solid var(--gray-100)',
          display: 'flex',
          gap: 8,
          alignItems: 'flex-end',
        }}
      >
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="描述您的需求，或回复 AI 的问题..."
          disabled={loading}
          rows={1}
          style={{
            flex: 1,
            resize: 'none',
            border: '1px solid var(--gray-200)',
            borderRadius: 8,
            padding: '8px 12px',
            fontSize: 13,
            lineHeight: '1.4',
            outline: 'none',
            minHeight: 36,
            maxHeight: 100,
            overflow: 'auto',
          }}
        />
        <button
          onClick={() => handleSend()}
          disabled={loading || !input.trim()}
          style={{
            padding: '8px 16px',
            borderRadius: 8,
            border: 'none',
            background: loading || !input.trim() ? 'var(--gray-200)' : '#0a72ef',
            color: loading || !input.trim() ? 'var(--gray-400)' : '#fff',
            fontSize: 13,
            fontWeight: 600,
            cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
            whiteSpace: 'nowrap',
          }}
        >
          发送
        </button>
      </div>
    </div>
  )
}

function MessageBubble({ message }: { message: NegotiationMessage }) {
  const isUser = message.role === 'user'

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
      }}
    >
      <div
        style={{
          maxWidth: '85%',
          padding: '10px 14px',
          borderRadius: isUser ? '12px 12px 2px 12px' : '12px 12px 12px 2px',
          background: isUser ? '#0a72ef' : 'var(--gray-50)',
          color: isUser ? '#fff' : 'var(--gray-800)',
          fontSize: 13,
          lineHeight: '1.5',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {message.content}
        {message.metadata?.config_updated && (
          <div
            style={{
              marginTop: 6,
              padding: '4px 8px',
              background: isUser ? 'rgba(255,255,255,0.15)' : 'rgba(10,114,239,0.08)',
              borderRadius: 4,
              fontSize: 11,
              color: isUser ? 'rgba(255,255,255,0.8)' : '#0a72ef',
            }}
          >
            ✦ 配置已更新
          </div>
        )}
      </div>
    </div>
  )
}

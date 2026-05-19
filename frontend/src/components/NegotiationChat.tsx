import { useState, useRef, useEffect, useCallback } from 'react'
import { useTaskStore, NegotiationMessage } from '../store/taskStore'
import { negotiateApi } from '../api/backend'

interface NegotiationChatProps {
  suppressInit?: boolean
}

// Lightweight inline markdown renderer (no external deps)
function renderMarkdown(text: string): React.ReactNode[] {
  const lines = text.split('\n')
  const elements: React.ReactNode[] = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]

    // Heading
    if (line.startsWith('### ')) {
      elements.push(<h4 key={i} style={{ margin: '8px 0 4px', fontSize: 14, fontWeight: 700, color: 'var(--gray-800)' }}>{renderInline(line.slice(4))}</h4>)
      i++; continue
    }
    // Sub-heading
    if (line.startsWith('## ')) {
      elements.push(<h3 key={i} style={{ margin: '10px 0 6px', fontSize: 15, fontWeight: 700, color: 'var(--gray-800)' }}>{renderInline(line.slice(3))}</h3>)
      i++; continue
    }
    if (line.startsWith('# ')) {
      elements.push(<h2 key={i} style={{ margin: '12px 0 6px', fontSize: 16, fontWeight: 700, color: 'var(--gray-800)' }}>{renderInline(line.slice(2))}</h2>)
      i++; continue
    }

    // Bullet list
    if (/^[-\*] /.test(line)) {
      elements.push(
        <div key={i} style={{ display: 'flex', gap: 6, marginBottom: 4, paddingLeft: 4 }}>
          <span style={{ color: '#0a72ef', marginTop: 2, flexShrink: 0 }}>•</span>
          <span style={{ flex: 1 }}>{renderInline(line.slice(2))}</span>
        </div>
      )
      i++; continue
    }

    // Numbered list
    if (/^\d+\. /.test(line)) {
      const num = line.match(/^(\d+)\. /)?.[1] ?? ''
      elements.push(
        <div key={i} style={{ display: 'flex', gap: 6, marginBottom: 4, paddingLeft: 4 }}>
          <span style={{ color: '#0a72ef', fontWeight: 700, minWidth: 16, flexShrink: 0 }}>{num}.</span>
          <span style={{ flex: 1 }}>{renderInline(line.replace(/^\d+\. /, ''))}</span>
        </div>
      )
      i++; continue
    }

    // Blockquote
    if (line.startsWith('> ')) {
      elements.push(
        <div key={i} style={{
          margin: '4px 0', padding: '6px 12px',
          borderLeft: '3px solid #0a72ef',
          background: 'rgba(10,114,239,0.05)',
          borderRadius: '0 6px 6px 0',
          color: 'var(--gray-600)', fontSize: 13,
        }}>
          {renderInline(line.slice(2))}
        </div>
      )
      i++; continue
    }

    // Code block marker
    if (line.startsWith('```')) {
      const codeLines: string[] = []
      i++
      while (i < lines.length && !lines[i].startsWith('```')) {
        codeLines.push(lines[i])
        i++
      }
      elements.push(
        <pre key={i} style={{
          margin: '8px 0', padding: '10px 14px',
          background: '#1e1e1e', borderRadius: 8,
          color: '#d4d4d4', fontSize: 12, fontFamily: 'monospace',
          overflowX: 'auto', whiteSpace: 'pre',
        }}>
          {codeLines.join('\n')}
        </pre>
      )
      i++; continue
    }

    // Empty line
    if (!line.trim()) {
      elements.push(<div key={i} style={{ height: 4 }} />)
      i++; continue
    }

    // Regular paragraph
    elements.push(
      <p key={i} style={{ margin: '2px 0', lineHeight: 1.6 }}>{renderInline(line)}</p>
    )
    i++
  }

  return elements
}

function renderInline(text: string): React.ReactNode {
  // Process inline formatting: **bold**, `code`, _italic_
  const parts: React.ReactNode[] = []
  let remaining = text
  let key = 0

  while (remaining) {
    // Bold
    const boldMatch = remaining.match(/\*\*(.+?)\*\*/)
    if (boldMatch && boldMatch.index !== undefined) {
      if (boldMatch.index > 0) parts.push(<span key={key++}>{remaining.slice(0, boldMatch.index)}</span>)
      parts.push(<strong key={key++} style={{ fontWeight: 700, color: 'var(--gray-900)' }}>{boldMatch[1]}</strong>)
      remaining = remaining.slice(boldMatch.index + boldMatch[0].length)
      continue
    }
    // Inline code
    const codeMatch = remaining.match(/`([^`]+)`/)
    if (codeMatch && codeMatch.index !== undefined) {
      if (codeMatch.index > 0) parts.push(<span key={key++}>{remaining.slice(0, codeMatch.index)}</span>)
      parts.push(<code key={key++} style={{
        padding: '1px 5px', background: 'rgba(0,0,0,0.06)',
        borderRadius: 4, fontSize: 12, fontFamily: 'monospace',
      }}>{codeMatch[1]}</code>)
      remaining = remaining.slice(codeMatch.index + codeMatch[0].length)
      continue
    }
    // Italic
    const italicMatch = remaining.match(/_([^_]+)_/)
    if (italicMatch && italicMatch.index !== undefined) {
      if (italicMatch.index > 0) parts.push(<span key={key++}>{remaining.slice(0, italicMatch.index)}</span>)
      parts.push(<em key={key++}>{italicMatch[1]}</em>)
      remaining = remaining.slice(italicMatch.index + italicMatch[0].length)
      continue
    }
    // No more matches
    parts.push(<span key={key++}>{remaining}</span>)
    break
  }

  return <>{parts}</>
}

interface StreamBubble {
  id: string
  content: string
  isUser: boolean
  timestamp: number
  configUpdated?: boolean
  converged?: boolean
  isStreaming?: boolean
}

export default function NegotiationChat({ suppressInit = false }: NegotiationChatProps) {
  const {
    taskId,
    conversationId,
    negotiationConverged,
    setConversationId,
    setNegotiatedConfig,
    setNegotiationConverged,
    resetNegotiation,
  } = useTaskStore()

  const [streams, setStreams] = useState<StreamBubble[]>([]) // 当前流式气泡列表
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const abortRef = useRef<(() => void) | null>(null)
  const streamIdRef = useRef(0)

  // 滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [streams])

  // 自动聚焦输入框
  useEffect(() => {
    if (!isLoading) inputRef.current?.focus()
  }, [isLoading])

  const handleStreamStart = useCallback((text: string, isUser: boolean) => {
    const id = `stream-${Date.now()}-${streamIdRef.current++}`
    setStreams(prev => [...prev, {
      id,
      content: text,
      isUser,
      timestamp: Date.now(),
      isStreaming: !isUser,
    }])
    return id
  }, [])

  const handleStreamAppend = useCallback((id: string, chunk: string) => {
    setStreams(prev => prev.map(s =>
      s.id === id ? { ...s, content: s.content + chunk } : s
    ))
  }, [])

  const handleStreamDone = useCallback((id: string, metadata?: { configUpdated?: boolean; converged?: boolean }) => {
    setStreams(prev => prev.map(s =>
      s.id === id ? { ...s, isStreaming: false, ...metadata } : s
    ))
  }, [])

  // Create empty AI bubble and return its id
  const startAIStream = useCallback(() => {
    const id = `stream-${Date.now()}-${streamIdRef.current++}`
    setStreams(prev => [...prev, {
      id,
      content: '',
      isUser: false,
      timestamp: Date.now(),
      isStreaming: true,
    }])
    return id
  }, [])

  // Append text to an existing AI bubble by id
  const appendToStream = useCallback((id: string, chunk: string) => {
    setStreams(prev => prev.map(s =>
      s.id === id ? { ...s, content: s.content + chunk } : s
    ))
  }, [])

  // Mark stream as done with optional metadata
  const finishStream = useCallback((id: string, meta?: { configUpdated?: boolean; converged?: boolean }) => {
    setStreams(prev => prev.map(s =>
      s.id === id ? { ...s, isStreaming: false, ...meta } : s
    ))
  }, [])

  const startStream = useCallback((message: string, isInitial = false) => {
    if (!taskId) return

    // Cancel previous stream
    abortRef.current?.()
    setError(null)
    setIsLoading(true)

    // Add user message bubble immediately
    handleStreamStart(message, true)

    // Add empty AI bubble immediately (before any chunk arrives)
    const aiBubbleId = startAIStream()

    const cleanup = negotiateApi.streamChat({
      task_id: taskId,
      message: isInitial ? '__INIT__' : message,
      conversation_id: conversationId,
      include_initial: isInitial || streams.length === 0,
    }, {
      onChunk: (chunk) => {
        appendToStream(aiBubbleId, chunk)
      },
      onDone: (data) => {
        // Update conversation ID
        if (data.conversation_id && data.conversation_id !== conversationId) {
          setConversationId(data.conversation_id)
        }
        finishStream(aiBubbleId, {
          configUpdated: data.updated_config != null,
          converged: data.convergence?.converged,
        })
        if (data.updated_config) {
          setNegotiatedConfig(data.updated_config as any)
        }
        setNegotiationConverged(data.convergence?.converged ?? false)
        setIsLoading(false)
        abortRef.current = null
      },
      onError: (err) => {
        setError(err.message || '对话请求失败，请重试')
        finishStream(aiBubbleId)
        setIsLoading(false)
        abortRef.current = null
      },
    })

    abortRef.current = cleanup
  }, [taskId, conversationId, streams.length, handleStreamStart, startAIStream, appendToStream, finishStream, setConversationId, setNegotiatedConfig, setNegotiationConverged])

  async function handleSend(message?: string, isInitial = false) {
    const text = message ?? input.trim()
    if (!text && !isInitial) return
    if (!taskId) return

    setInput('')
    startStream(text, isInitial)
  }

  async function handleReset() {
    if (!taskId) return
    if (!confirm('确定要重新开始对话吗？当前对话内容将被清除。')) return
    abortRef.current?.()
    setStreams([])
    setIsLoading(false)
    setError(null)
    try {
      await negotiateApi.reset(taskId)
      resetNegotiation()
      setTimeout(() => startStream('', true), 200)
    } catch (err: any) {
      setError(err?.message || '重置失败')
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const hasMessages = streams.length > 0

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
        <div style={{
          width: 28, height: 28, borderRadius: '50%',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: '#fff', fontSize: 12, fontWeight: 700, flexShrink: 0,
        }}>
          AI
        </div>
        <span>需求确认助手</span>

        {isLoading && (
          <div style={{
            marginLeft: 8, display: 'flex', alignItems: 'center', gap: 4,
            fontSize: 11, color: '#0a72ef', fontWeight: 500,
          }}>
            <span className="typing-dot" />
            <span className="typing-dot" style={{ animationDelay: '0.2s' }} />
            <span className="typing-dot" style={{ animationDelay: '0.4s' }} />
            <span>生成中</span>
          </div>
        )}

        {hasMessages && (
          <button
            onClick={handleReset}
            disabled={isLoading}
            style={{
              marginLeft: 'auto',
              padding: '2px 10px',
              borderRadius: 6,
              border: '1px solid var(--gray-200)',
              background: '#fff',
              color: 'var(--gray-500)',
              fontSize: 11,
              cursor: isLoading ? 'not-allowed' : 'pointer',
              opacity: isLoading ? 0.5 : 1,
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
        {/* Welcome state */}
        {!hasMessages && !isLoading && (
          <div style={{
            flex: 1, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            gap: 12, padding: '32px 0',
            color: 'var(--gray-400)',
          }}>
            <div style={{
              width: 56, height: 56, borderRadius: '50%',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 22, color: '#fff', boxShadow: '0 4px 12px rgba(102,126,234,0.3)',
            }}>
              🤖
            </div>
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: 14, fontWeight: 600, color: 'var(--gray-600)', margin: '0 0 4px' }}>
                AI 需求确认助手
              </p>
              <p style={{ fontSize: 12, color: 'var(--gray-400)', margin: 0 }}>
                正在连接…
              </p>
            </div>
          </div>
        )}

        {streams.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {/* Loading indicator */}
        {isLoading && (
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
            <div style={{
              width: 28, height: 28, borderRadius: '50%', flexShrink: 0,
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: '#fff', fontSize: 10, fontWeight: 700,
            }}>
              AI
            </div>
            <div style={{
              padding: '10px 14px',
              borderRadius: '12px 12px 12px 2px',
              background: 'var(--gray-50)',
              border: '1px solid var(--gray-100)',
              display: 'flex',
              gap: 4,
              alignItems: 'center',
            }}>
              <span className="thinking-dot" />
              <span className="thinking-dot" style={{ animationDelay: '0.15s' }} />
              <span className="thinking-dot" style={{ animationDelay: '0.3s' }} />
            </div>
          </div>
        )}

        {error && (
          <div style={{
            padding: '10px 14px',
            background: 'rgba(239,68,68,0.06)',
            borderRadius: 8,
            color: '#dc2626',
            fontSize: 12,
            border: '1px solid rgba(239,68,68,0.15)',
          }}>
            <strong>错误：</strong>{error}
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
          placeholder="描述您的需求，或回复 AI 的问题…"
          disabled={isLoading}
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
            transition: 'border-color 0.15s',
          }}
          onFocus={(e) => (e.target.style.borderColor = '#0a72ef')}
          onBlur={(e) => (e.target.style.borderColor = 'var(--gray-200)')}
        />
        <button
          onClick={() => handleSend()}
          disabled={isLoading || !input.trim()}
          style={{
            padding: '8px 16px',
            borderRadius: 8,
            border: 'none',
            background: isLoading || !input.trim() ? 'var(--gray-200)' : '#0a72ef',
            color: isLoading || !input.trim() ? 'var(--gray-400)' : '#fff',
            fontSize: 13,
            fontWeight: 600,
            cursor: isLoading || !input.trim() ? 'not-allowed' : 'pointer',
            whiteSpace: 'nowrap',
            transition: 'background 0.15s',
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}
        >
          {isLoading ? (
            <>
              <span className="mini-spinner" />
              生成中
            </>
          ) : '发送'}
        </button>
      </div>
    </div>
  )
}

function MessageBubble({ message }: { message: StreamBubble }) {
  const isUser = message.isUser

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        alignItems: 'flex-end',
        gap: 8,
      }}
    >
      {!isUser && (
        <div style={{
          width: 28, height: 28, borderRadius: '50%', flexShrink: 0,
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: '#fff', fontSize: 10, fontWeight: 700,
        }}>
          AI
        </div>
      )}

      <div
        style={{
          maxWidth: '82%',
          padding: '10px 14px',
          borderRadius: isUser
            ? '16px 16px 4px 16px'
            : '16px 16px 16px 4px',
          background: isUser
            ? 'linear-gradient(135deg, #0a72ef 0%, #0057d9 100%)'
            : '#fff',
          border: isUser ? 'none' : '1px solid var(--gray-100)',
          boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
          color: isUser ? '#fff' : 'var(--gray-800)',
          fontSize: 13,
          lineHeight: '1.6',
          wordBreak: 'break-word',
          position: 'relative',
        }}
      >
        {/* Streaming cursor */}
        {message.isStreaming && (
          <span style={{
            display: 'inline-block',
            width: 2, height: 14,
            background: isUser ? '#fff' : '#0a72ef',
            marginLeft: 2,
            verticalAlign: 'text-bottom',
            animation: 'blink 0.8s ease-in-out infinite',
          }} />
        )}

        {renderMarkdown(message.content)}

        {/* Config badge */}
        {message.configUpdated && !message.isStreaming && (
          <div style={{
            marginTop: 8,
            padding: '4px 8px',
            background: isUser ? 'rgba(255,255,255,0.15)' : 'rgba(10,114,239,0.08)',
            borderRadius: 4,
            fontSize: 11,
            color: isUser ? 'rgba(255,255,255,0.85)' : '#0a72ef',
            fontWeight: 500,
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}>
            <span>✦</span>
            <span>检测配置已更新</span>
          </div>
        )}

        {/* Time */}
        <div style={{
          marginTop: 4,
          fontSize: 10,
          color: isUser ? 'rgba(255,255,255,0.5)' : 'var(--gray-300)',
          textAlign: 'right',
        }}>
          {new Date(message.timestamp).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}
        </div>
      </div>

      {isUser && (
        <div style={{
          width: 28, height: 28, borderRadius: '50%', flexShrink: 0,
          background: 'linear-gradient(135deg, #11998e 0%, #0a72ef 100%)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: '#fff', fontSize: 10, fontWeight: 700,
        }}>
          U
        </div>
      )}
    </div>
  )
}

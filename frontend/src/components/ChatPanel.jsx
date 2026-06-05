import { useState, useEffect, useRef } from 'react'
import { streamChat } from '../lib/api.js'
import MessageBubble from './MessageBubble.jsx'

const STARTER_QUESTIONS = [
  'Explain the overall architecture in 3 sentences',
  'Where does request routing happen?',
  'What are the main data models?',
  'Trace the authentication flow',
  'What would break if I deleted the main entry point?',
]

export default function ChatPanel({ repo, summary, stats, initialQuestion }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const threadRef = useRef(null)
  const textareaRef = useRef(null)
  const abortRef = useRef(null)

  useEffect(() => {
    if (initialQuestion) {
      sendMessage(initialQuestion)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (threadRef.current) {
      threadRef.current.scrollTop = threadRef.current.scrollHeight
    }
  }, [messages])

  function sendMessage(text) {
    const trimmed = (text ?? input).trim()
    if (!trimmed || loading) return

    setInput('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }

    const snapshot = messages.filter((m) => !m.streaming)
    const history = snapshot.map((m) => ({ role: m.role, content: m.content }))

    const userMsg = { role: 'user', content: trimmed }
    const assistantPlaceholder = { role: 'assistant', content: '', sources: [], streaming: true }

    setMessages([...snapshot, userMsg, assistantPlaceholder])
    setLoading(true)

    if (abortRef.current) abortRef.current()

    abortRef.current = streamChat(repo, trimmed, history, (event) => {
      if (event.type === 'text') {
        setMessages((prev) => {
          const updated = [...prev]
          const last = updated[updated.length - 1]
          if (last?.role === 'assistant') {
            updated[updated.length - 1] = { ...last, content: last.content + event.content }
          }
          return updated
        })
      } else if (event.type === 'sources') {
        setMessages((prev) => {
          const updated = [...prev]
          const last = updated[updated.length - 1]
          if (last?.role === 'assistant') {
            updated[updated.length - 1] = { ...last, sources: event.sources }
          }
          return updated
        })
      } else if (event.type === 'done' || event.type === 'error') {
        setMessages((prev) => {
          const updated = [...prev]
          const last = updated[updated.length - 1]
          if (last?.role === 'assistant') {
            updated[updated.length - 1] = {
              ...last,
              streaming: false,
              error: event.type === 'error' ? event.message : null,
            }
          }
          return updated
        })
        setLoading(false)
      }
    })
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  function handleTextareaChange(e) {
    setInput(e.target.value)
    const el = e.target
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }

  const repoName = repo?.replace('https://github.com/', '') ?? repo
  const description = summary?.description ?? null
  const bestFirstQuestion = summary?.best_first_question ?? null
  const languages = Array.isArray(stats?.languages) ? stats.languages.slice(0, 4) : []

  return (
    <div style={s.root}>
      {/* ── Sidebar ────────────────────────────────────────── */}
      <aside style={s.sidebar}>
        <div style={s.sideInner}>

          {/* Repo link */}
          <section style={s.sideSection}>
            <p style={s.sideLabel}>Repository</p>
            <a href={repo} target="_blank" rel="noopener noreferrer" style={s.repoLink}>
              {repoName}
            </a>
          </section>

          {/* Stats */}
          {stats && (
            <section style={s.sideSection}>
              <p style={s.sideLabel}>Index stats</p>
              <div style={s.statsList}>
                {stats.files != null && (
                  <div style={s.statRow}>
                    <span style={s.statVal}>{stats.files.toLocaleString()}</span>
                    <span style={s.statLab}> files</span>
                  </div>
                )}
                {stats.chunks != null && (
                  <div style={s.statRow}>
                    <span style={s.statVal}>{stats.chunks.toLocaleString()}</span>
                    <span style={s.statLab}> chunks</span>
                  </div>
                )}
                {languages.length > 0 && (
                  <p style={s.langText}>{languages.join(' · ')}</p>
                )}
              </div>
            </section>
          )}

          {/* Description */}
          {description && (
            <section style={s.sideSection}>
              <p style={s.descText}>{description}</p>
            </section>
          )}

          <div style={s.sideDivider} />

          {/* Suggested questions */}
          <section style={s.questionsSection}>
            <p style={s.sideLabel}>Suggested questions</p>
            <div style={s.questionList}>
              {bestFirstQuestion && (
                <button
                  style={s.bestQuestion}
                  onClick={() => sendMessage(bestFirstQuestion)}
                  disabled={loading}
                >
                  <span style={s.bestIcon}>◈</span>
                  <span>{bestFirstQuestion}</span>
                </button>
              )}
              {STARTER_QUESTIONS.map((q) => (
                <button
                  key={q}
                  style={loading ? { ...s.questionBtn, opacity: 0.45, cursor: 'not-allowed' } : s.questionBtn}
                  onClick={() => sendMessage(q)}
                  disabled={loading}
                >
                  {q}
                </button>
              ))}
            </div>
          </section>

        </div>
      </aside>

      {/* ── Main chat area ─────────────────────────────────── */}
      <div style={s.main}>

        {/* Thread */}
        <div ref={threadRef} style={s.thread}>
          {messages.length === 0 ? (
            <div style={s.emptyState}>
              <span style={s.emptyIcon}>◈</span>
              <p style={s.emptyTitle}>Ask anything about the codebase</p>
              <p style={s.emptySub}>
                Choose a suggested question on the left, or type your own below.
              </p>
            </div>
          ) : (
            <div style={s.messageList}>
              {messages.map((msg, i) => (
                <MessageBubble key={i} msg={msg} />
              ))}
            </div>
          )}
        </div>

        {/* Input area */}
        <div style={s.inputArea}>
          <div style={s.inputRow}>
            <textarea
              ref={textareaRef}
              style={s.textarea}
              value={input}
              onChange={handleTextareaChange}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
              rows={1}
              disabled={loading}
            />
            <button
              style={
                loading || !input.trim()
                  ? { ...s.sendBtn, opacity: 0.35, cursor: 'not-allowed' }
                  : s.sendBtn
              }
              onClick={() => sendMessage()}
              disabled={loading || !input.trim()}
            >
              ↑
            </button>
          </div>
          <p style={s.inputHint}>Enter to send · Shift+Enter for newline</p>
        </div>

      </div>
    </div>
  )
}

/* ─── Styles ────────────────────────────────────────────────────────────────── */
const s = {
  // Layout
  root: {
    flex: 1,
    display: 'flex',
    overflow: 'hidden',
    height: '100%',
  },

  // Sidebar
  sidebar: {
    width: 240,
    flexShrink: 0,
    borderRight: '1px solid var(--color-border)',
    overflowY: 'auto',
    background: 'var(--color-surface)',
  },
  sideInner: {
    display: 'flex',
    flexDirection: 'column',
    padding: '1.25rem 1rem',
    gap: '1.25rem',
    minHeight: '100%',
  },
  sideSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
  },
  sideLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.625rem',
    fontWeight: 600,
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    color: 'var(--color-text-dim)',
  },
  repoLink: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.75rem',
    color: 'var(--color-accent)',
    textDecoration: 'none',
    wordBreak: 'break-all',
    lineHeight: 1.5,
  },
  statsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.25rem',
  },
  statRow: {
    display: 'flex',
    alignItems: 'baseline',
    gap: '0.25rem',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.75rem',
  },
  statVal: {
    color: 'var(--color-text)',
    fontWeight: 600,
  },
  statLab: {
    color: 'var(--color-text-muted)',
  },
  langText: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.6875rem',
    color: 'var(--color-text-muted)',
    marginTop: '0.125rem',
  },
  descText: {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.8125rem',
    color: 'var(--color-text-muted)',
    lineHeight: 1.55,
  },
  sideDivider: {
    height: 1,
    background: 'var(--color-border-subtle)',
    margin: '0 -1rem',
  },
  questionsSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
    flex: 1,
  },
  questionList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.375rem',
  },
  bestQuestion: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '0.5rem',
    background: 'rgba(124, 106, 247, 0.1)',
    border: '1px solid rgba(124, 106, 247, 0.3)',
    borderRadius: 'var(--radius-md)',
    padding: '0.5rem 0.625rem',
    fontFamily: 'var(--font-sans)',
    fontSize: '0.75rem',
    color: 'var(--color-text)',
    lineHeight: 1.45,
    cursor: 'pointer',
    textAlign: 'left',
    transition: 'background 0.15s',
  },
  bestIcon: {
    color: 'var(--color-accent)',
    fontSize: '0.75rem',
    flexShrink: 0,
    marginTop: '0.0625rem',
  },
  questionBtn: {
    background: 'transparent',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-md)',
    padding: '0.4375rem 0.625rem',
    fontFamily: 'var(--font-sans)',
    fontSize: '0.75rem',
    color: 'var(--color-text-muted)',
    lineHeight: 1.45,
    cursor: 'pointer',
    textAlign: 'left',
    transition: 'border-color 0.15s, color 0.15s',
  },

  // Main area
  main: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  thread: {
    flex: 1,
    overflowY: 'auto',
    padding: '1.5rem',
  },

  // Empty state
  emptyState: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '0.75rem',
    height: '100%',
    minHeight: 240,
    color: 'var(--color-text-muted)',
    textAlign: 'center',
  },
  emptyIcon: {
    fontSize: '2rem',
    color: 'var(--color-accent)',
    opacity: 0.4,
    lineHeight: 1,
  },
  emptyTitle: {
    fontFamily: 'var(--font-sans)',
    fontWeight: 600,
    fontSize: '1rem',
    color: 'var(--color-text)',
  },
  emptySub: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.8125rem',
    color: 'var(--color-text-muted)',
    maxWidth: 340,
    lineHeight: 1.6,
  },

  // Message list
  messageList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem',
    maxWidth: 760,
    margin: '0 auto',
    width: '100%',
  },

  // Input area
  inputArea: {
    borderTop: '1px solid var(--color-border)',
    padding: '0.875rem 1.5rem 1rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.375rem',
    background: 'var(--color-surface)',
  },
  inputRow: {
    display: 'flex',
    gap: '0.625rem',
    alignItems: 'flex-end',
  },
  textarea: {
    flex: 1,
    fontFamily: 'var(--font-sans)',
    fontSize: '0.9375rem',
    background: 'var(--color-surface-2)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-lg)',
    padding: '0.625rem 0.875rem',
    color: 'var(--color-text)',
    resize: 'none',
    outline: 'none',
    lineHeight: 1.5,
    minWidth: 0,
    maxHeight: 160,
    overflowY: 'auto',
    transition: 'border-color 0.15s',
  },
  sendBtn: {
    flexShrink: 0,
    width: 38,
    height: 38,
    borderRadius: 'var(--radius-md)',
    background: 'var(--color-accent)',
    color: '#fff',
    border: 'none',
    cursor: 'pointer',
    fontSize: '1.125rem',
    fontWeight: 700,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'opacity 0.15s',
  },
  inputHint: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.625rem',
    color: 'var(--color-text-dim)',
    letterSpacing: '0.02em',
  },
}

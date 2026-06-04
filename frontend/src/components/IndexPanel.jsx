import { useState, useEffect, useRef } from 'react'
import { indexRepo } from '../lib/api.js'

const DEMO_REPOS = [
  'https://github.com/shrxyo/embedding_model',
  'https://github.com/shrxyo/portfolio',
]

const STATUS_MESSAGES = [
  'Cloning repository…',
  'Scanning files…',
  'Parsing code structure…',
  'Building embeddings…',
  'Indexing vector store…',
  'Generating summary…',
  'Almost there…',
]

const FEATURE_CARDS = [
  {
    icon: '⬡',
    title: 'RAG Pipeline',
    desc: 'Hybrid BM25 + vector retrieval with cross-encoder reranking for precise code search.',
  },
  {
    icon: '◈',
    title: 'Agent Layer',
    desc: 'Tool-calling agent that searches, reads files, and traces reasoning step by step.',
  },
  {
    icon: '▷',
    title: 'Streaming',
    desc: 'SSE token streaming with live source citations linked to exact file:line spans.',
  },
]

export default function IndexPanel({ onSuccess }) {
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [msgIndex, setMsgIndex] = useState(0)
  const [msgKey, setMsgKey] = useState(0)
  const intervalRef = useRef(null)
  const fillRef = useRef(null)

  useEffect(() => {
    return () => clearInterval(intervalRef.current)
  }, [])

  function startMessageCycle() {
    setMsgIndex(0)
    setMsgKey((k) => k + 1)
    clearInterval(intervalRef.current)
    intervalRef.current = setInterval(() => {
      setMsgIndex((i) => {
        const next = i + 1
        if (next < STATUS_MESSAGES.length) {
          setMsgKey((k) => k + 1)
          return next
        }
        return i
      })
    }, 1800)
  }

  async function submit(repoUrl) {
    const trimmed = (repoUrl ?? url).trim()
    if (!trimmed) return

    setError(null)
    setLoading(true)
    startMessageCycle()

    // Restart CSS progress animation by remounting the element
    if (fillRef.current) {
      fillRef.current.classList.remove('progress-fill')
      void fillRef.current.offsetWidth
      fillRef.current.classList.add('progress-fill')
    }

    try {
      const data = await indexRepo(trimmed)
      clearInterval(intervalRef.current)
      onSuccess({
        repo: data.repo_url,
        stats: {
          files: data.files_indexed,
          chunks: data.chunks_created,
          languages: data.languages,
        },
        summary: data.summary,
      })
    } catch (err) {
      clearInterval(intervalRef.current)
      setLoading(false)
      setError(err.message)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter') submit()
  }

  function handleDemoChip(demoUrl) {
    setUrl(demoUrl)
    submit(demoUrl)
  }

  return (
    <div style={s.root}>
      <div style={s.inner}>
        {/* Brand */}
        <div style={s.brand}>
          <span style={s.brandDiamond}>◈</span>
          <span style={s.brandName}>Codebase-QA-Agent</span>
        </div>
        <p style={s.tagline}>Drop in a GitHub URL. Ask anything about the codebase.</p>

        {/* Input card */}
        <div style={s.card}>
          {!loading ? (
            <>
              <div style={s.inputRow}>
                <input
                  style={s.input}
                  type="url"
                  placeholder="https://github.com/owner/repo"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  onKeyDown={handleKeyDown}
                  autoFocus
                  spellCheck={false}
                />
                <button
                  style={url.trim() ? s.btnActive : s.btn}
                  onClick={() => submit()}
                  disabled={!url.trim()}
                >
                  Analyze →
                </button>
              </div>

              {/* Demo chips */}
              <div style={s.chipRow}>
                <span style={s.chipLabel}>Try:</span>
                {DEMO_REPOS.map((r) => (
                  <button key={r} style={s.chip} onClick={() => handleDemoChip(r)}>
                    {r.replace('https://github.com/', '')}
                  </button>
                ))}
              </div>
            </>
          ) : (
            /* Loading state */
            <div style={s.loadingWrap}>
              <div style={s.repoLine}>
                <span style={s.spinnerWrap}>
                  <span className="spin" style={s.spinner} />
                </span>
                <span style={s.loadingUrl}>{url.trim()}</span>
              </div>

              {/* Progress bar */}
              <div style={s.barTrack}>
                <div ref={fillRef} className="progress-fill progress-shimmer" style={s.barFill} />
              </div>

              {/* Status message */}
              <p key={msgKey} className="fade-up" style={s.statusMsg}>
                {STATUS_MESSAGES[msgIndex]}
              </p>
            </div>
          )}
        </div>

        {/* Error box */}
        {error && (
          <div style={s.errorBox}>
            <span style={s.errorIcon}>✕</span>
            <span style={s.errorText}>{error}</span>
          </div>
        )}

        {/* Feature grid */}
        {!loading && (
          <div style={s.featureGrid}>
            {FEATURE_CARDS.map((card) => (
              <div key={card.title} style={s.featureCard}>
                <span style={s.featureIcon}>{card.icon}</span>
                <p style={s.featureTitle}>{card.title}</p>
                <p style={s.featureDesc}>{card.desc}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

const s = {
  root: {
    flex: 1,
    overflowY: 'auto',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '2rem 1rem',
  },
  inner: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '1.5rem',
    width: '100%',
    maxWidth: 680,
  },

  // Brand
  brand: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  brandDiamond: {
    fontSize: '1.75rem',
    color: 'var(--color-accent)',
    lineHeight: 1,
  },
  brandName: {
    fontFamily: 'var(--font-sans)',
    fontWeight: 800,
    fontSize: '2rem',
    letterSpacing: '-0.02em',
    color: 'var(--color-text)',
  },
  tagline: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.875rem',
    color: 'var(--color-text-muted)',
    textAlign: 'center',
    marginTop: '-0.5rem',
  },

  // Input card
  card: {
    width: '100%',
    background: 'var(--color-surface)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-xl)',
    padding: '1.25rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.875rem',
  },
  inputRow: {
    display: 'flex',
    gap: '0.625rem',
  },
  input: {
    flex: 1,
    fontFamily: 'var(--font-mono)',
    fontSize: '0.875rem',
    background: 'var(--color-surface-2)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-md)',
    padding: '0.625rem 0.875rem',
    color: 'var(--color-text)',
    outline: 'none',
    transition: 'border-color 0.15s',
    minWidth: 0,
  },
  btn: {
    fontFamily: 'var(--font-sans)',
    fontWeight: 600,
    fontSize: '0.875rem',
    padding: '0.625rem 1.25rem',
    borderRadius: 'var(--radius-md)',
    background: 'var(--color-surface-2)',
    color: 'var(--color-text-dim)',
    border: '1px solid var(--color-border)',
    cursor: 'not-allowed',
    flexShrink: 0,
    transition: 'all 0.15s',
  },
  btnActive: {
    fontFamily: 'var(--font-sans)',
    fontWeight: 600,
    fontSize: '0.875rem',
    padding: '0.625rem 1.25rem',
    borderRadius: 'var(--radius-md)',
    background: 'var(--color-accent)',
    color: '#fff',
    border: '1px solid transparent',
    cursor: 'pointer',
    flexShrink: 0,
    transition: 'all 0.15s',
  },

  // Demo chips
  chipRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    flexWrap: 'wrap',
  },
  chipLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.75rem',
    color: 'var(--color-text-dim)',
  },
  chip: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.75rem',
    color: 'var(--color-accent)',
    background: 'var(--color-accent-glow)',
    border: '1px solid rgba(124,106,247,0.25)',
    borderRadius: '999px',
    padding: '0.25rem 0.75rem',
    cursor: 'pointer',
    transition: 'background 0.15s',
  },

  // Loading
  loadingWrap: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.875rem',
  },
  repoLine: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.625rem',
  },
  spinnerWrap: {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  spinner: {
    display: 'inline-block',
    width: 16,
    height: 16,
    borderRadius: '50%',
    border: '2px solid var(--color-border)',
    borderTopColor: 'var(--color-accent)',
  },
  loadingUrl: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.8125rem',
    color: 'var(--color-text-muted)',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    minWidth: 0,
  },
  barTrack: {
    width: '100%',
    height: 4,
    borderRadius: 2,
    background: 'var(--color-border)',
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    borderRadius: 2,
    width: '0%',
  },
  statusMsg: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.8125rem',
    color: 'var(--color-text-muted)',
  },

  // Error
  errorBox: {
    width: '100%',
    display: 'flex',
    alignItems: 'flex-start',
    gap: '0.625rem',
    background: 'rgba(248, 113, 113, 0.08)',
    border: '1px solid rgba(248, 113, 113, 0.25)',
    borderRadius: 'var(--radius-md)',
    padding: '0.75rem 1rem',
  },
  errorIcon: {
    color: 'var(--color-error)',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.75rem',
    marginTop: '0.1rem',
    flexShrink: 0,
  },
  errorText: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.8125rem',
    color: 'var(--color-error)',
    lineHeight: 1.5,
  },

  // Feature grid
  featureGrid: {
    width: '100%',
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '0.875rem',
  },
  featureCard: {
    background: 'var(--color-surface)',
    border: '1px solid var(--color-border-subtle)',
    borderRadius: 'var(--radius-lg)',
    padding: '1.125rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
  },
  featureIcon: {
    fontSize: '1.25rem',
    color: 'var(--color-accent)',
    lineHeight: 1,
  },
  featureTitle: {
    fontFamily: 'var(--font-sans)',
    fontWeight: 700,
    fontSize: '0.9375rem',
    color: 'var(--color-text)',
  },
  featureDesc: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.75rem',
    color: 'var(--color-text-muted)',
    lineHeight: 1.6,
  },
}

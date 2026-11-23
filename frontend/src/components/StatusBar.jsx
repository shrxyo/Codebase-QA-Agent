import { useState, useEffect, useRef } from 'react'
import { checkHealth } from '../lib/api.js'

const POLL_INTERVAL_MS = 30_000

export default function StatusBar({ repo, stats, onHome }) {
  const [health, setHealth] = useState(null) // null = loading, 'ok' | 'error'
  const [chunksVerified, setChunksVerified] = useState(0)
  const timerRef = useRef(null)

  async function poll() {
    try {
      const data = await checkHealth()
      setHealth('ok')
      setChunksVerified(data.retrieved_chunks ?? 0)
    } catch {
      setHealth('error')
    }
  }

  useEffect(() => {
    poll()
    timerRef.current = setInterval(poll, POLL_INTERVAL_MS)
    return () => clearInterval(timerRef.current)
  }, [])

  const languageList = stats?.languages
    ? (Array.isArray(stats.languages)
        ? stats.languages.join(', ')
        : Object.keys(stats.languages).join(', '))
    : null

  return (
    <div style={styles.bar}>
      {/* Left: home button (once past the landing page) + health status */}
      <span style={styles.healthSection}>
        {onHome && (
          <button
            type="button"
            style={styles.homeBtn}
            onClick={onHome}
            title="Back to the landing page to index another repo"
            aria-label="Home"
            onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--color-accent)'; e.currentTarget.style.borderColor = 'var(--color-accent)' }}
            onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--color-text-muted)'; e.currentTarget.style.borderColor = 'var(--color-border)' }}
          >
            ⌂ Home
          </button>
        )}
        <span
          style={{
            ...styles.dot,
            background: health === 'ok'
              ? 'var(--color-success)'
              : health === 'error'
                ? 'var(--color-error)'
                : 'var(--color-text-dim)',
          }}
        />
        <span style={health === 'error' ? styles.offline : styles.healthy}>
          {health === null && 'Checking…'}
          {health === 'ok' && `RAG pipeline operational · ${chunksVerified} chunk${chunksVerified === 1 ? '' : 's'} verified`}
          {health === 'error' && 'Backend offline'}
        </span>
      </span>

      {/* Center: brand */}
      <span style={styles.brand}>
        Codebase-QA-Agent · RAG-powered code exploration
      </span>

      {/* Right: repo stats */}
      {repo && stats && (
        <span style={styles.statsSection}>
          <span style={styles.statItem}>{stats.files} file{stats.files === 1 ? '' : 's'}</span>
          <span style={styles.sep}>·</span>
          <span style={styles.statItem}>{stats.chunks} chunk{stats.chunks === 1 ? '' : 's'}</span>
          {languageList && (
            <>
              <span style={styles.sep}>·</span>
              <span style={styles.statItem}>{languageList}</span>
            </>
          )}
        </span>
      )}
    </div>
  )
}

const styles = {
  bar: {
    height: 'var(--status-bar-height)',
    flexShrink: 0,
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '1rem',
    paddingInline: '1rem',
    background: 'var(--color-surface)',
    borderTop: '1px solid var(--color-border-subtle)',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.72rem',
    color: 'var(--color-text-dim)',
    overflow: 'hidden',
  },
  healthSection: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.4rem',
    flexShrink: 0,
    minWidth: 0,
  },
  homeBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.25rem',
    marginRight: '0.5rem',
    padding: '0.1rem 0.5rem',
    background: 'transparent',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.72rem',
    cursor: 'pointer',
    whiteSpace: 'nowrap',
    transition: 'color 0.15s ease, border-color 0.15s ease',
  },
  dot: {
    width: 6,
    height: 6,
    borderRadius: '50%',
    flexShrink: 0,
    transition: 'background 0.3s ease',
  },
  healthy: {
    color: 'var(--color-success)',
    whiteSpace: 'nowrap',
  },
  offline: {
    color: 'var(--color-error)',
    whiteSpace: 'nowrap',
  },
  brand: {
    color: 'var(--color-accent)',
    fontWeight: 500,
    flexShrink: 0,
    whiteSpace: 'nowrap',
    position: 'absolute',
    left: '50%',
    transform: 'translateX(-50%)',
  },
  statsSection: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.3rem',
    flexShrink: 0,
    color: 'var(--color-text-dim)',
    overflow: 'hidden',
    whiteSpace: 'nowrap',
  },
  statItem: {
    color: 'var(--color-text-muted)',
  },
  sep: {
    color: 'var(--color-border)',
  },
}

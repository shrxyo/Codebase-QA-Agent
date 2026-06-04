export default function StatusBar({ repo, stats }) {
  return (
    <div style={styles.bar}>
      <span style={styles.brand}>Codebase-QA-Agent · RAG-powered code exploration</span>
      {repo && (
        <span style={styles.repoInfo}>
          <span style={styles.dot} />
          <span style={styles.repoUrl}>{repo}</span>
          {stats && (
            <span style={styles.stats}>
              {stats.files} files · {stats.chunks} chunks
            </span>
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
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    paddingInline: '1rem',
    background: 'var(--color-surface)',
    borderTop: '1px solid var(--color-border-subtle)',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.75rem',
    color: 'var(--color-text-dim)',
    overflow: 'hidden',
  },
  brand: {
    color: 'var(--color-accent)',
    fontWeight: 500,
    flexShrink: 0,
  },
  repoInfo: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    overflow: 'hidden',
    minWidth: 0,
  },
  dot: {
    width: 6,
    height: 6,
    borderRadius: '50%',
    background: 'var(--color-success)',
    flexShrink: 0,
  },
  repoUrl: {
    color: 'var(--color-text-muted)',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  stats: {
    color: 'var(--color-text-dim)',
    flexShrink: 0,
  },
}

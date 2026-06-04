// Stub — implemented in Story 2.4
export default function ChatPanel({ repo, summary }) {
  return (
    <div style={styles.root}>
      <p style={styles.label}>ChatPanel</p>
      <p style={styles.sub}>Implemented in Story 2.4</p>
      <p style={styles.repo}>{repo}</p>
    </div>
  )
}

const styles = {
  root: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '0.75rem',
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.875rem',
  },
  label: {
    color: 'var(--color-accent)',
    fontFamily: 'var(--font-sans)',
    fontWeight: 700,
    fontSize: '1.125rem',
  },
  sub: {
    color: 'var(--color-text-dim)',
  },
  repo: {
    color: 'var(--color-text)',
    fontFamily: 'var(--font-mono)',
  },
}

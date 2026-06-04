const COMPLEXITY_BADGE = {
  simple:   { label: 'simple',   bg: 'rgba(74, 222, 128, 0.12)', border: 'rgba(74, 222, 128, 0.3)',  color: '#4ade80' },
  moderate: { label: 'moderate', bg: 'rgba(251, 191, 36, 0.12)',  border: 'rgba(251, 191, 36, 0.3)', color: '#fbbf24' },
  complex:  { label: 'complex',  bg: 'rgba(249, 115, 22, 0.12)',  border: 'rgba(249, 115, 22, 0.3)', color: '#f97316' },
}

export default function RepoSummaryCard({ summary, stats, onContinue }) {
  const complexity = summary?.complexity && COMPLEXITY_BADGE[summary.complexity]
    ? summary.complexity
    : null

  const badge = complexity ? COMPLEXITY_BADGE[complexity] : null

  const repoName = summary?.repo_name ?? null
  const description = summary?.description ?? null
  const purpose = summary?.purpose ?? null
  const techStack = Array.isArray(summary?.tech_stack) && summary.tech_stack.length ? summary.tech_stack : null
  const keyConcepts = Array.isArray(summary?.key_concepts) && summary.key_concepts.length ? summary.key_concepts : null
  const entryPoints = Array.isArray(summary?.entry_points) && summary.entry_points.length ? summary.entry_points : null
  const bestFirstQuestion = summary?.best_first_question ?? null

  const languages = Array.isArray(stats?.languages) ? stats.languages : []

  return (
    <div style={s.shell}>
      <div className="fade-up" style={s.card}>

        {/* ── Top meta row ─────────────────────────────────────── */}
        <div style={s.metaRow}>
          <div style={s.headerLeft}>
            {repoName && <h1 style={s.repoName}>{repoName}</h1>}
            {badge && (
              <span style={{ ...s.complexityBadge, background: badge.bg, border: `1px solid ${badge.border}`, color: badge.color }}>
                {badge.label}
              </span>
            )}
          </div>

          {/* Stats */}
          {stats && (
            <div style={s.statsRow}>
              {stats.files != null && (
                <span style={s.statItem}>
                  <span style={s.statValue}>{stats.files}</span>
                  <span style={s.statLabel}> files</span>
                </span>
              )}
              {stats.chunks != null && (
                <>
                  <span style={s.statDot}>·</span>
                  <span style={s.statItem}>
                    <span style={s.statValue}>{stats.chunks}</span>
                    <span style={s.statLabel}> chunks</span>
                  </span>
                </>
              )}
              {languages.length > 0 && (
                <>
                  <span style={s.statDot}>·</span>
                  <span style={s.statLabel}>{languages.slice(0, 4).join(', ')}</span>
                </>
              )}
            </div>
          )}
        </div>

        {/* Divider */}
        <div style={s.divider} />

        {/* ── Description ──────────────────────────────────────── */}
        {description && (
          <p style={s.description}>{description}</p>
        )}

        {/* ── Purpose ──────────────────────────────────────────── */}
        {purpose && (
          <p style={s.purpose}>{purpose}</p>
        )}

        {/* ── Tech stack ───────────────────────────────────────── */}
        {techStack && (
          <section style={s.section}>
            <p style={s.sectionLabel}>Tech stack</p>
            <div style={s.chipRow}>
              {techStack.map((lang) => (
                <span key={lang} style={s.langChip}>{lang}</span>
              ))}
            </div>
          </section>
        )}

        {/* ── Key concepts ─────────────────────────────────────── */}
        {keyConcepts && (
          <section style={s.section}>
            <p style={s.sectionLabel}>Key concepts</p>
            <ul style={s.conceptList}>
              {keyConcepts.map((concept, i) => (
                <li key={i} style={s.conceptItem}>
                  <span style={s.bullet}>▸</span>
                  <span>{concept}</span>
                </li>
              ))}
            </ul>
          </section>
        )}

        {/* ── Entry points ─────────────────────────────────────── */}
        {entryPoints && (
          <section style={s.section}>
            <p style={s.sectionLabel}>Entry points</p>
            <div style={s.chipRow}>
              {entryPoints.map((file) => (
                <span key={file} style={s.fileChip}>{file}</span>
              ))}
            </div>
          </section>
        )}

        {/* ── Best first question callout ───────────────────────── */}
        {bestFirstQuestion && (
          <div style={s.callout}>
            <div style={s.calloutHeader}>
              <span style={s.calloutIcon}>◈</span>
              <span style={s.calloutTitle}>Best first question</span>
            </div>
            <p style={s.calloutQuestion}>{bestFirstQuestion}</p>
            <button style={s.askBtn} onClick={() => onContinue(bestFirstQuestion)}>
              Ask this →
            </button>
          </div>
        )}

        {/* ── Primary CTA ──────────────────────────────────────── */}
        <button style={s.startBtn} onClick={() => onContinue()}>
          Start Exploring →
        </button>

      </div>
    </div>
  )
}

const s = {
  shell: {
    flex: 1,
    overflowY: 'auto',
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'center',
    padding: '2rem 1rem',
  },
  card: {
    width: '100%',
    maxWidth: 720,
    background: 'var(--color-surface)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-xl)',
    padding: '2rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '1.25rem',
  },

  // Header
  metaRow: {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: '1rem',
    flexWrap: 'wrap',
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    flexWrap: 'wrap',
  },
  repoName: {
    fontFamily: 'var(--font-sans)',
    fontWeight: 800,
    fontSize: '1.75rem',
    letterSpacing: '-0.025em',
    color: 'var(--color-text)',
    lineHeight: 1.1,
  },
  complexityBadge: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.6875rem',
    fontWeight: 600,
    letterSpacing: '0.05em',
    textTransform: 'uppercase',
    borderRadius: '999px',
    padding: '0.2rem 0.625rem',
    flexShrink: 0,
  },

  // Stats
  statsRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.375rem',
    flexShrink: 0,
    marginTop: '0.25rem',
  },
  statItem: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.75rem',
  },
  statValue: {
    color: 'var(--color-text)',
    fontWeight: 600,
  },
  statLabel: {
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.75rem',
  },
  statDot: {
    color: 'var(--color-text-dim)',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.75rem',
  },

  divider: {
    height: 1,
    background: 'var(--color-border-subtle)',
    margin: '0 -0.25rem',
  },

  // Description & purpose
  description: {
    fontFamily: 'var(--font-sans)',
    fontSize: '1.0625rem',
    fontWeight: 500,
    color: 'var(--color-text)',
    lineHeight: 1.5,
  },
  purpose: {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.9375rem',
    color: 'var(--color-text-muted)',
    lineHeight: 1.65,
  },

  // Sections
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.625rem',
  },
  sectionLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.6875rem',
    fontWeight: 600,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
    color: 'var(--color-text-dim)',
  },

  // Tech stack / lang chips
  chipRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '0.5rem',
  },
  langChip: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.75rem',
    color: 'var(--color-accent)',
    background: 'var(--color-accent-glow)',
    border: '1px solid rgba(124,106,247,0.25)',
    borderRadius: '999px',
    padding: '0.25rem 0.75rem',
  },

  // Key concepts
  conceptList: {
    listStyle: 'none',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
  },
  conceptItem: {
    display: 'flex',
    alignItems: 'baseline',
    gap: '0.5rem',
    fontFamily: 'var(--font-sans)',
    fontSize: '0.9375rem',
    color: 'var(--color-text-muted)',
    lineHeight: 1.5,
  },
  bullet: {
    color: 'var(--color-accent)',
    fontSize: '0.75rem',
    flexShrink: 0,
  },

  // Entry points
  fileChip: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.75rem',
    color: 'var(--color-text-muted)',
    background: 'var(--color-surface-2)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-sm)',
    padding: '0.2rem 0.625rem',
  },

  // Best first question callout
  callout: {
    background: 'rgba(124, 106, 247, 0.07)',
    border: '1px solid rgba(124, 106, 247, 0.2)',
    borderRadius: 'var(--radius-lg)',
    padding: '1rem 1.25rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.625rem',
  },
  calloutHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  calloutIcon: {
    color: 'var(--color-accent)',
    fontSize: '0.875rem',
    lineHeight: 1,
  },
  calloutTitle: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.6875rem',
    fontWeight: 600,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
    color: 'var(--color-accent)',
  },
  calloutQuestion: {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.9375rem',
    color: 'var(--color-text)',
    lineHeight: 1.5,
  },
  askBtn: {
    alignSelf: 'flex-start',
    fontFamily: 'var(--font-sans)',
    fontWeight: 600,
    fontSize: '0.8125rem',
    padding: '0.4375rem 1rem',
    borderRadius: 'var(--radius-md)',
    background: 'var(--color-accent)',
    color: '#fff',
    border: 'none',
    cursor: 'pointer',
    transition: 'opacity 0.15s',
  },

  // Primary CTA
  startBtn: {
    alignSelf: 'flex-end',
    fontFamily: 'var(--font-sans)',
    fontWeight: 700,
    fontSize: '0.9375rem',
    padding: '0.625rem 1.5rem',
    borderRadius: 'var(--radius-md)',
    background: 'var(--color-success)',
    color: '#000',
    border: 'none',
    cursor: 'pointer',
    transition: 'opacity 0.15s',
    letterSpacing: '-0.01em',
  },
}

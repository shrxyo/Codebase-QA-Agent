import { useState } from 'react'

export default function MessageBubble({ msg }) {
  const [sourcesOpen, setSourcesOpen] = useState(true)
  const isUser = msg.role === 'user'

  if (isUser) {
    return (
      <div className="fade-up" style={s.userRow}>
        <div style={s.userBubble}>{msg.content}</div>
      </div>
    )
  }

  const hasError = Boolean(msg.error)

  return (
    <div className="fade-up" style={s.assistantRow}>
      <div style={s.assistantWrap}>

        {/* Header */}
        <div style={s.assistantHeader}>
          <span style={s.assistantIcon}>◈</span>
          <span style={s.assistantLabel}>Agent</span>
          {msg.streaming && <span style={s.streamingDot} />}
        </div>

        {/* Sources panel — above the answer, open by default */}
        {msg.sources?.length > 0 && (
          <div style={s.sourcesPanel}>
            <button
              style={s.sourcesToggle}
              onClick={() => setSourcesOpen((o) => !o)}
              aria-expanded={sourcesOpen}
            >
              <span style={s.toggleChevron}>{sourcesOpen ? '▾' : '▸'}</span>
              <span>Sources · {msg.sources.length}</span>
            </button>
            {sourcesOpen && (
              <div style={s.sourceChips}>
                {msg.sources.map((src, i) => (
                  <span key={i} style={s.sourceChip}>
                    {src.file}{src.lines ? `:${src.lines}` : ''}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Answer content */}
        <div style={hasError ? s.answerError : s.answerContent}>
          {msg.content ? parseMarkdown(msg.content) : null}
          {msg.streaming && <span style={s.cursor}>▋</span>}
        </div>

        {/* Error indicator */}
        {hasError && (
          <div style={s.errorRow}>
            <span>⚠</span>
            <span>{msg.error}</span>
          </div>
        )}

      </div>
    </div>
  )
}

/* ─── Markdown renderer ──────────────────────────────────────────────────────── */

function parseMarkdown(text) {
  const nodes = []
  const codeBlockRe = /```(\w*)\n?([\s\S]*?)```/g
  let lastIndex = 0
  let match
  let key = 0

  while ((match = codeBlockRe.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(...parseTextBlock(text.slice(lastIndex, match.index), key))
      key += 500
    }
    const lang = match[1].trim() || 'text'
    nodes.push(
      <div key={key++} style={s.codeBlock}>
        <div style={s.codeLangLabel}>{lang}</div>
        <pre style={s.codePre}>
          <code style={s.codeContent}>{match[2]}</code>
        </pre>
      </div>
    )
    lastIndex = match.index + match[0].length
  }

  if (lastIndex < text.length) {
    nodes.push(...parseTextBlock(text.slice(lastIndex), key))
  }

  return nodes
}

function parseTextBlock(text, baseKey) {
  const nodes = []
  let key = baseKey

  for (const line of text.split('\n')) {
    if (line.startsWith('### ')) {
      nodes.push(<h3 key={key++} style={s.h3}>{parseInline(line.slice(4))}</h3>)
    } else if (line.startsWith('## ')) {
      nodes.push(<h2 key={key++} style={s.h2}>{parseInline(line.slice(3))}</h2>)
    } else if (line.startsWith('# ')) {
      nodes.push(<h1 key={key++} style={s.h1}>{parseInline(line.slice(2))}</h1>)
    } else if (line.trim() === '') {
      nodes.push(<div key={key++} style={s.lineSpacer} />)
    } else {
      nodes.push(<p key={key++} style={s.para}>{parseInline(line)}</p>)
    }
  }

  return nodes
}

function parseInline(text) {
  const parts = []
  const re = /(\*\*([\s\S]+?)\*\*|`([^`]+)`)/g
  let last = 0
  let match
  let key = 0

  while ((match = re.exec(text)) !== null) {
    if (match.index > last) {
      parts.push(text.slice(last, match.index))
    }
    if (match[2] !== undefined) {
      parts.push(<strong key={key++} style={s.bold}>{match[2]}</strong>)
    } else {
      parts.push(<code key={key++} style={s.inlineCode}>{match[3]}</code>)
    }
    last = match.index + match[0].length
  }

  if (last < text.length) {
    parts.push(text.slice(last))
  }

  return parts
}

/* ─── Styles ────────────────────────────────────────────────────────────────── */
const s = {
  // User message
  userRow: {
    display: 'flex',
    justifyContent: 'flex-end',
  },
  userBubble: {
    background: 'var(--color-accent)',
    color: '#fff',
    borderRadius: 'var(--radius-lg)',
    borderBottomRightRadius: 'var(--radius-sm)',
    padding: '0.625rem 1rem',
    fontFamily: 'var(--font-sans)',
    fontSize: '0.9375rem',
    lineHeight: 1.55,
    maxWidth: '72%',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  },

  // Assistant message
  assistantRow: {
    display: 'flex',
    justifyContent: 'flex-start',
    width: '100%',
  },
  assistantWrap: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
    width: '100%',
    minWidth: 0,
  },
  assistantHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.375rem',
  },
  assistantIcon: {
    color: 'var(--color-accent)',
    fontSize: '0.75rem',
    lineHeight: 1,
  },
  assistantLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.6875rem',
    fontWeight: 600,
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
    color: 'var(--color-accent)',
  },
  streamingDot: {
    display: 'inline-block',
    width: 6,
    height: 6,
    borderRadius: '50%',
    background: 'var(--color-accent)',
    opacity: 0.7,
    animation: 'spin 1s linear infinite',
  },

  // Sources panel
  sourcesPanel: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.375rem',
    borderLeft: '2px solid var(--color-border)',
    paddingLeft: '0.75rem',
  },
  sourcesToggle: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '0.3rem',
    background: 'none',
    border: 'none',
    padding: 0,
    cursor: 'pointer',
    color: 'var(--color-text-dim)',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.625rem',
    fontWeight: 600,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
  },
  toggleChevron: {
    fontSize: '0.75rem',
  },
  sourceChips: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '0.375rem',
  },
  sourceChip: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.6875rem',
    color: 'var(--color-text-muted)',
    background: 'var(--color-surface-2)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-sm)',
    padding: '0.1875rem 0.5rem',
  },

  // Answer
  answerContent: {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.9375rem',
    color: 'var(--color-text)',
    lineHeight: 1.65,
    wordBreak: 'break-word',
  },
  answerError: {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.9375rem',
    color: 'var(--color-error)',
    lineHeight: 1.65,
    wordBreak: 'break-word',
  },

  // Error row
  errorRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.375rem',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.8125rem',
    color: 'var(--color-error)',
    lineHeight: 1.5,
  },

  // Streaming cursor
  cursor: {
    color: 'var(--color-accent)',
    opacity: 0.9,
    animation: 'blink 1s step-end infinite',
  },

  // Inline text elements
  para: {
    margin: 0,
    lineHeight: 1.65,
  },
  lineSpacer: {
    height: '0.5rem',
  },
  bold: {
    fontWeight: 700,
  },
  inlineCode: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.875em',
    background: 'var(--color-surface-2)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-sm)',
    padding: '0.125em 0.375em',
    color: 'var(--color-text)',
  },

  // Headings
  h1: {
    fontFamily: 'var(--font-sans)',
    fontSize: '1.25rem',
    fontWeight: 700,
    color: 'var(--color-text)',
    lineHeight: 1.35,
    marginTop: '0.75rem',
    marginBottom: '0.25rem',
  },
  h2: {
    fontFamily: 'var(--font-sans)',
    fontSize: '1.0625rem',
    fontWeight: 700,
    color: 'var(--color-text)',
    lineHeight: 1.4,
    marginTop: '0.625rem',
    marginBottom: '0.25rem',
  },
  h3: {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.9375rem',
    fontWeight: 700,
    color: 'var(--color-text)',
    lineHeight: 1.4,
    marginTop: '0.5rem',
    marginBottom: '0.125rem',
  },

  // Code block
  codeBlock: {
    background: 'var(--color-surface-2)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-md)',
    overflow: 'hidden',
    margin: '0.375rem 0',
  },
  codeLangLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.625rem',
    fontWeight: 600,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
    color: 'var(--color-text-dim)',
    padding: '0.3125rem 0.75rem',
    borderBottom: '1px solid var(--color-border)',
    background: 'rgba(0,0,0,0.25)',
  },
  codePre: {
    overflow: 'auto',
    padding: '0.75rem',
    margin: 0,
  },
  codeContent: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.8125rem',
    color: 'var(--color-text)',
    lineHeight: 1.6,
    display: 'block',
    whiteSpace: 'pre',
  },
}

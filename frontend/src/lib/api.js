const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const POLL_INTERVAL_MS = 2000
// Render's free CPU needs ~10 min for a medium repo; give up well after that.
const POLL_TIMEOUT_MS = 30 * 60 * 1000

async function errorFrom(res) {
  const body = await res.json().catch(() => ({}))
  return new Error(body.detail ?? `Server error ${res.status}`)
}

// POST /index answers 200 with the repo if it's already indexed, or 202 once
// a background build is queued. In the 202 case, poll /index/status until the
// build finishes, so callers always get the final indexed payload.
export async function indexRepo(repoUrl, onStatus) {
  const res = await fetch(`${BASE_URL}/index`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ repo_url: repoUrl }),
  })
  if (!res.ok) throw await errorFrom(res)

  let data = await res.json()
  const deadline = Date.now() + POLL_TIMEOUT_MS
  while (data.status !== 'indexed') {
    if (data.status === 'failed') throw new Error(data.error ?? 'Indexing failed')
    if (Date.now() > deadline) throw new Error('Indexing is taking too long. Try again later.')
    onStatus?.(data.status)
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS))
    const poll = await fetch(`${BASE_URL}/index/status?repo_url=${encodeURIComponent(data.repo_url)}`)
    if (!poll.ok) throw await errorFrom(poll)
    data = await poll.json()
  }
  return data
}

export async function checkHealth() {
  const res = await fetch(`${BASE_URL}/health`)
  if (!res.ok) throw new Error(`Server error ${res.status}`)
  return res.json()
}

export async function listRepos() {
  const res = await fetch(`${BASE_URL}/repos`)
  if (!res.ok) throw new Error(`Server error ${res.status}`)
  return res.json()
}

export function streamChat(repoUrl, question, history, onEvent) {
  const controller = new AbortController()

  fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ repo_url: repoUrl, question, history }),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        onEvent({ type: 'error', message: body.detail ?? `Server error ${res.status}` })
        return
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const raw = line.slice(6).trim()
            if (raw === '[DONE]') {
              onEvent({ type: 'done' })
              return
            }
            try {
              onEvent(JSON.parse(raw))
            } catch {
              // malformed line — skip
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        onEvent({ type: 'error', message: err.message })
      }
    })

  return () => controller.abort()
}

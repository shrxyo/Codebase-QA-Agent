const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export async function indexRepo(repoUrl) {
  const res = await fetch(`${BASE_URL}/index`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ repo_url: repoUrl }),
  })

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? `Server error ${res.status}`)
  }

  return res.json()
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

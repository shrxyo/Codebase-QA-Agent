import json
from typing import Generator

import llm_client
from indexer import RepoIndexer

SYSTEM_PROMPT = (
    "You are an expert code archaeologist. Answer questions about a GitHub repository "
    "using retrieved code chunks. Always cite files using [filename:lines] notation inline. "
    "Trace actual call flows through the code. Be precise and concise. "
    "If context is insufficient, say so clearly."
)

MAX_HISTORY_MESSAGES = 6
MAX_TOKENS = 1500


def stream_answer(
    indexer: RepoIndexer,
    question: str,
    history: list[dict],
) -> Generator[str, None, None]:
    chunks = indexer.query(question, n_results=8)

    sources = [{"file": c["file"], "lines": c["lines"]} for c in chunks]
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

    context_parts = []
    for c in chunks:
        context_parts.append(f"### {c['file']} (lines {c['lines']})\n```{c['language']}\n{c['content']}\n```")
    context_block = "\n\n".join(context_parts)

    trimmed_history = history[-MAX_HISTORY_MESSAGES:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(trimmed_history)
    messages.append({
        "role": "user",
        "content": f"Relevant code context:\n\n{context_block}\n\nQuestion: {question}",
    })

    client = llm_client.get_llm_client()
    model = llm_client.get_model_name()

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield f"data: {json.dumps({'type': 'text', 'content': delta.content})}\n\n"

    yield "data: [DONE]\n\n"

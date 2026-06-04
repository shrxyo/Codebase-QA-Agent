import json
import logging
import os
from typing import TYPE_CHECKING

import llm_client

if TYPE_CHECKING:
    from indexer import RepoIndexer

logger = logging.getLogger(__name__)

_README_CANDIDATES = ["README.md", "readme.md", "README.rst"]
_ENTRY_POINT_CANDIDATES = [
    "main.py", "app.py", "index.js", "index.ts",
    "server.py", "manage.py", os.path.join("cmd", "main.go"),
]
_README_MAX_CHARS = 3000
_ENTRY_POINT_MAX_LINES = 100

_FALLBACK: dict = {
    "name": "",
    "description": "Could not generate summary",
    "purpose": "",
    "tech_stack": [],
    "entry_points": [],
    "key_concepts": [],
    "complexity": "",
    "best_first_question": "",
}

_SYSTEM_PROMPT = (
    "You are a senior software engineer. "
    "Analyze the provided repository context and return ONLY a raw JSON object "
    "with no markdown fences, no explanation, no extra text. "
    "The JSON must have exactly these keys: "
    "name (string), description (one sentence string), purpose (2-3 sentences string), "
    "tech_stack (array of strings), entry_points (array of strings), "
    "key_concepts (array of 3-5 strings), "
    "complexity (exactly one of: simple, moderate, complex), "
    "best_first_question (string)."
)


def _read_readme(repo_dir: str) -> str:
    for name in _README_CANDIDATES:
        path = os.path.join(repo_dir, name)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    return f.read(_README_MAX_CHARS)
            except OSError:
                pass
    return ""


def _top_level_listing(repo_dir: str) -> list[str]:
    try:
        entries = os.listdir(repo_dir)
    except OSError:
        return []
    return sorted(e for e in entries if not e.startswith("."))


def _read_entry_points(repo_dir: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for candidate in _ENTRY_POINT_CANDIDATES:
        path = os.path.join(repo_dir, candidate)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= _ENTRY_POINT_MAX_LINES:
                            break
                        lines.append(line)
                found[candidate] = "".join(lines)
            except OSError:
                pass
    return found


def _complexity_label(file_count: int) -> str:
    if file_count < 20:
        return "simple"
    if file_count > 200:
        return "complex"
    return "moderate"


def generate_repo_summary(
    indexer: "RepoIndexer",
    repo_dir: str,
    repo_url: str,
) -> dict:
    file_count = indexer._count_unique_files(indexer.chunks)
    complexity = _complexity_label(file_count)

    readme = _read_readme(repo_dir)
    listing = _top_level_listing(repo_dir)
    entry_point_contents = _read_entry_points(repo_dir)

    context_parts: list[str] = [
        f"Repository URL: {repo_url}",
        f"File count: {file_count} ({complexity} complexity)",
        "",
        "Top-level contents:",
        "  " + ", ".join(listing) if listing else "  (empty)",
    ]

    if readme:
        context_parts += ["", "README (truncated to 3000 chars):", readme]

    for ep_name, ep_content in entry_point_contents.items():
        context_parts += [
            "",
            f"Entry point — {ep_name} (first {_ENTRY_POINT_MAX_LINES} lines):",
            ep_content,
        ]

    user_message = "\n".join(context_parts)

    client = llm_client.get_llm_client()
    model = llm_client.get_model_name()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=600,
            temperature=0.2,
        )
        raw = response.choices[0].message.content or ""
        summary = json.loads(raw.strip())
        # Force the complexity we computed rather than trusting the LLM
        summary["complexity"] = complexity
        return summary
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning("Failed to parse summary JSON: %s", exc)
        return {**_FALLBACK, "complexity": complexity}
    except Exception as exc:
        logger.warning("Summary generation failed: %s", exc)
        return {**_FALLBACK, "complexity": complexity}

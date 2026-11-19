"""Pick a chunker per file: AST-based for Python, fixed line windows for the rest."""
from __future__ import annotations

from chunkers import chunk_lines
from chunkers.python_chunker import chunk_python


def chunk_file(lines: list[str], file: str, language: str, function_aware: bool = True) -> list[dict]:
    """language is the file extension without the dot ("py", "md", ...)."""
    if function_aware and language == "py":
        return chunk_python(lines, file, language)
    return chunk_lines(lines, file, language)

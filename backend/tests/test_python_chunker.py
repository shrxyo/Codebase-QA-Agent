import textwrap

import pytest

from chunkers import chunk_lines
from chunkers.python_chunker import MAX_LINES, SMALL_LINES, chunk_python
from chunkers.router import chunk_file

REQUIRED_KEYS = {"content", "file", "lines", "language", "node_type"}


def src(text: str) -> list[str]:
    return textwrap.dedent(text).lstrip("\n").splitlines(keepends=True)


def span(chunk):
    a, b = chunk["lines"].split("-")
    return int(a), int(b)


def assert_valid(chunks, lines):
    """Shape from docs/CLAUDE.md, content matches its line range, no overlaps."""
    prev_end = 0
    for c in chunks:
        assert set(c) >= REQUIRED_KEYS
        a, b = span(c)
        assert 1 <= a <= b <= len(lines)
        assert c["content"] == "".join(lines[a - 1:b])
        assert a > prev_end, "chunks overlap or are out of order"
        prev_end = b


def find(chunks, needle):
    return next(c for c in chunks if needle in c["content"])


def filler_function(name: str, n_body: int) -> str:
    body = "".join(f"    x{i} = compute_something_long({i})  # step {i}\n" for i in range(n_body))
    return f"def {name}():\n    \"\"\"Docstring of {name}.\"\"\"\n{body}    return x0\n"


# ---------------------------------------------------------------------------


def test_each_def_and_class_starts_its_own_chunk_with_decorators_and_docstring():
    lines = src(filler_function("first", 25)) + ["\n", "\n"] + src('''
        # Leading comment that belongs to the decorated function.
        @app.post("/chat")
        @requires_auth
        def chat(body):
            """Answer a chat question."""
            return stream(body)


        class Indexer:
            """Indexes repos."""

            def query(self, q):
                return q
    ''')
    chunks = chunk_python(lines, "main.py")
    assert_valid(chunks, lines)

    chat = find(chunks, "def chat")
    assert chat["content"].startswith("# Leading comment")
    assert '@app.post("/chat")' in chat["content"]
    assert '"""Answer a chat question."""' in chat["content"]
    assert chat["node_type"] in {"function", "group"}
    assert "def first" not in chat["content"]  # first() is 29 lines: not merged

    cls = find(chunks, "class Indexer")
    assert cls["content"].lstrip().startswith(("class Indexer", "@", "#")) or cls["node_type"] == "group"
    assert "def query" in cls["content"]  # a small class stays whole


def test_definition_is_at_the_start_of_its_chunk_not_mid_window():
    # 40 lines of other code, then the definition: a 60-line window starting at
    # line 1 would bury it at line 41; the AST chunker starts a chunk there.
    lines = src(filler_function("before", 37) + "\n\nclass ConnectTimeout(Exception):\n    pass\n")
    chunks = chunk_python(lines, "exceptions.py")
    target = find(chunks, "class ConnectTimeout")
    assert target["content"].startswith("class ConnectTimeout")
    assert chunk_lines(lines, "exceptions.py", "py")[0]["content"].count("class ConnectTimeout") == 1


def test_large_function_split_at_statement_boundaries_first_piece_keeps_signature():
    lines = src(filler_function("huge", 150))
    chunks = chunk_python(lines, "big.py")
    assert_valid(chunks, lines)
    assert len(chunks) >= 3
    assert all(span(c)[1] - span(c)[0] + 1 <= MAX_LINES for c in chunks)
    assert chunks[0]["content"].startswith("def huge():\n    \"\"\"Docstring of huge.\"\"\"")
    # Every line is covered exactly once.
    assert "".join(c["content"] for c in chunks) == "".join(lines)


def test_large_class_split_into_methods():
    methods = "".join(
        textwrap.indent(filler_function(f"m{i}", 20), "    ") + "\n" for i in range(4)
    )
    lines = src('class Big:\n    """Big class docstring."""\n    LIMIT = 30\n\n' + methods)
    chunks = chunk_python(lines, "big.py")
    assert_valid(chunks, lines)
    assert chunks[0]["content"].startswith("class Big:")
    assert "Big class docstring" in chunks[0]["content"]
    for i in range(4):
        assert find(chunks, f"def m{i}")["content"].lstrip().startswith(f"def m{i}")


def test_single_statement_too_long_falls_back_to_line_windows():
    big_dict = "CONFIG = {\n" + "".join(f"    'k{i}': {i},\n" for i in range(130)) + "}\n"
    lines = src("import os\n\n" + big_dict)
    chunks = chunk_python(lines, "config.py")
    assert any(c["node_type"] == "chunk" for c in chunks)
    assert all(span(c)[1] - span(c)[0] + 1 <= MAX_LINES for c in chunks)


def test_small_adjacent_pieces_are_merged():
    lines = src("".join(f"class E{i}(Exception):\n    pass\n\n\n" for i in range(12)))
    chunks = chunk_python(lines, "exceptions.py")
    assert_valid(chunks, lines)
    assert len(chunks) < 12
    assert all(span(c)[1] - span(c)[0] + 1 <= SMALL_LINES for c in chunks)


@pytest.mark.parametrize("bad", ["def broken(:\n    pass\n" * 40, "x = 1\n\x00\n" * 40])
def test_unparseable_file_falls_back_to_60_line_chunks(bad):
    lines = bad.splitlines(keepends=True)
    chunks = chunk_python(lines, "bad.py")
    assert chunks == chunk_lines(lines, "bad.py", "py")
    assert {c["node_type"] for c in chunks} == {"chunk"}
    assert span(chunks[0]) == (1, 60)


def test_empty_and_tiny_files():
    assert chunk_python([], "empty.py") == []
    assert chunk_python(["x = 1\n"], "tiny.py") == []  # under MIN_CHUNK_CHARS, like the line chunker


def test_router_sends_py_to_ast_chunker_and_others_to_lines():
    py = src(filler_function("a", 40) + "\n\n" + filler_function("b", 40))
    routed = chunk_file(py, "a.py", "py")
    assert routed == chunk_python(py, "a.py")
    assert any(c["node_type"] == "function" for c in routed)
    assert chunk_file(py, "a.py", "py", function_aware=False) == chunk_lines(py, "a.py", "py")

    md = ["# Title\n"] + [f"Line {i} of the readme text.\n" for i in range(100)]
    assert chunk_file(md, "README.md", "md") == chunk_lines(md, "README.md", "md")


def test_real_file_from_this_repo_is_fully_covered():
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "indexer.py"
    lines = path.read_text().splitlines(keepends=True)
    chunks = chunk_python(lines, "backend/indexer.py")
    assert_valid(chunks, lines)
    assert find(chunks, "def query(self")["content"].lstrip().startswith("def query(self")

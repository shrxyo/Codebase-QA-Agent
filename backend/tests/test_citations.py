"""Unit tests for agent.parse_citations / agent.verify_citations.

Several inputs are copied from real model answers in the task 5 baseline
(eval/results/answers_20261119T045547Z.json), including the U+2011
non-breaking hyphens the model writes inside line ranges and file names.
"""
import pytest

from agent import parse_citations, verify_citations

INDEXED = {
    "backend/main.py",
    "backend/indexer.py",
    "docker-compose.yml",
    "src/requests/auth.py",
    "src/requests/sessions.py",
    "tests/test_auth.py",
    "Dockerfile",
}
RETRIEVED = [
    {"file": "backend/indexer.py", "lines": "1-60"},
    {"file": "docker-compose.yml", "lines": "1-18"},
    {"file": "src/requests/auth.py", "lines": "251-310"},
    {"file": "src/requests/sessions.py", "lines": "151-210"},
    {"file": "Dockerfile", "lines": "1-20"},
]


def spans(answer):
    return [(c["file"], c["start"], c["end"]) for c in parse_citations(answer)]


# --- parsing ---------------------------------------------------------------

@pytest.mark.parametrize("answer, expected", [
    ("see [backend/main.py:10-20].", [("backend/main.py", 10, 20)]),
    ("see [backend/main.py:42]", [("backend/main.py", 42, 42)]),
    ("[src/requests/auth.py#L273-L300]", [("src/requests/auth.py", 273, 300)]),
    ("[src/requests/auth.py lines 273-300]", [("src/requests/auth.py", 273, 300)]),
    ("[src/requests/auth.py, line 273]", [("src/requests/auth.py", 273, 273)]),
    ("[`backend/main.py:5-9`]", [("backend/main.py", 5, 9)]),
    ("[./backend/main.py:5]", [("./backend/main.py", 5, 5)]),
    ("[Dockerfile:3]", [("Dockerfile", 3, 3)]),
    # Real baseline output: U+2011 in the range and in the file name.
    ("# [docker‑compose.yml:5‑12]", [("docker-compose.yml", 5, 12)]),
    ("[backend/indexer.py:1‑15]", [("backend/indexer.py", 1, 15)]),
    # En dash and narrow no-break space.
    ("[a/b.py: 10 – 20]", [("a/b.py", 10, 20)]),
    # Several ranges / several files in one bracket.
    ("[backend/main.py:10-20, 30-40]", [("backend/main.py", 10, 20), ("backend/main.py", 30, 40)]),
    ("[backend/main.py:1-5; docker-compose.yml:2]", [("backend/main.py", 1, 5), ("docker-compose.yml", 2, 2)]),
])
def test_parses_formats_the_model_produces(answer, expected):
    assert spans(answer) == expected


@pytest.mark.parametrize("answer", [
    # Non-citation brackets that appear in real baseline answers.
    "data: [DONE]",
    "Dict[str, _VT]",
    'headers["Location"]',
    '["hook", lambda x: None, "hook"]',
    "cite with [filename:lines] notation",
    "[lines 151‑178]",
    "messages[0]",
    # A file without lines is not a [file:lines] citation.
    "[backend/main.py]",
    # Python slicing.
    "xs[i:3]",
    # Prose references are not counted (strict definition, see agent.py).
    "see `backend/indexer.py` lines 51‑110",
])
def test_ignores_non_citations(answer):
    assert parse_citations(answer) == []


def test_repeated_citation_counts_once():
    assert len(parse_citations("[backend/main.py:5] and again [backend/main.py:5]")) == 1


def test_raw_is_exact_bracket_text():
    raw = "[docker‑compose.yml:5‑12]"
    assert parse_citations(f"x {raw} y")[0]["raw"] == raw


# --- verification ----------------------------------------------------------

def verify(answer):
    return verify_citations(answer, RETRIEVED, INDEXED)


def test_valid_when_lines_overlap_a_retrieved_chunk():
    r = verify("[backend/indexer.py:55-70]")  # overlaps 1-60 at 55-60
    assert r["score"] == 1.0 and r["citations"][0]["valid"]
    assert r["reason"] == "The citation points at code retrieved for this answer."


def test_boundary_overlap_counts():
    assert verify("[backend/indexer.py:60]")["score"] == 1.0
    assert verify("[backend/indexer.py:61]")["score"] == 0.0


def test_lines_outside_retrieved_chunks():
    r = verify("[backend/indexer.py:100-120]")
    assert r["score"] == 0.0
    assert r["citations"][0]["why"] == "lines outside the retrieved chunks"


def test_file_indexed_but_not_retrieved():
    r = verify("[backend/main.py:1-10]")
    assert r["citations"][0]["why"] == "file not retrieved for this answer"


def test_file_not_in_index():
    r = verify("[backend/nope.py:1-10]")
    assert r["citations"][0]["why"] == "file not in the index"
    assert "backend/nope.py:1-10 (file not in the index)" in r["reason"]


def test_suffix_match_for_dropped_folders():
    # The model writes "requests/auth.py" for "src/requests/auth.py".
    r = verify("[requests/auth.py:273]")
    assert r["score"] == 1.0
    assert r["citations"][0]["resolved"] == "src/requests/auth.py"


def test_suffix_match_is_by_whole_path_component():
    # "auth.py" must not match "test_auth.py"; it matches src/requests/auth.py.
    r = verify("[auth.py:260]")
    assert r["citations"][0]["resolved"] == "src/requests/auth.py"


def test_unicode_file_name_resolves():
    assert verify("[docker‑compose.yml:5‑12]")["score"] == 1.0


def test_malformed_range():
    r = verify("[backend/indexer.py:20-10]")
    assert r["citations"][0]["why"] == "malformed line range"


def test_zero_citations_scores_zero_without_dividing():
    r = verify("The answer is in the indexer, no brackets here.")
    assert r["score"] == 0.0 and r["total"] == 0
    assert "No [file:lines] citations" in r["reason"]


def test_mixed_score_and_reason():
    r = verify(
        "[backend/indexer.py:10] [src/requests/sessions.py:160-170] "
        "[backend/main.py:5] [ghost.py:1]"
    )
    assert (r["valid"], r["total"], r["score"]) == (2, 4, 0.5)
    assert r["reason"].startswith("2 of 4 citations verified. Not verified: backend/main.py:5")
    assert "ghost.py:1 (file not in the index)" in r["reason"]


def test_reason_truncates_long_invalid_list():
    answer = " ".join(f"[ghost.py:{i}]" for i in range(1, 9))
    assert verify(answer)["reason"].endswith("; +3 more.")


def test_works_without_indexed_files():
    # Retrieved files always count as indexed.
    assert verify_citations("[Dockerfile:2]", RETRIEVED, [])["score"] == 1.0

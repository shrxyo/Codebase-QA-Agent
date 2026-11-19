from retrieval.bm25_index import BM25Index, tokenize


def test_tokenize_splits_snake_camel_dots_slashes_and_keeps_compounds():
    toks = tokenize("HTTPAdapter.send_request in src/requests/sessions.py")
    for t in ["http", "adapter", "send", "request", "src", "requests", "sessions", "py",
              "httpadapter", "send_request", "httpadapter.send_request"]:
        assert t in toks, t
    assert "in" not in toks  # stopword


def test_tokenize_camel_case_and_digits():
    assert tokenize("getUserById2")[:4] == ["get", "user", "id", "2"]  # "by" is a stopword
    assert "getuserbyid2" in tokenize("getUserById2")


def test_tokenize_snake_case_not_double_counted():
    assert tokenize("should_strip_auth").count("should_strip_auth") == 1


def test_search_prefers_exact_identifier_and_drops_non_matches():
    idx = BM25Index.from_documents(
        ["a", "b", "c"],
        ["class ConnectTimeout(ConnectionError): pass",
         "def send(self): raise ReadTimeout",
         "unrelated readme text"],
    )
    ranked = idx.search("Where is ConnectTimeout defined?", 3)
    assert ranked[0][0] == "a"
    assert "c" not in [cid for cid, _ in ranked]


def test_empty_index_and_query():
    assert BM25Index([], []).search("anything", 5) == []
    assert BM25Index.from_documents(["a"], ["text"]).search("the", 5) == []  # only stopwords


def test_save_load_roundtrip(tmp_path):
    idx = BM25Index.from_documents(["a", "b"], ["alpha beta", "gamma delta"])
    path = str(tmp_path / "bm25" / "x.json")
    idx.save(path)
    assert BM25Index.load(path).search("gamma", 2) == idx.search("gamma", 2)

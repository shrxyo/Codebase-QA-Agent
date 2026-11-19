from indexer import MAX_JSON_BYTES, should_index


def test_source_files_are_indexed_at_any_size():
    assert should_index("indexer.py", 500_000)
    assert should_index("App.jsx", 100)
    assert should_index("README.md", 200_000)


def test_unsupported_extensions_are_skipped():
    assert not should_index("logo.png", 100)
    assert not should_index("Makefile", 100)


def test_lockfiles_are_skipped_by_name():
    assert not should_index("package-lock.json", 100)
    assert not should_index("pnpm-lock.yaml", 100)


def test_small_json_configs_are_kept():
    assert should_index("package.json", 400)
    assert should_index("tsconfig.json", MAX_JSON_BYTES)


def test_large_json_data_dumps_are_skipped():
    assert not should_index("answers_20261119.json", MAX_JSON_BYTES + 1)
    assert not should_index("FIXTURE.JSON", 70_000)


def test_size_limit_applies_only_to_json():
    assert should_index("pnpm-workspace.yaml", 70_000)
    assert should_index("CHANGELOG.md", 70_000)

import pytest

from retrieval.fusion import RRF_K, reciprocal_rank_fusion


def scores(result):
    return dict(result)


def test_single_list_keeps_order_and_uses_1_over_k_plus_rank():
    out = reciprocal_rank_fusion([["a", "b", "c"]])
    assert [d for d, _ in out] == ["a", "b", "c"]
    assert scores(out)["a"] == pytest.approx(1 / (RRF_K + 1))
    assert scores(out)["c"] == pytest.approx(1 / (RRF_K + 3))


def test_item_in_both_lists_beats_item_top_of_one():
    # "b" is 2nd in both lists; "a" and "x" are each 1st in only one.
    out = reciprocal_rank_fusion([["a", "b", "c"], ["x", "b", "y"]])
    assert out[0][0] == "b"
    assert scores(out)["b"] == pytest.approx(2 / (RRF_K + 2))


def test_union_of_all_items():
    out = reciprocal_rank_fusion([["a", "b"], ["c"]])
    assert {d for d, _ in out} == {"a", "b", "c"}


def test_empty_inputs():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    assert [d for d, _ in reciprocal_rank_fusion([[], ["a"]])] == ["a"]


def test_ties_are_deterministic_by_best_rank_then_list_order():
    # a and x both score 1/(k+1); a's list comes first.
    out = reciprocal_rank_fusion([["a"], ["x"]])
    assert [d for d, _ in out] == ["a", "x"]
    out = reciprocal_rank_fusion([["x"], ["a"]])
    assert [d for d, _ in out] == ["x", "a"]
    # p (ranks 1 and 3) and q (2 and 2): equal-ish? 1/61+1/63 > 2/62, so p wins.
    out = reciprocal_rank_fusion([["p", "q", "r"], ["s", "q", "p"]])
    assert [d for d, _ in out][:2] == ["p", "q"]


def test_duplicates_within_a_list_count_once_at_best_rank():
    out = reciprocal_rank_fusion([["a", "a", "b"]])
    assert scores(out)["a"] == pytest.approx(1 / (RRF_K + 1))
    assert scores(out)["b"] == pytest.approx(1 / (RRF_K + 3))


def test_k_controls_how_much_top_ranks_dominate():
    lists = [["a", "b", "c", "d"], ["d", "c", "b", "a"]]
    # Symmetric lists: every item ties on total rank, so with k=60 they're
    # almost equal; with k=0, being 1st in one list wins outright.
    k0 = scores(reciprocal_rank_fusion(lists, k=0))
    assert k0["a"] > k0["b"]
    k60 = scores(reciprocal_rank_fusion(lists))
    assert k60["a"] - k60["b"] < k0["a"] - k0["b"]


def test_pure_inputs_not_mutated_and_any_hashable_works():
    lists = [[("f.py", "1-60"), ("g.py", "1-60")], [("g.py", "1-60")]]
    snapshot = [list(l) for l in lists]
    out = reciprocal_rank_fusion(lists)
    assert lists == snapshot
    assert out[0][0] == ("g.py", "1-60")


def test_negative_k_rejected():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)

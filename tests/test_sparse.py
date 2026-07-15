from src.retrieval.sparse import tokenize, top_positive_score_indices


def test_tokenize_preserves_financial_reporting_terms() -> None:
    assert tokenize("The 10-K showed 12.5% year-over-year growth.") == [
        "the",
        "10-k",
        "showed",
        "12.5",
        "year-over-year",
        "growth",
    ]


def test_top_positive_score_indices_returns_top_k_positive_scores() -> None:
    scores = [0.0, 1.2, -0.5, 3.4, 2.1]

    assert top_positive_score_indices(scores, k=2) == [3, 4]


def test_top_positive_score_indices_excludes_non_positive_scores() -> None:
    scores = [0.0, -0.2, 0.5]

    assert top_positive_score_indices(scores, k=10) == [2]

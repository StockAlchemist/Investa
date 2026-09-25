"""Tests for weighting the AI review into the Buffett ranking.

The review is a soft, unbacktestable signal, so the cases here pin down that it
can only ever do what its weight says: nothing at weight zero, nothing to a
company that has not been reviewed, and nothing that escapes the confidence
factor a thin filing record earns.
"""

import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import ai_review_scores as ars
import buffett_rank as br

DIMS = list(ars.AI_DIMENSIONS)


def _frame():
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "name": ["Alpha Corp", "Beta Inc", "Gamma Ltd"],
            "model": ["generic", "generic", "bank"],
            "rank": [1, 2, 3],
            "quality_score": [80.0, 70.0, 60.0],
            "value_score": [60.0, 50.0, np.nan],
            "confidence": [1.0, 0.5, 1.0],
        }
    )


def _reviews(ratings):
    frame = pd.DataFrame(
        {c: [float(r) for r in ratings.values()] for c in DIMS},
        index=pd.Index(list(ratings.keys()), name="symbol"),
    )
    frame["ai_reviewed_at"] = "2026-09-01"
    return frame


# --- the blend --------------------------------------------------------------


def test_zero_weight_reproduces_the_stored_blend():
    frame = ars.attach_ai_scores(_frame(), _reviews({"AAA": 9, "BBB": 3}))
    scores = br.blend_scores(frame, quality_weight=0.6, ai_weight=0.0)
    # 0.6 * 80 + 0.4 * 60; BBB halved by confidence; CCC has no value score.
    assert scores.tolist() == pytest.approx([72.0, 31.0, 60.0])


def test_weight_mixes_the_ai_percentile_into_the_base():
    frame = ars.attach_ai_scores(_frame(), _reviews({"AAA": 9, "BBB": 3}))
    scores = br.blend_scores(frame, quality_weight=0.6, ai_weight=0.25)
    # AAA is the top of two reviewed companies (100th pct), BBB the bottom (50th).
    assert scores.iloc[0] == pytest.approx(0.75 * 72.0 + 0.25 * 100.0)
    # Confidence still applies to the whole blend, review included.
    assert scores.iloc[1] == pytest.approx((0.75 * 62.0 + 0.25 * 50.0) * 0.5)


def test_an_unreviewed_company_keeps_its_base_score():
    frame = ars.attach_ai_scores(_frame(), _reviews({"AAA": 9}))
    scores = br.blend_scores(frame, quality_weight=0.6, ai_weight=0.9)
    assert scores.iloc[2] == pytest.approx(60.0)


def test_weight_is_clamped_to_the_unit_interval():
    frame = ars.attach_ai_scores(_frame(), _reviews({"AAA": 9, "BBB": 3}))
    assert br.blend_scores(frame, 0.6, 5.0).tolist() == pytest.approx(
        br.blend_scores(frame, 0.6, 1.0).tolist()
    )
    assert br.blend_scores(frame, 0.6, -1.0).tolist() == pytest.approx(
        br.blend_scores(frame, 0.6, 0.0).tolist()
    )


def test_rerank_keeps_the_stored_rank_as_base_rank():
    frame = ars.attach_ai_scores(_frame(), _reviews({"AAA": 1, "BBB": 9, "CCC": 5}))
    ranked = br.rerank(frame, quality_weight=0.6, ai_weight=1.0)
    # With the review alone deciding, CCC's 67th percentile at full confidence
    # beats BBB's 100th halved to 50; AAA's 1/10 review sinks it to the 33rd.
    assert ranked["symbol"].tolist() == ["CCC", "BBB", "AAA"]
    assert ranked["rank"].tolist() == [1, 2, 3]
    assert ranked["base_rank"].tolist() == [3, 2, 1]


# --- turning the review into a score ----------------------------------------


def test_zero_scores_are_read_as_missing_not_as_the_worst_business():
    reviews = _reviews({"AAA": 8, "BBB": 6})
    reviews.loc["AAA", "ai_growth"] = np.nan  # what the loader makes of a 0
    frame = ars.attach_ai_scores(_frame(), reviews)
    assert frame.loc[0, "ai_rating"] == pytest.approx(8.0)
    assert np.isnan(frame.loc[2, "ai_score"])


def test_the_score_is_a_percentile_so_bunched_ratings_still_spread():
    frame = ars.attach_ai_scores(
        _frame(), _reviews({"AAA": 6.5, "BBB": 6.0, "CCC": 5.5})
    )
    assert frame["ai_score"].tolist() == pytest.approx([100.0, 200 / 3, 100 / 3])


def test_loader_takes_the_newest_scored_row_and_drops_zeros(tmp_path, monkeypatch):
    path = tmp_path / "screener.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE screener_cache (symbol TEXT, universe TEXT, ai_moat REAL, "
        "ai_financial_strength REAL, ai_predictability REAL, ai_growth REAL, "
        "updated_at TEXT)"
    )
    conn.executemany(
        "INSERT INTO screener_cache VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("AAA", "sp500", 4, 4, 4, 4, "2026-01-01"),
            ("AAA", "manual", 8, 8, 8, 0, "2026-09-01"),
            ("BBB", "sp500", None, None, None, None, "2026-09-01"),
        ],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("db_utils.get_global_screener_db_path", lambda: str(path))

    reviews = ars.load_ai_reviews(["aaa", "bbb"])

    assert list(reviews.index) == ["AAA"]
    assert reviews.loc["AAA", "ai_moat"] == 8.0
    assert np.isnan(reviews.loc["AAA", "ai_growth"])


def test_loader_failure_leaves_the_ranking_servable(monkeypatch):
    def boom():
        raise OSError("no screener cache")

    monkeypatch.setattr("db_utils.get_global_screener_db_path", boom)
    frame = ars.attach_ai_scores(_frame())
    assert frame["ai_score"].isna().all()
    assert br.blend_scores(frame, 0.6, 0.5).tolist() == pytest.approx(
        [72.0, 31.0, 60.0]
    )


# --- the served page --------------------------------------------------------


def test_served_page_reranks_before_filtering(monkeypatch):
    from server.routes import buffett_rank as route

    stored = _frame()

    class FakeStore:
        def get_run(self, run_id=None):
            return {"run_id": 7, "parameters": '{"quality_weight": 0.6}'}

        def get_scores_frame(self, run_id=None):
            return stored

    monkeypatch.setattr(route, "get_store", lambda: FakeStore())
    monkeypatch.setattr(
        "ai_review_scores.load_ai_reviews",
        lambda symbols=None: _reviews({"AAA": 1, "BBB": 9, "CCC": 5}),
    )

    page = route._blended_page(None, 1.0, 10, 0, None, None)
    assert [r["symbol"] for r in page["rows"]] == ["CCC", "BBB", "AAA"]
    assert page["ai_reviewed"] == 3

    # A search keeps the rank the company holds in the full re-ranked list.
    searched = route._blended_page(None, 1.0, 10, 0, None, "alpha")
    assert searched["total"] == 1
    assert searched["rows"][0]["rank"] == 3
    assert searched["rows"][0]["base_rank"] == 1

    banks = route._blended_page(None, 0.0, 10, 0, "bank", None)
    assert [(r["symbol"], r["rank"]) for r in banks["rows"]] == [("CCC", 2)]

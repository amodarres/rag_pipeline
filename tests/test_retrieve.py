"""Tests for Stage 2 (RETRIEVE). No network, no API key.

Chunks are built by hand so these tests don't depend on the chunker.
"""

import math

import pytest

from models import Chunk
from retrieve import _build_index, _cosine, _tokenize, retrieve, score_all


def make_chunks(*texts: str) -> list[Chunk]:
    return [Chunk(id=i, text=t, heading="", start_char=0) for i, t in enumerate(texts)]


TOPICS = make_chunks(
    "Cats are small carnivorous mammals that purr and sleep most of the day.",
    "To install Python on Linux, run sudo apt install python3 and check python3 --version.",
    "Tax returns are due on April 15. File an extension if you need more time.",
)


# --- the worked example from our TF-IDF walkthrough --------------------------

def test_worked_example_scores():
    chunks = make_chunks(
        "install python on linux with apt",
        "python is a snake found in asia",
        "install the printer driver on windows",
    )
    scores = {s.chunk.id: s.score for s in score_all("How do I install python on linux?", chunks)}
    assert scores[0] == pytest.approx(0.748, abs=0.001)
    assert scores[1] == pytest.approx(0.068, abs=0.001)
    assert scores[2] == pytest.approx(0.068, abs=0.001)
    # With the default 0.10 threshold, only the right chunk survives.
    assert [s.chunk.id for s in retrieve("How do I install python on linux?", chunks)] == [0]


def test_worked_example_idf_values():
    chunks = make_chunks(
        "install python on linux with apt",
        "python is a snake found in asia",
        "install the printer driver on windows",
    )
    idf, _ = _build_index(chunks)
    assert idf["python"] == pytest.approx(math.log(3 / 2))
    assert idf["linux"] == pytest.approx(math.log(3))


# --- behaviour ---------------------------------------------------------------

def test_finds_the_relevant_chunk():
    hits = retrieve("how do I install python", TOPICS)
    assert hits[0].chunk.id == 1


def test_top_k_is_respected():
    chunks = make_chunks(*[f"python note number {i}" for i in range(10)], "unrelated cooking text")
    assert len(retrieve("python note", chunks, top_k=3, min_score=0.0)) == 3


def test_threshold_is_respected():
    hits = retrieve("when are tax returns due", TOPICS, min_score=0.2)
    assert hits
    assert all(s.score >= 0.2 for s in hits)


def test_unrelated_question_returns_nothing():
    assert retrieve("quantum chromodynamics", TOPICS) == []


def test_results_are_sorted_best_first():
    hits = retrieve("install python linux tax", TOPICS, min_score=0.0)
    scores = [s.score for s in hits]
    assert scores == sorted(scores, reverse=True)


def test_ties_are_broken_by_chunk_id():
    chunks = make_chunks("apple banana", "cherry", "apple banana")
    ids = [s.chunk.id for s in score_all("apple", chunks)]
    assert ids == [0, 2, 1]


def test_deterministic():
    assert retrieve("install python", TOPICS) == retrieve("install python", TOPICS)


def test_heading_words_count_toward_the_score():
    chunks = [
        Chunk(0, "Run the package manager command.", "Installation > Linux", 0),
        Chunk(1, "Use the brew command.", "Installation > macOS", 0),
    ]
    assert retrieve("linux", chunks)[0].chunk.id == 0


def test_no_chunks_returns_empty():
    assert retrieve("anything", []) == []


def test_word_in_every_chunk_carries_no_weight():
    # Plain IDF: log(N/N) = 0. A single-chunk document can therefore never be
    # retrieved. This test documents that limitation on purpose.
    assert retrieve("python", make_chunks("python tips"), min_score=0.0)[0].score == 0.0


# --- helpers -----------------------------------------------------------------

def test_tokenize_lowercases_strips_punctuation_and_stop_words():
    assert _tokenize("How do I install Python on Linux?") == ["install", "python", "linux"]


def test_tokenize_splits_markdown_and_code():
    assert _tokenize("## Run `sudo apt install python3`.") == [
        "run", "sudo", "apt", "install", "python3",
    ]


def test_cosine_of_vector_with_itself_is_one():
    v = {"a": 1.0, "b": 2.0}
    assert _cosine(v, v) == pytest.approx(1.0)


def test_cosine_of_disjoint_vectors_is_zero():
    assert _cosine({"a": 1.0}, {"b": 1.0}) == 0.0


def test_cosine_with_empty_vector_is_zero():
    assert _cosine({}, {"a": 1.0}) == 0.0

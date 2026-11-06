"""Tests for Stage 3 (GENERATE).

Most tests are offline: they check the pure helpers, or use a FakeClient that
records requests and returns a canned reply. The tests marked `live` call the
real API and are skipped unless ANTHROPIC_API_KEY is set.
"""

import os
from types import SimpleNamespace

import pytest

import config
from generate import NO_ANSWER, build_prompt, generate, parse_citations
from models import Chunk, ScoredChunk


def scored(id: int, text: str, heading: str = "Section") -> ScoredChunk:
    return ScoredChunk(Chunk(id=id, text=text, heading=heading, start_char=0), score=0.5)


CHUNKS = [
    scored(2, "On Debian, install with sudo apt install driftwood.", "Installation > Linux"),
    scored(5, "On macOS, install with brew install driftwood.", "Installation > macOS"),
]


class FakeClient:
    """Stands in for anthropic.Anthropic(). Records every request it receives."""

    def __init__(self, reply: str = "Use apt [2].", stop_reason: str = "end_turn"):
        self.reply = reply
        self.stop_reason = stop_reason
        self.calls: list[dict] = []
        self.beta = SimpleNamespace(messages=self)  # so client.beta.messages.create works

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=self.reply),
            ],
        )


# --- build_prompt (pure) -----------------------------------------------------

def test_prompt_contains_every_chunk_id_heading_and_text():
    _, user = build_prompt("How do I install?", CHUNKS)
    for s in CHUNKS:
        assert f'id="{s.chunk.id}"' in user
        assert s.chunk.heading in user
        assert s.chunk.text in user


def test_prompt_ends_with_the_question():
    _, user = build_prompt("How do I install?", CHUNKS)
    assert user.endswith("Question: How do I install?")


def test_system_prompt_states_the_rules():
    system, _ = build_prompt("q", CHUNKS)
    assert "ONLY the sources" in system
    assert "[2]" in system        # shows the citation format
    assert NO_ANSWER in system    # tells Claude the exact fallback sentence


def test_double_quotes_in_headings_cannot_break_the_attribute():
    _, user = build_prompt("q", [scored(0, "text", 'The "best" > the rest')])
    assert "section=\"The 'best' > the rest\"" in user


# --- parse_citations (pure) --------------------------------------------------

def test_parse_citations_dedupes_in_order_of_appearance():
    assert parse_citations("A [3] and B [1][3].", {1, 3}) == [3, 1]


def test_parse_citations_accepts_comma_lists():
    assert parse_citations("Both say so [1, 3].", {1, 3}) == [1, 3]


def test_parse_citations_drops_made_up_ids():
    assert parse_citations("True [1], invented [9].", {1, 3}) == [1]


def test_parse_citations_ignores_non_numeric_brackets():
    assert parse_citations("See [docs] and [1].", {1}) == [1]


def test_parse_citations_with_no_citations():
    assert parse_citations(NO_ANSWER, {1}) == []


# --- generate with a fake client ---------------------------------------------

def test_empty_retrieval_says_i_dont_know_without_calling_claude():
    client = FakeClient()
    answer = generate("anything", [], client=client)
    assert answer.text == NO_ANSWER
    assert answer.cited_ids == []
    assert client.calls == []


def test_generate_returns_text_and_citations():
    answer = generate("How do I install on Debian?", CHUNKS, client=FakeClient("Use apt [2]."))
    assert answer.text == "Use apt [2]."
    assert answer.cited_ids == [2]
    assert answer.used_chunks == CHUNKS


def test_generate_sends_the_right_request():
    client = FakeClient()
    generate("How do I install?", CHUNKS, client=client, model="some-model", max_tokens=123)
    [request] = client.calls
    expected_system, expected_user = build_prompt("How do I install?", CHUNKS)
    assert request["model"] == "some-model"
    assert request["max_tokens"] == 123
    assert request["system"] == expected_system
    assert request["messages"] == [{"role": "user", "content": expected_user}]
    assert request["output_config"] == {"effort": config.EFFORT}
    assert request["fallbacks"] == "default"


def test_generate_raises_on_refusal():
    with pytest.raises(RuntimeError):
        generate("q", CHUNKS, client=FakeClient(stop_reason="refusal"))


# --- live (real API) ---------------------------------------------------------

live = pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")


@pytest.mark.live
@live
def test_live_answer_cites_a_source():
    answer = generate("How do I install Driftwood on Debian?", CHUNKS)
    assert answer.cited_ids, f"no valid citations in: {answer.text!r}"
    assert "apt" in answer.text


@pytest.mark.live
@live
def test_live_unanswerable_question_says_i_dont_know():
    answer = generate("What is the capital of France?", CHUNKS)
    assert NO_ANSWER in answer.text

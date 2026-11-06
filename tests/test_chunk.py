"""Tests for Stage 1 (CHUNK). No network, no API key."""

import pytest

from chunk import _split_sections, chunk_markdown


def numbered_words(n: int, prefix: str = "w") -> str:
    """'w0 w1 w2 ...' - unique words make overlap and coverage easy to check."""
    return " ".join(f"{prefix}{i}" for i in range(n))


# --- size and overlap --------------------------------------------------------

def test_no_chunk_exceeds_chunk_size():
    chunks = chunk_markdown(numbered_words(100), chunk_size=10, overlap=3)
    assert chunks
    assert all(len(c.text.split()) <= 10 for c in chunks)


def test_consecutive_chunks_overlap_by_exactly_overlap_words():
    chunks = chunk_markdown(numbered_words(50), chunk_size=10, overlap=3)
    for a, b in zip(chunks, chunks[1:]):
        assert a.text.split()[-3:] == b.text.split()[:3]


def test_windows_advance_by_chunk_size_minus_overlap():
    chunks = chunk_markdown(numbered_words(20), chunk_size=10, overlap=4)
    first_words = [c.text.split()[0] for c in chunks]
    assert first_words == ["w0", "w6", "w12"]  # step = 10 - 4 = 6


def test_zero_overlap_gives_disjoint_chunks():
    chunks = chunk_markdown(numbered_words(20), chunk_size=5, overlap=0)
    assert [c.text for c in chunks] == [
        "w0 w1 w2 w3 w4", "w5 w6 w7 w8 w9", "w10 w11 w12 w13 w14", "w15 w16 w17 w18 w19",
    ]


# --- coverage ----------------------------------------------------------------

def test_every_word_appears_in_some_chunk():
    doc = f"# A\n{numbered_words(37, 'a')}\n\n## B\n{numbered_words(23, 'b')}\n"
    chunks = chunk_markdown(doc, chunk_size=10, overlap=2)
    covered = {word for c in chunks for word in c.text.split()}
    expected = set(numbered_words(37, "a").split()) | set(numbered_words(23, "b").split())
    assert covered == expected


def test_last_window_does_not_repeat_a_fully_covered_tail():
    # 10 words, size 10: exactly one chunk, not a second chunk of leftovers.
    assert len(chunk_markdown(numbered_words(10), chunk_size=10, overlap=3)) == 1


# --- headings ----------------------------------------------------------------

def test_chunks_never_cross_heading_boundaries():
    doc = f"## Cats\n{numbered_words(15, 'cat')}\n## Dogs\n{numbered_words(15, 'dog')}\n"
    for c in chunk_markdown(doc, chunk_size=10, overlap=2):
        words = c.text.split()
        assert all(w.startswith("cat") for w in words) or all(w.startswith("dog") for w in words)


def test_heading_path_tracks_nesting():
    doc = (
        "# Guide\nintro\n"
        "## Install\ngeneral\n"
        "### Linux\napt\n"
        "### macOS\nbrew\n"
        "## Usage\nrun it\n"
    )
    headings = {c.text: c.heading for c in chunk_markdown(doc)}
    assert headings == {
        "intro": "Guide",
        "general": "Guide > Install",
        "apt": "Guide > Install > Linux",
        "brew": "Guide > Install > macOS",
        "run it": "Guide > Usage",
    }


def test_text_before_first_heading_has_empty_heading():
    chunks = chunk_markdown("preamble words\n# Title\nbody\n")
    assert (chunks[0].text, chunks[0].heading) == ("preamble words", "")


def test_hash_lines_inside_code_fences_are_not_headings():
    doc = "## Setup\n```bash\n# install it\npip install x\n```\nafter\n"
    chunks = chunk_markdown(doc)
    assert len(chunks) == 1
    assert chunks[0].heading == "Setup"
    assert "# install it" in chunks[0].text


def test_empty_sections_are_skipped():
    sections = _split_sections("# Empty\n\n## Also empty\n## Full\ncontent\n")
    assert [(h, b.strip()) for h, b, _ in sections] == [("Empty > Full", "content")]


# --- ids and offsets ---------------------------------------------------------

def test_ids_are_sequential_in_document_order():
    doc = f"# A\n{numbered_words(30)}\n# B\n{numbered_words(30)}\n"
    chunks = chunk_markdown(doc, chunk_size=10, overlap=2)
    assert [c.id for c in chunks] == list(range(len(chunks)))


def test_start_char_points_at_the_chunks_first_word():
    doc = f"# Title\n{numbered_words(25)}\n"
    for c in chunk_markdown(doc, chunk_size=10, overlap=2):
        first_word = c.text.split()[0]
        assert doc[c.start_char : c.start_char + len(first_word)] == first_word


# --- edge cases --------------------------------------------------------------

def test_empty_document_gives_no_chunks():
    assert chunk_markdown("") == []
    assert chunk_markdown("# Only a heading\n") == []


def test_short_document_gives_one_chunk():
    chunks = chunk_markdown("just a few words", chunk_size=10, overlap=2)
    assert [c.text for c in chunks] == ["just a few words"]


@pytest.mark.parametrize("size, overlap", [(10, 10), (10, 11), (10, -1), (0, 0)])
def test_invalid_size_or_overlap_raises(size, overlap):
    with pytest.raises(ValueError):
        chunk_markdown("some text", chunk_size=size, overlap=overlap)

"""Stage 2: RETRIEVE - given a question, find the top-k most relevant chunks.

Scoring is TF-IDF + cosine similarity, written out in plain Python:

  TF(word, text)  = how many times the word appears in that text
  IDF(word)       = log(N / DF), where N = number of chunks and
                    DF = number of chunks containing the word
  weight          = TF * IDF

Every chunk becomes a vector {word: weight}. The question becomes a vector the
same way: TF from the question itself, IDF from the chunks. The score is the
cosine of the angle between the two vectors, which is always in [0, 1].

Two consequences of the plain IDF formula worth knowing:
  - A word that appears in every chunk gets IDF = log(1) = 0 and adds nothing.
    In particular, a document with only ONE chunk can never score above 0.
  - A question word that appears in no chunk has no IDF and is ignored.
"""

import math
import re
import sys
from collections import Counter
from pathlib import Path

import config
from models import Chunk, ScoredChunk

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "how", "i", "if", "in", "is", "it", "its", "my", "of", "on",
    "or", "should", "so", "that", "the", "then", "there", "this", "to", "was",
    "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
}

Vector = dict[str, float]


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on anything that isn't a letter or digit, drop stop words.

    Chunks and questions MUST go through this same function, or words that
    should match won't.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOP_WORDS]


def _vectorize(tokens: list[str], idf: dict[str, float]) -> Vector:
    """Turn tokens into a {word: tf * idf} vector. Words with no IDF are dropped."""
    return {word: tf * idf[word] for word, tf in Counter(tokens).items() if word in idf}


def _build_index(chunks: list[Chunk]) -> tuple[dict[str, float], list[Vector]]:
    """Compute the IDF table and one TF-IDF vector per chunk.

    The heading is scored together with the text, so a chunk under
    "Installation > Linux" matches a question about installing on Linux even
    if the body never says "Linux".
    """
    tokenized = [_tokenize(f"{c.heading} {c.text}") for c in chunks]

    n = len(chunks)
    df = Counter(word for tokens in tokenized for word in set(tokens))
    idf = {word: math.log(n / count) for word, count in df.items()}

    vectors = [_vectorize(tokens, idf) for tokens in tokenized]
    return idf, vectors


def _cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity of two sparse vectors; 0.0 if either is all zeros."""
    dot = sum(weight * b[word] for word, weight in a.items() if word in b)
    norm_a = math.sqrt(sum(w * w for w in a.values()))
    norm_b = math.sqrt(sum(w * w for w in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def score_all(question: str, chunks: list[Chunk]) -> list[ScoredChunk]:
    """Score every chunk against the question, best first, with no cutoffs.

    Useful for seeing the full score distribution when tuning MIN_SCORE.
    """
    if not chunks:
        return []
    idf, vectors = _build_index(chunks)
    q = _vectorize(_tokenize(question), idf)
    scored = [ScoredChunk(c, _cosine(q, v)) for c, v in zip(chunks, vectors)]
    # Highest score first; ties broken by chunk id so results are deterministic.
    return sorted(scored, key=lambda s: (-s.score, s.chunk.id))


def retrieve(
    question: str,
    chunks: list[Chunk],
    top_k: int = config.TOP_K,
    min_score: float = config.MIN_SCORE,
) -> list[ScoredChunk]:
    """Return up to top_k chunks scoring >= min_score, best first.

    Returns [] when nothing clears the threshold. That is a real outcome, not
    an error: GENERATE then answers "I don't know" without calling Claude.
    """
    return [s for s in score_all(question, chunks) if s.score >= min_score][:top_k]


if __name__ == "__main__":
    # Usage: python retrieve.py data/sample.md "your question"
    from chunk import chunk_markdown  # only for this demo; the stage itself doesn't depend on it

    if len(sys.argv) < 3:
        sys.exit('usage: python retrieve.py <markdown file> "<question>"')
    chunks = chunk_markdown(Path(sys.argv[1]).read_text())
    hits = retrieve(sys.argv[2], chunks)
    print(f"Question tokens: {_tokenize(sys.argv[2])}\n")
    print("All scores (threshold {:.2f}, top_k {}):".format(config.MIN_SCORE, config.TOP_K))
    for s in score_all(sys.argv[2], chunks):
        mark = "*" if s in hits else " "
        print(f"  {mark} {s.score:.3f}  chunk {s.chunk.id}  {s.chunk.heading}")
    print("\n* = returned by retrieve()")

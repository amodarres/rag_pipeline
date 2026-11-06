# PLAN: Minimal RAG Pipeline

A learning project. Every stage is one small, readable module. No LangChain or
LlamaIndex, and no vector database. The only third-party runtime dependency is
the `anthropic` SDK, used for the GENERATE stage.

```
markdown file ──► CHUNK ──► list[Chunk] ──► RETRIEVE(question) ──► top-k chunks ──► GENERATE ──► cited answer
```

---

## 1. File structure

```
rag_pipeline/
├── PLAN.md
├── README.md
├── requirements.txt        # anthropic, pytest
├── config.py               # every tunable knob, in one place
├── chunk.py                # Stage 1: markdown -> list[Chunk]
├── retrieve.py             # Stage 2: question + chunks -> top-k ScoredChunk
├── generate.py             # Stage 3: question + chunks -> cited Answer (calls Claude)
├── models.py               # the dataclasses passed between stages
├── main.py                 # CLI that wires the three stages together
├── data/
│   └── sample.md           # a small markdown doc to play with
└── tests/
    ├── test_chunk.py
    ├── test_retrieve.py
    └── test_generate.py
```

Each stage module imports only `models.py` and `config.py`, never another
stage. `main.py` is the only file that knows about all three. That rule keeps
the stages separate: you can read, run, or test any one of them by itself.

---

## 2. Data passed between stages (`models.py`)

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Chunk:
    id: int              # position in the document: 0, 1, 2, ... (used for citations)
    text: str            # the chunk content
    heading: str         # nearest markdown heading above it, e.g. "Installation > Linux"
    start_char: int      # offset into the source doc, for debugging and tracing back

@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float         # relevance in [0, 1]; higher means more relevant

@dataclass(frozen=True)
class Answer:
    text: str            # the answer, with inline citations like [2]
    cited_ids: list[int] # chunk ids that actually appear as citations in `text`
    used_chunks: list[ScoredChunk]  # what was sent to Claude
```

These are plain dataclasses, so you can print them and see exactly what each
stage produced.

---

## 3. Stage signatures

### Stage 1: CHUNK (`chunk.py`)

```python
def chunk_markdown(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split a markdown document into small, overlapping chunks."""
```

The algorithm, kept simple on purpose:
1. Walk the document line by line and track the current heading path
   (`#`, `##`, `###`) so every chunk knows which section it came from.
2. Split each section into words and slide a window of `chunk_size` words,
   moving forward `chunk_size - overlap` words at each step.
3. Chunks never cross a heading boundary, so one chunk won't mix two topics.
   A short section becomes a single (smaller) chunk.

Sizes are counted in **words** rather than tokens. Words are easy to count by
eye, need no tokenizer, and are close enough for learning (≈1.3 tokens per
word in English).

Helper (private, still testable): `_split_sections(text) -> list[tuple[str, str, int]]`,
which returns `(heading_path, section_body, start_char)`.

### Stage 2: RETRIEVE (`retrieve.py`)

```python
def retrieve(
    question: str,
    chunks: list[Chunk],
    top_k: int = config.TOP_K,
    min_score: float = config.MIN_SCORE,
) -> list[ScoredChunk]:
    """Return up to top_k chunks scoring >= min_score, best first."""
```

**Scoring method: TF-IDF + cosine similarity, written in pure Python (~40 lines).**
- `_tokenize(text) -> list[str]`: lowercase, split on non-letters/digits,
  drop a small stop-word list.
- `_build_index(chunks) -> (idf: dict[str, float], vectors: list[dict[str, float]])`
- `_cosine(a: dict, b: dict) -> float`

Why TF-IDF rather than embeddings, for this first version:
- You can read every line of the scoring and work out a score by hand.
- It runs offline, has no extra API key or model download, and is fully deterministic, which keeps the tests simple.
- Cosine scores fall in [0, 1], so a **relevance threshold** has a clear meaning.

Its weakness is that it only matches exact words ("car" won't match
"automobile"). That makes a good follow-up exercise: keep `retrieve()`'s
signature, replace the scoring with embeddings, and compare results on the
same questions. No other stage changes.

Behaviour notes:
- Results are sorted by score, highest first. Ties are broken by chunk id so results are deterministic.
- If nothing clears `min_score`, it returns `[]`. That is a real outcome, not an error.

### Stage 3: GENERATE (`generate.py`)

```python
def build_prompt(question: str, chunks: list[ScoredChunk]) -> tuple[str, str]:
    """Return (system_prompt, user_message). Pure function, no API call."""

def parse_citations(answer_text: str, valid_ids: set[int]) -> list[int]:
    """Extract [n] citations from the answer, keeping only ids that were provided."""

def generate(
    question: str,
    chunks: list[ScoredChunk],
    client: anthropic.Anthropic | None = None,
    model: str = config.MODEL,
    max_tokens: int = config.MAX_TOKENS,
    effort: str = config.EFFORT,
) -> Answer:
    """Ask Claude to answer using ONLY the given chunks, citing them as [id]."""
```

Prompt design:
- The **system prompt** tells Claude to use only the provided sources, cite
  every claim with `[id]`, and reply exactly `I don't know based on the
  provided documents.` when the sources don't contain the answer.
- The **user message** wraps each chunk in tags so the boundaries are clear:
  ```
  <source id="3" section="Installation > Linux">
  ...chunk text...
  </source>

  Question: ...
  ```
- **Empty-retrieval short-circuit:** if `chunks` is empty, `generate()` returns
  the "I don't know" answer **without calling the API**. Claude can never
  answer from its own background knowledge because no sources passed the
  threshold.

`client` is injectable so tests can pass a fake client. If it is `None`, the
function creates `anthropic.Anthropic()`, which reads `ANTHROPIC_API_KEY` from
the environment.

### Wiring (`main.py`)

```python
def answer_question(doc_path: str, question: str) -> Answer:
    text = Path(doc_path).read_text()
    chunks = chunk_markdown(text)
    hits = retrieve(question, chunks)
    return generate(question, hits)
```

CLI: `python main.py data/sample.md "How do I install on Linux?" [--show-chunks]`.
`--show-chunks` prints the retrieved chunks with their scores before the
answer, so you can see what Claude was given.

---

## 4. Config knobs (`config.py`)

| Knob | Default | Stage | What it controls / what to experiment with |
|---|---|---|---|
| `CHUNK_SIZE` | `120` (words) | CHUNK | Larger chunks carry more context but match less precisely. Try 50 and 300. |
| `CHUNK_OVERLAP` | `30` (words) | CHUNK | Words repeated between neighbouring chunks, so a sentence cut at a boundary still appears whole in one of them. Must satisfy `0 <= overlap < chunk_size`. |
| `TOP_K` | `4` | RETRIEVE | Maximum number of chunks sent to Claude. More chunks improve recall but add noise and cost. |
| `MIN_SCORE` | `0.10` | RETRIEVE | Relevance threshold (cosine similarity, 0 to 1). Chunks below it are dropped even if fewer than `TOP_K` remain. Raise it to get more "I don't know" answers and fewer hallucinations. |
| `MODEL` | `"claude-sonnet-5-5"` | GENERATE | Which Claude model answers. |
| `MAX_TOKENS` | `16000` | GENERATE | Upper limit on Claude's reply. Claude's thinking counts toward it, so a small value can cut an answer off. |
| `EFFORT` | `"medium"` | GENERATE | How hard Claude thinks before answering (`low` to `max`). Lower is faster and cheaper. |

All knobs are plain module-level constants. Each function takes them as
keyword arguments with these defaults, so a test or experiment can override
one value without editing the file. `chunk_markdown` raises `ValueError` for
invalid combinations (e.g. `overlap >= chunk_size`).

---

## 5. Testing each stage on its own

Tests use `pytest`. Stages 1 and 2 need **no network and no API key**.

### `tests/test_chunk.py`: CHUNK
Uses small inline markdown strings, so every expected value can be checked by eye.
- **Size:** no chunk has more than `chunk_size` words.
- **Overlap:** for consecutive chunks in the same section, the last `overlap`
  words of chunk *n* equal the first `overlap` words of chunk *n+1*.
- **Coverage:** every word in the document appears in at least one chunk (nothing lost).
- **Heading boundaries:** a doc with two `##` sections never produces a chunk
  containing text from both; each chunk's `heading` is correct.
- **Edge cases:** empty doc returns `[]`; a doc shorter than `chunk_size` gives
  one chunk; `overlap >= chunk_size` raises `ValueError`.
- **Ids:** chunk ids are `0..n-1` in document order.

### `tests/test_retrieve.py`: RETRIEVE
Builds `Chunk` objects by hand (no dependence on the chunker) about clearly
different topics, e.g. "cats", "Python installation", "tax deadlines".
- **Finds the right chunk:** "how do I install python" ranks the Python chunk first.
- **top_k respected:** never returns more than `top_k`.
- **Threshold respected:** every returned score is `>= min_score`; an unrelated
  question ("quantum chromodynamics") returns `[]`.
- **Ordering:** scores are non-increasing.
- **Determinism:** the same input always gives the same output.
- **Unit checks:** `_cosine` of a vector with itself is 1.0 and of disjoint vectors is 0.0.

### `tests/test_generate.py`: GENERATE
Split into offline tests (always run) and one live test (opt-in).
- **`build_prompt` (pure):** every chunk id and its text appear in the user
  message; the question appears; the system prompt contains the "only use the
  sources" and citation rules.
- **`parse_citations` (pure):** `"A [1] and B [3][1]"` with valid ids `{1,3}`
  gives `[1, 3]`; made-up ids like `[9]` are dropped.
- **Empty chunks:** `generate(q, [], client=FakeClient())` returns the
  "I don't know" answer, and the fake client records **zero** calls.
- **Fake client:** a tiny `FakeClient` class whose `messages.create(...)`
  returns a canned response like `"Use apt [2]."`. Check that `Answer.cited_ids == [2]`
  and that the request contained the right model and sources.
- **Live smoke test** (`@pytest.mark.live`, skipped unless `ANTHROPIC_API_KEY`
  is set): ask a question answerable from one chunk, then check that the answer
  cites at least one valid id. Ask an unanswerable question, then check that it
  says "I don't know".

### Manual, stage-by-stage exploration
Each module gets a small `if __name__ == "__main__":` block so you can run a
stage by itself and look at its output:
- `python chunk.py data/sample.md`: prints each chunk with its id, heading, and word count.
- `python retrieve.py data/sample.md "your question"`: prints the top-k chunks with scores.
- `python generate.py`: sends a hard-coded question plus two hand-written chunks to Claude.

Run everything offline with `pytest -m "not live"`, or include the live test
with `pytest`.

---

## 6. Build order

1. `models.py` + `config.py`
2. `chunk.py` + `tests/test_chunk.py`, then review the output on `data/sample.md`
3. `retrieve.py` + `tests/test_retrieve.py`, then tune `MIN_SCORE` by checking real scores
4. `generate.py` + `tests/test_generate.py`
5. `main.py` end to end

Each step is a separate commit, so you can review one stage at a time.

---

## Changes made during implementation

The code follows this plan, with these differences:

1. **Model, `MAX_TOKENS` and a new `EFFORT` knob.** `MODEL` is `claude-sonnet-5-5`, as in the original plan (briefly `claude-opus-5-5` during implementation). `MAX_TOKENS` is 16000 instead of 1024, because Claude's thinking counts toward the limit and 1024 risks cutting answers off. `EFFORT` controls how much Claude thinks; set it to `"low"` to save cost.
2. **Refusal fallback.** `generate()` calls `client.beta.messages.create(..., fallbacks="default")`. If Claude's safety classifiers decline a request, the API retries it on a fallback model in the same call. If the whole chain still refuses, `generate()` raises `RuntimeError`.
3. **Headings are scored too.** `retrieve` indexes `heading + text` for each chunk, so a chunk under "Installation > Linux" matches "linux" even if its body doesn't say it.
4. **`score_all(question, chunks)` added to `retrieve.py`.** It returns every chunk's score with no cutoffs. `retrieve()` is now `score_all()` filtered by `min_score` and cut to `top_k`. `python retrieve.py ...` prints all scores, which is what you need to tune `MIN_SCORE`.
5. **Code fences in the chunker.** A `# comment` line inside a fenced code block is not treated as a heading.
6. **Known limitation (left in on purpose).** Plain IDF gives zero weight to a word that appears in every chunk. In `data/sample.md` the top heading "Driftwood User Guide" is part of every chunk's heading path, so "What is Driftwood?" retrieves nothing and the answer is "I don't know". Possible fixes: smoothed IDF `log((1 + N) / (1 + df)) + 1`, or index only the last heading level.

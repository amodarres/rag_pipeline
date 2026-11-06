# rag_pipeline

A minimal retrieval-augmented generation (RAG) pipeline, written from scratch in plain Python as a learning project. No LangChain, no LlamaIndex, no vector database.

```
markdown file ──► CHUNK ──► RETRIEVE(question) ──► GENERATE ──► cited answer
```

| Stage | File | What it does |
|---|---|---|
| CHUNK | `chunk.py` | Splits markdown into small, overlapping word windows that never cross a heading. |
| RETRIEVE | `retrieve.py` | Scores chunks against the question with TF-IDF and cosine similarity, written in about 40 lines of plain Python. |
| GENERATE | `generate.py` | Sends only the retrieved chunks to Claude and gets an answer that cites them as `[id]`. |

All tunable settings are in `config.py`. See `PLAN.md` for the design and the reasoning behind it.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # only needed for the GENERATE stage
```

## Usage

```bash
python main.py data/sample.md "How do I install on Ubuntu?"
python main.py data/sample.md "What happens if I edit a note on two computers?" --show-chunks
```

`data/sample.md` documents a made-up tool called Driftwood. Claude can't know anything about it from training, so every correct answer has to come from the document.

### Running one stage at a time

```bash
python chunk.py data/sample.md                       # print every chunk
python retrieve.py data/sample.md "your question"    # print every chunk's score
python generate.py                                   # send two hand-written chunks to Claude
```

## Tests

```bash
pytest -m "not live"   # offline: chunk, retrieve, and generate with a fake client
pytest                 # also runs the live Claude API tests if ANTHROPIC_API_KEY is set
```

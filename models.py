"""The data passed between the three stages.

CHUNK produces Chunk objects, RETRIEVE wraps them in ScoredChunk, and GENERATE
returns an Answer. They are plain frozen dataclasses so you can print them and
see exactly what each stage produced.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: int          # position in the document: 0, 1, 2, ... (used for citations)
    text: str        # the chunk content
    heading: str     # nearest markdown heading path above it, e.g. "Installation > Linux"
    start_char: int  # offset into the source doc, for tracing a chunk back to its origin


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float     # relevance in [0, 1]; higher means more relevant


@dataclass(frozen=True)
class Answer:
    text: str                       # the answer, with inline citations like [2]
    cited_ids: list[int]            # chunk ids that actually appear as citations in `text`
    used_chunks: list[ScoredChunk]  # what was sent to Claude

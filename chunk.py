"""Stage 1: CHUNK - split a markdown document into small, overlapping chunks.

How it works:
  1. Walk the document line by line, tracking the current heading path
     ("Installation > Linux") so each chunk knows which section it came from.
  2. Split each section's body into words and slide a window of `chunk_size`
     words over it, moving forward `chunk_size - overlap` words at each step.
  3. Chunks never cross a heading boundary, so one chunk never mixes two topics.

Sizes are counted in words (not LLM tokens): easy to count by eye, no tokenizer
needed. In English, one word is roughly 1.3 tokens.
"""

import re
import sys
from pathlib import Path

import config
from models import Chunk

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def _split_sections(text: str) -> list[tuple[str, str, int]]:
    """Split markdown into sections at headings.

    Returns a list of (heading_path, section_body, start_char), where
    start_char is the offset of the body in `text`. Heading lines themselves
    are not part of the body; they are recorded in heading_path instead.
    Sections with an empty body are dropped.
    """
    sections: list[tuple[str, str, int]] = []
    heading_stack: list[tuple[int, str]] = []  # (level, title), e.g. [(1, "Guide"), (2, "Install")]
    body_lines: list[str] = []
    body_start = 0  # char offset in `text` where the current section's body begins (just after its heading line)
    in_code_fence = False  # a "# comment" inside ``` is code, not a heading

    def flush() -> None:
        body = "".join(body_lines)
        if body.strip():
            path = " > ".join(title for _, title in heading_stack)
            sections.append((path, body, body_start))

    offset = 0
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_code_fence = not in_code_fence

        match = None if in_code_fence else HEADING_RE.match(line)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2)
            # Going to level n closes every open heading at level >= n.
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            body_lines = []
            body_start = offset + len(line)
        else:
            body_lines.append(line)
        offset += len(line)

    flush()
    return sections


def chunk_markdown(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split a markdown document into small, overlapping chunks."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if not 0 <= overlap < chunk_size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < chunk_size, got {overlap}")

    step = chunk_size - overlap
    chunks: list[Chunk] = []

    for heading, body, body_start in _split_sections(text):
        # Each match is one word plus its position, so chunks can point back to the source.
        words = list(re.finditer(r"\S+", body))

        for start in range(0, len(words), step):
            window = words[start : start + chunk_size]
            chunks.append(
                Chunk(
                    id=len(chunks),
                    text=" ".join(w.group() for w in window),
                    heading=heading,
                    start_char=body_start + window[0].start(),
                )
            )
            if start + chunk_size >= len(words):
                break  # this window reached the end of the section

    return chunks


if __name__ == "__main__":
    # Usage: python chunk.py data/sample.md
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.md"
    for c in chunk_markdown(Path(path).read_text()):
        print(f"--- chunk {c.id} | {c.heading} | {len(c.text.split())} words | char {c.start_char}")
        print(c.text)
        print()

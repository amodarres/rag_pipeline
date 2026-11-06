"""Wire the three stages together: CHUNK -> RETRIEVE -> GENERATE.

This is the only file that knows about all three stages.

Usage:
    python main.py data/sample.md "How do I install on Linux?"
    python main.py data/sample.md "How do I install on Linux?" --show-chunks
"""

import argparse
import sys
from pathlib import Path

import anthropic

from chunk import chunk_markdown
from generate import generate
from models import Answer
from retrieve import retrieve


def answer_question(doc_path: str, question: str) -> Answer:
    text = Path(doc_path).read_text()
    chunks = chunk_markdown(text)      # 1. CHUNK
    hits = retrieve(question, chunks)  # 2. RETRIEVE
    return generate(question, hits)    # 3. GENERATE


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about a markdown document.")
    parser.add_argument("doc", help="path to a markdown file")
    parser.add_argument("question", help="your question, in quotes")
    parser.add_argument("--show-chunks", action="store_true", help="print the retrieved chunks first")
    args = parser.parse_args()

    try:
        answer = answer_question(args.doc, args.question)
    except anthropic.AuthenticationError:
        sys.exit("Claude rejected the API key. Check ANTHROPIC_API_KEY and try again.")
    except TypeError as e:
        if "authentication" not in str(e):
            raise  # a real bug, not a missing key
        # anthropic.Anthropic() raises this when it finds no credentials at all.
        sys.exit("No Claude API key found. Set ANTHROPIC_API_KEY and try again.")
    except anthropic.APIError as e:
        sys.exit(f"Claude API error: {e}")

    if args.show_chunks:
        print(f"Retrieved {len(answer.used_chunks)} chunk(s):\n")
        for s in answer.used_chunks:
            print(f"[{s.chunk.id}] score {s.score:.3f} | {s.chunk.heading}")
            print(f"    {s.chunk.text}\n")
        print("-" * 60)

    print(answer.text)
    if answer.cited_ids:
        print("\nSources:")
        for s in answer.used_chunks:
            if s.chunk.id in answer.cited_ids:
                print(f"  [{s.chunk.id}] {s.chunk.heading}")


if __name__ == "__main__":
    main()

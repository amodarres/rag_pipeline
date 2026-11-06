"""Stage 3: GENERATE - call Claude with ONLY the retrieved chunks and get a cited answer.

Three pieces:
  build_prompt()     builds the system prompt and user message (no API call)
  parse_citations()  pulls [n] citations out of Claude's answer (no API call)
  generate()         the only function that talks to Claude

If retrieval found nothing, generate() answers "I don't know" WITHOUT calling
Claude, so Claude never gets a chance to answer from its own background knowledge.
"""

import re

import anthropic

import config
from models import Answer, Chunk, ScoredChunk

NO_ANSWER = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the sources provided in the user's message.

Rules:
- Use only information stated in the sources. Do not use outside knowledge, even if you are confident it is correct.
- Cite every claim with the id of the source that supports it, in square brackets, e.g. [2]. Cite multiple sources as [1][3].
- If the sources do not contain the answer, reply with exactly: {NO_ANSWER}
- If the sources answer only part of the question, answer that part and say what is missing.
- Be concise."""


def build_prompt(question: str, chunks: list[ScoredChunk]) -> tuple[str, str]:
    """Return (system_prompt, user_message). Pure function, no API call.

    Each chunk is wrapped in a <source> tag so Claude can see where one source
    ends and the next begins, and which id to cite.
    """
    sources = []
    for s in chunks:
        section = s.chunk.heading.replace('"', "'")  # a " would end the attribute early
        sources.append(f'<source id="{s.chunk.id}" section="{section}">\n{s.chunk.text}\n</source>')
    user_message = "\n\n".join(sources) + f"\n\nQuestion: {question}"
    return SYSTEM_PROMPT, user_message


def parse_citations(answer_text: str, valid_ids: set[int]) -> list[int]:
    """Extract [n] citations from the answer, keeping only ids that were provided.

    Accepts [2], [1][3] and [1, 3]. Returns ids in order of first appearance,
    without duplicates. Ids that weren't among the sources (made up) are dropped.
    """
    cited: list[int] = []
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer_text):
        for part in group.split(","):
            n = int(part)
            if n in valid_ids and n not in cited:
                cited.append(n)
    return cited


def generate(
    question: str,
    chunks: list[ScoredChunk],
    client: anthropic.Anthropic | None = None,
    model: str = config.MODEL,
    max_tokens: int = config.MAX_TOKENS,
    effort: str = config.EFFORT,
) -> Answer:
    """Ask Claude to answer using ONLY the given chunks, citing them as [id]."""
    if not chunks:
        return Answer(text=NO_ANSWER, cited_ids=[], used_chunks=[])

    if client is None:
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    system, user_message = build_prompt(question, chunks)

    # We use the beta endpoint only for `fallbacks="default"`: if Claude's safety
    # classifiers decline the request, the API retries it on a fallback model
    # inside the same call instead of returning a refusal.
    response = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_message}],
        output_config={"effort": effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to answer this question.")

    # The reply is a list of content blocks (thinking, text, ...); keep the text.
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    valid_ids = {s.chunk.id for s in chunks}
    return Answer(text=text, cited_ids=parse_citations(text, valid_ids), used_chunks=chunks)


if __name__ == "__main__":
    # Usage: python generate.py   (needs ANTHROPIC_API_KEY)
    # Sends a fixed question and two hand-written chunks, so you can see this
    # stage work without the other two.
    demo_chunks = [
        ScoredChunk(Chunk(0, "The office kettle is in the second-floor kitchen, left of the sink.", "Office > Kitchen", 0), 0.6),
        ScoredChunk(Chunk(1, "Visitors must sign in at the front desk before 9am.", "Office > Visitors", 0), 0.2),
    ]
    question = "Where is the kettle?"
    print("SYSTEM PROMPT:\n" + build_prompt(question, demo_chunks)[0])
    print("\nUSER MESSAGE:\n" + build_prompt(question, demo_chunks)[1])
    answer = generate(question, demo_chunks)
    print(f"\nANSWER:\n{answer.text}\n\nCITED: {answer.cited_ids}")

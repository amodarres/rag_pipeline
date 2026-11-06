"""Every tunable knob, in one place.

Each stage function takes these as keyword-argument defaults, so a test or an
experiment can override one value without editing this file.
"""

# --- CHUNK -------------------------------------------------------------------
CHUNK_SIZE = 120     # max words per chunk
CHUNK_OVERLAP = 30   # words shared between neighbouring chunks; must be < CHUNK_SIZE

# --- RETRIEVE ----------------------------------------------------------------
TOP_K = 4            # max chunks sent to Claude
MIN_SCORE = 0.10     # relevance threshold (cosine similarity, 0 to 1)

# --- GENERATE ----------------------------------------------------------------
MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000   # upper limit on Claude's reply (thinking counts toward it too)
EFFORT = "medium"    # how hard Claude thinks: "low" | "medium" | "high" | "xhigh" | "max"

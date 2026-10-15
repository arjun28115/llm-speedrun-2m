"""Byte-level BPE tokenizer, trained on train_corpus.txt only.

Interface contract (unchanged from the starter):
    load()          -> tokenizer object, called with NO arguments
    .encode(str)    -> list[int]
    .decode(list)   -> str
    .vocab_size     -> int

Losslessness: the base vocabulary is the 256 raw bytes, so every token id
expands to a byte string and decode(encode(t)) reproduces the exact original
bytes for ANY UTF-8 input, including text with characters never seen in
training. Pre-tokenization is coverage-checked (gaps between regex matches are
emitted as their own chunks), so nothing can be silently dropped.

Why BPE at all: the corpus is ~14% Devanagari by character but ~33% by BYTE
(every Devanagari codepoint is 3 UTF-8 bytes), and the dev file is ~44%
Devanagari by byte. A vocab-256 byte model spends three timesteps per Hindi
character and its 128-token window covers ~43 Hindi characters of context.
BPE fixes both the sequence length and the effective context window.

Devanagari gotcha: Python's \\w does NOT match Unicode combining marks
(categories Mn/Mc), so a plain GPT-2 style regex shatters "शब्द" into
["शब", "द"] and wastes merges re-learning the pieces. The word class below
explicitly includes the Devanagari matra/virama/nukta ranges and ZWJ/ZWNJ.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(HERE, "tokenizer.json")

# Combining marks + joiners that must stay attached to their base letter.
_MARKS = (
    "\u0900-\u0903"   # Devanagari signs (candrabindu, anusvara, visarga)
    "\u093A-\u094F"   # matras, virama, nukta
    "\u0951-\u0957"   # accents / additional vowel signs
    "\u0962\u0963"    # vocalic L matras
    "\u0300-\u036F"   # Latin combining diacriticals
    "\u200C\u200D"    # ZWNJ / ZWJ (present 65 / 526 times in the corpus)
)

SPLIT_PAT = re.compile(
    r"'(?:[sdmt]|ll|ve|re)"      # English contractions
    r"| ?\d{1,3}"                # digit runs, capped at 3 (keeps numbers generic)
    r"| ?[\w" + _MARKS + r"]+"   # one word, leading space attached
    r"| ?[^\s\w" + _MARKS + r"]+"  # punctuation run, leading space attached
    r"|\s+(?!\S)"                # trailing whitespace before EOF/newline
    r"|\s+"                      # any other whitespace run
)


def pretokenize(text):
    """Split text into chunks. Guaranteed lossless: ''.join(out) == text."""
    out, pos = [], 0
    for m in SPLIT_PAT.finditer(text):
        if m.start() > pos:              # regex gap -> emit verbatim
            out.append(text[pos:m.start()])
        out.append(m.group())
        pos = m.end()
    if pos < len(text):
        out.append(text[pos:])
    return out


class BPETokenizer:
    """merges: list of [a, b] pairs, in learned order. Token id = 256 + rank."""

    def __init__(self, merges):
        self.merges = {(int(a), int(b)): 256 + i for i, (a, b) in enumerate(merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        self.vocab_size = 256 + len(merges)
        self._cache = {}

    def _encode_chunk(self, chunk):
        hit = self._cache.get(chunk)
        if hit is not None:
            return hit
        ids = list(chunk.encode("utf-8", errors="surrogatepass"))
        while len(ids) >= 2:
            best, best_rank = None, None
            for pair in zip(ids, ids[1:]):
                r = self.merges.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            a, b = best
            new, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    new.append(best_rank)
                    i += 2
                else:
                    new.append(ids[i])
                    i += 1
            ids = new
        if len(self._cache) < 500_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text):
        out = []
        for chunk in pretokenize(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids):
        buf = b"".join(self.vocab[int(i)] for i in ids)
        return buf.decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "bpe",
                       "vocab_size": self.vocab_size,
                       "merges": [[a, b] for (a, b) in self.merges]}, f)


class ByteTokenizer:
    """Starter tokenizer, kept for the A/B ablation in RUNLOG.md."""
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8", errors="surrogatepass"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "byte"}, f)


def load(path=None):
    """Called with NO arguments by train.py and evaluate.py.

    Resolves tokenizer.json relative to THIS file so grading works from any cwd.
    Set SPEEDRUN_TOKENIZER=byte to force the byte baseline (ablation only).
    """
    if os.environ.get("SPEEDRUN_TOKENIZER", "").lower() == "byte":
        return ByteTokenizer()
    path = path or os.environ.get("SPEEDRUN_TOKENIZER_JSON") or DEFAULT_JSON
    if not os.path.exists(path):
        return ByteTokenizer()          # fallback: never crash the scorer
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    if blob.get("type") == "byte":
        return ByteTokenizer()
    return BPETokenizer(blob["merges"])

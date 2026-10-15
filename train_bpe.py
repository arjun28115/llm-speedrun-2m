"""Train the byte-level BPE on train_corpus.txt ONLY, and write tokenizer.json.

    python train_bpe.py --data data/train_corpus.txt --vocab 2048

Because BPE merges are hierarchical, the first K merges of a longer run are
exactly the tokenizer you would get by training to vocab 256+K. --report
therefore prints compression for every prefix size in one pass, which is how
the vocab-size table in RUNLOG.md was produced.

Algorithm: classic BPE over unique pre-token chunks weighted by frequency,
with an incremental pair index (pair -> set of word ids) so each merge only
rescans the words it actually touched.
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict

import tokenizer as tok_mod


def train(text, n_merges, verbose=True):
    freqs = Counter(tok_mod.pretokenize(text))
    words = [list(w.encode("utf-8", errors="surrogatepass")) for w in freqs]
    counts = list(freqs.values())
    if verbose:
        print(f"{len(words):,} unique chunks, {sum(counts):,} chunk occurrences")

    pair_count = Counter()
    pair_where = defaultdict(set)
    for wi, w in enumerate(words):
        c = counts[wi]
        for p in zip(w, w[1:]):
            pair_count[p] += c
            pair_where[p].add(wi)

    merges, t0 = [], time.time()
    for step in range(n_merges):
        if not pair_count:
            break
        pair = max(pair_count, key=pair_count.get)
        if pair_count[pair] < 2:
            break
        new_id = 256 + len(merges)
        merges.append(list(pair))
        a, b = pair

        for wi in list(pair_where[pair]):
            w, c = words[wi], counts[wi]
            for p in zip(w, w[1:]):                # retract old contribution
                pair_count[p] -= c
                if pair_count[p] <= 0:
                    del pair_count[p]
            new, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    new.append(new_id)
                    i += 2
                else:
                    new.append(w[i])
                    i += 1
            words[wi] = new
            for p in zip(new, new[1:]):            # add new contribution
                pair_count[p] += c
                pair_where[p].add(wi)
        pair_where.pop(pair, None)

        if verbose and (step + 1) % 500 == 0:
            print(f"  merge {step + 1:5d}  vocab {new_id + 1:5d}  "
                  f"({time.time() - t0:.0f}s)")
    return merges


def compression(merges, texts, sizes):
    rows = []
    for k in sizes:
        t = tok_mod.BPETokenizer(merges[:k - 256])
        row = {"vocab": k}
        for name, txt in texts.items():
            n_bytes = len(txt.encode("utf-8"))
            n_tok = len(t.encode(txt))
            row[name] = n_bytes / n_tok
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_corpus.txt")
    ap.add_argument("--dev", default="data/dev_eval.txt")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--out", default="tokenizer.json")
    ap.add_argument("--report", action="store_true",
                    help="print bytes/token for every power-of-two prefix")
    args = ap.parse_args()

    text = open(args.data, encoding="utf-8").read()
    assert "".join(tok_mod.pretokenize(text[:200_000])) == text[:200_000], \
        "pre-tokenizer lost characters"

    merges = train(text, args.vocab - 256)
    t = tok_mod.BPETokenizer(merges)
    t.save(args.out)
    print(f"saved {args.out}  vocab={t.vocab_size}")

    dev = open(args.dev, encoding="utf-8").read()
    assert t.decode(t.encode(dev)) == dev, "ROUND-TRIP FAILED on dev"
    assert t.decode(t.encode(text[:500_000])) == text[:500_000], "ROUND-TRIP FAILED on train"
    print("round-trip lossless: OK")

    if args.report:
        sizes = [s for s in (512, 1024, 2048, 4096, 8192, 16384)
                 if s <= t.vocab_size]
        print(f"\n{'vocab':>7} {'train B/tok':>12} {'dev B/tok':>11}")
        for r in compression(merges, {"train": text[:1_000_000], "dev": dev}, sizes):
            print(f"{r['vocab']:>7} {r['train']:>12.2f} {r['dev']:>11.2f}")


if __name__ == "__main__":
    main()

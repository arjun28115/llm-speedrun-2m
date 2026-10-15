"""Trainer for the 2,000-step speedrun.

    python train.py --data data/train_corpus.txt --steps 2000 --out ckpt.pt

HARD CAPS (asserted below): <= 2000 optimizer steps, <= 2,000,000 params,
train_corpus.txt only, pure PyTorch/numpy/stdlib, CPU.

What changed vs the starter trainer, and why:
  1. AdamW, betas (0.9, 0.95), wd 0.1 on matrices only. The starter used plain
     Adam with beta2=0.999, whose second-moment estimate needs ~1/(1-b2)=1000
     steps to warm up -- half of our entire budget.
  2. Linear warmup (5%) then cosine decay to 10% of peak. A constant LR spends
     the last 500 steps bouncing around the minimum instead of settling.
  3. Peak LR 3e-3, ~10x the starter's 3e-4. 3e-4 is a 100M-parameter setting;
     a 1.8M-parameter model under a 2000-step cap is nowhere near its stability
     limit, and LR is the single highest-leverage knob here.
  4. Gradient clipping at 1.0 -- cheap insurance that lets (3) be aggressive.
  5. tokens/step is the real budget. Steps are capped, so batch*block is the
     only way to see more data; the starter saw 8*128*2000 = 2.05M tokens,
     less than one epoch of the corpus.
  6. Token ids are cached to disk, so re-running an experiment does not pay
     the BPE encode cost twice.
  7. Dev bpb is measured DURING the run with the official scorer, so the
     RUNLOG numbers come from the same code path the graders use.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

import tokenizer as tokenizer_mod
from evaluate import bits_per_byte
from model import GPT, Config

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000
HERE = os.path.dirname(os.path.abspath(__file__))


def load_ids(path, tok, cache_dir=os.path.join(HERE, "experiments", "cache")):
    os.makedirs(cache_dir, exist_ok=True)
    stamp = f"{os.path.basename(path)}.v{tok.vocab_size}.{int(os.path.getmtime(path))}.npy"
    cache = os.path.join(cache_dir, stamp)
    text = open(path, encoding="utf-8").read()
    if os.path.exists(cache):
        return np.load(cache), len(text.encode("utf-8"))
    t0 = time.time()
    ids = np.asarray(tok.encode(text), dtype=np.uint16 if tok.vocab_size <= 65535
                     else np.int32)
    np.save(cache, ids)
    print(f"tokenized in {time.time() - t0:.0f}s -> {cache}")
    return ids, len(text.encode("utf-8"))


def lr_at(step, total, peak, warmup_frac, floor_frac):
    warm = max(1, int(total * warmup_frac))
    if step <= warm:
        return peak * step / warm
    import math
    prog = (step - warm) / max(1, total - warm)
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def make_optimizer(model, args):
    decay, no_decay, matrices = [], [], []
    for name, p in model.named_parameters():
        if p.ndim < 2:
            no_decay.append(p)
        elif args.opt == "muon" and "tok_emb" not in name:
            matrices.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": args.wd},
              {"params": no_decay, "weight_decay": 0.0}]
    adamw = torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2),
                              eps=1e-8)
    if args.opt == "muon":
        from optim_muon import Muon
        return [adamw, Muon(matrices, lr=args.muon_lr, momentum=0.95)]
    return [adamw]


@torch.no_grad()
def dev_bpb(model, cfg, tok, text):
    model.eval()
    bpb, _, _ = bits_per_byte(model, cfg, tok, text)
    model.train()
    return bpb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_corpus.txt")
    ap.add_argument("--dev", default="data/dev_eval.txt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--n_layer", type=int, default=None)
    ap.add_argument("--n_head", type=int, default=None)
    ap.add_argument("--n_embd", type=int, default=None)
    ap.add_argument("--n_kv_head", type=int, default=None,
                    help="1 = multi-query attention; default = n_head (MHA)")
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--no_tie", action="store_true")
    ap.add_argument("--opt", choices=["adamw", "muon"], default="adamw")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--lr_floor", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.0, help="0 disables")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_chars", type=int, default=40000,
                    help="dev prefix used for mid-run evals (full file at end)")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} optimizer steps"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cpu"

    tok = tokenizer_mod.load()
    ids_np, n_bytes = load_ids(args.data, tok)
    ids = torch.from_numpy(ids_np.astype(np.int64))
    print(f"corpus: {n_bytes:,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size}, {n_bytes / len(ids):.2f} bytes/token)")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    for k, v in [("block_size", args.block), ("n_layer", args.n_layer),
                 ("n_head", args.n_head), ("n_embd", args.n_embd),
                 ("dropout", args.dropout)]:
        if v is not None:
            setattr(cfg, k, v)
    if args.n_head is not None:
        cfg.n_kv_head = args.n_head
    if args.n_kv_head is not None:
        cfg.n_kv_head = args.n_kv_head
    if args.no_tie:
        cfg.tie_weights = False

    model = GPT(cfg).to(device)
    n = model.n_params()
    tokens_per_step = args.batch * args.accum * cfg.block_size
    print(f"model: {n:,} params  |  {tokens_per_step:,} tokens/step  |  "
          f"{tokens_per_step * args.steps / len(ids):.2f} epochs")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params (got {n:,})"

    opts = make_optimizer(model, args)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} \
        if args.ema else None
    dev_text_short = open(args.dev, encoding="utf-8").read()[:args.eval_chars]
    dev_text_full = open(args.dev, encoding="utf-8").read()

    hi = len(ids) - cfg.block_size - 1
    ar = torch.arange(cfg.block_size + 1)
    model.train()
    t0, losses, evals = time.time(), [], []
    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.steps, args.lr, args.warmup, args.lr_floor)
        for i, o in enumerate(opts):
            scale = 1.0 if i == 0 else args.muon_lr / args.lr   # opts[1] is Muon
            for g in o.param_groups:
                g["lr"] = lr * scale

        for o in opts:
            o.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(args.accum):
            ix = torch.randint(hi, (args.batch,))
            chunk = ids[ix[:, None] + ar]            # (B, block+1)
            x, y = chunk[:, :-1], chunk[:, 1:]
            _, loss = model(x, y)
            (loss / args.accum).backward()
            tot += loss.item() / args.accum
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for o in opts:
            o.step()
        if ema is not None:
            with torch.no_grad():
                d = min(args.ema, (1 + step) / (10 + step))
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(d).add_(v.detach(), alpha=1 - d)
        losses.append(tot)

        if step % args.log_every == 0 or step == 1:
            w = losses[-args.log_every:]
            print(f"step {step:5d}  loss {sum(w) / len(w):.4f}  lr {lr:.2e}  "
                  f"({(time.time() - t0) / step * 1000:.0f} ms/step)")
        if args.eval_every and step % args.eval_every == 0 and step != args.steps:
            b = dev_bpb(model, cfg, tok, dev_text_short)
            evals.append([step, b])
            print(f"    dev bpb @{step}: {b:.4f}  (first {args.eval_chars} chars)")

    if ema is not None:
        model.load_state_dict(ema)
    final_bpb = dev_bpb(model, cfg, tok, dev_text_full)
    took = time.time() - t0
    print(f"FINAL dev bpb (full file): {final_bpb:.4f}   [{took:.0f}s]")

    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_") and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses,
                "dev_bpb": final_bpb,
                "args": vars(args)}, args.out)
    print(f"saved {args.out}")

    row = {"tag": args.tag, "dev_bpb": round(final_bpb, 4), "params": n,
           "steps": args.steps, "vocab": tok.vocab_size,
           "block": cfg.block_size, "batch": args.batch, "accum": args.accum,
           "n_layer": cfg.n_layer, "n_embd": cfg.n_embd, "opt": args.opt,
           "lr": args.lr, "wd": args.wd, "warmup": args.warmup,
           "dropout": cfg.dropout, "ema": args.ema,
           "tokens_per_step": tokens_per_step,
           "final_train_loss": round(sum(losses[-50:]) / len(losses[-50:]), 4),
           "seconds": round(took)}
    os.makedirs(os.path.join(HERE, "experiments"), exist_ok=True)
    csv_path = os.path.join(HERE, "experiments", "results.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(row))
        if new:
            wtr.writeheader()
        wtr.writerow(row)
    print(json.dumps(row))


if __name__ == "__main__":
    main()

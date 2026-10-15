# 2,000-Step LLM Speedrun (≤2M parameters, CPU only)

A decoder-only language model trained from scratch on a mixed English + Hindi corpus, under the constraints of the Plivo model-optimization challenge:

- at most **2,000 optimizer steps**
- at most **2,000,000 parameters**
- **CPU only**, pure PyTorch + NumPy + standard library
- training data: `train_corpus.txt` only (the tokenizer is trained on it too)

The score is **bits per byte (BPB)** on held-out text. Lower is better.

## Result

| Model | Vocab | Params | Steps | Dev BPB |
|---|---|---|---|---|
| Baseline submission ([Priyanshu Pandey](https://github.com/priyanshup0046/plivo-ml--240809-)) | 2,048 | 1,837,616 | 2,000 | 1.6779 |
| **This repo** | **2,944** | **1,995,312** | **2,000** | **1.6678** |

The only change between the two rows is the tokenizer vocabulary (2,048 → 2,944), with the same 4-layer, 176-dim model, optimizer, schedule and seed. The larger vocabulary compresses the corpus to 3.65 bytes per token, so each of the 2,000 steps sees more text, and the extra embedding rows still fit under the 2M-parameter cap (1,995,312).

Both numbers were measured with the unmodified `evaluate.py` on `data/dev_eval.txt`, on the same machine.

## Ablations (600-step proxy runs, vocab 2,944)

Each run changes one thing from the 4L × 176 AdamW configuration. 600-step runs use the same cosine schedule scaled to 600 steps, so they rank configurations rather than predict final scores.

| Run | Change | Params | Dev BPB @600 |
|---|---|---|---|
| A0 | reference: MHA, AdamW, 4L × 176 | 1,995,312 | 2.0195 |
| A1 | multi-query attention (1 KV head) | 1,809,456 | **1.9983** |
| A2 | deeper and narrower: 6L × 144 | 1,918,800 | 2.0554 |
| A3 | dropout 0.1 | 1,995,312 | 2.0466 |
| A4 | Muon on hidden matrices | 1,995,312 | **1.7652** |

What this says:
- **Muon** converges much faster than AdamW over the first 600 steps (1.765 vs 2.020). It was not run for the full 2,000 steps here, so the submitted checkpoint is still the AdamW one.
- **MQA** is slightly better than full multi-head attention while using about 9% fewer parameters, which frees budget for other uses.
- **Deeper, narrower** (6L × 144) and **dropout 0.1** both hurt. With about 4 passes over a small corpus, the model is under-trained rather than overfitting, so regularisation costs more than it saves.

Raw numbers for every run are in `experiments/results.csv`.

## Model

- Byte-level BPE tokenizer with a Devanagari-aware pre-tokenizer (combining marks stay attached to their base letter), byte fallback for any UTF-8 input
- 4-layer decoder-only Transformer, d = 176, 4 heads, context 256
- RMSNorm, RoPE, SwiGLU feed-forward, tied input/output embeddings
- AdamW (β = 0.9, 0.95), weight decay 0.1 on matrices, peak LR 3e-3, 5% warmup then cosine decay to 10%, gradient clipping at 1.0
- Batch 16 × 256 tokens per step

## Reproduce

The challenge data (`train_corpus.txt`, `dev_eval.txt`) is not redistributed here. Put both files in `data/`.

```bash
pip install -r requirements.txt

# 1. tokenizer (vocab 2,944), trained on the training corpus only
python train_bpe.py --data data/train_corpus.txt --vocab 2944 --out tokenizer.json

# 2. full 2,000-step run (about 20 minutes on 2 CPU cores)
python train.py --steps 2000 --batch 16 --block 256 --out ckpt.pt --tag final

# 3. score
python evaluate.py --checkpoint ckpt.pt --text_file data/dev_eval.txt

# ablations, e.g. MQA or Muon at 600 steps
python train.py --steps 600 --n_kv_head 1 --tag mqa --out /tmp/mqa.pt
python train.py --steps 600 --opt muon --muon_lr 0.02 --tag muon --out /tmp/muon.pt
```

`ckpt.pt` and `tokenizer.json` in this repo are the final submission and give 1.6678 BPB with step 3.

## Files

| File | Purpose |
|---|---|
| `model.py` | Transformer (RMSNorm, RoPE, SwiGLU, optional grouped/multi-query attention) |
| `tokenizer.py` | byte-level BPE tokenizer, loads `tokenizer.json` |
| `train_bpe.py` | trains the BPE tokenizer |
| `train.py` | training loop; asserts the step and parameter caps |
| `optim_muon.py` | Muon optimizer |
| `evaluate.py` | the challenge's official BPB scorer |
| `RUNLOG.md` | one entry per run: hypothesis, change, result, conclusion |
| `NOTES.md` | short summary of the final configuration |

## Credits

This project builds directly on **[Priyanshu Pandey](https://github.com/priyanshup0046)**'s submission, [`plivo-ml--240809-`](https://github.com/priyanshup0046/plivo-ml--240809-), which provides the BPE tokenizer, the RMSNorm/RoPE/SwiGLU model, the AdamW training setup, the Muon optimizer and the 1.6779 baseline. This repo adds the 2,944-token vocabulary run, a multi-query attention flag in `train.py`, and the ablation study above. The starter code and data come from the Plivo challenge handout.

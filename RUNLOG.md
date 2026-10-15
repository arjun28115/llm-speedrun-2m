# RUNLOG

All runs use the same data, seed (1337), batch 16 × 256 tokens, AdamW (unless stated), peak LR 3e-3 with 5% warmup and cosine decay. Dev BPB is the full `dev_eval.txt` scored by the official `evaluate.py`.

## R0. Reference: baseline submission, vocab 2,048

- **What:** the 2,048-vocab, 4L × 176 checkpoint from [Priyanshu Pandey's submission](https://github.com/priyanshup0046/plivo-ml--240809-), re-scored on this machine.
- **Result:** 1.6779 BPB, 1,837,616 params, 2,000 steps.
- **Conclusion:** reproduces the published number exactly; this is the bar to beat.

## R1. Larger vocabulary: 2,048 → 2,944 (final submission)

- **Hypothesis:** the cap is on steps, not wall-clock, so tokens per step is the real budget. A larger vocabulary packs more bytes into each token, so each step sees more text. At d = 176 the extra 896 embedding rows cost 157,696 params, which still fits under 2M (1,995,312).
- **Change:** retrained BPE to 2,944 tokens on `train_corpus.txt`; nothing else changed.
- **Effect on data:** 3.65 bytes per token on the training corpus.
- **Result:** **1.6678 BPB** (was 1.6779), 1,995,312 params, 2,000 steps, 1,149 s.
- **Conclusion:** a 0.6% improvement from the tokenizer alone, using the parameter headroom the baseline left unused. Kept.

## A0. 600-step proxy reference (vocab 2,944)

- **Why:** a 2,000-step run takes about 20 minutes here, so the ablations below use 600-step runs (same schedule, scaled) to rank changes one at a time.
- **Result:** 2.0195 BPB.

## A1. Multi-query attention

- **Hypothesis:** a 176-dim model with 4 heads may not need separate keys and values per head; sharing one KV head frees parameters.
- **Change:** `--n_kv_head 1` (new flag in `train.py`).
- **Result:** 1.9983 BPB with 1,809,456 params (185,856 fewer).
- **Conclusion:** slightly better than MHA with about 9% fewer parameters. The freed budget could fund a wider model or larger vocabulary; not yet tested at 2,000 steps.

## A2. Deeper and narrower: 6L × 144

- **Hypothesis:** more layers at a similar parameter count might model longer-range structure better.
- **Result:** 2.0554 BPB (worse than A0).
- **Conclusion:** at this scale and step budget, width per layer matters more than depth. Rejected.

## A3. Dropout 0.1

- **Hypothesis:** regularisation might help if the model overfits.
- **Result:** 2.0466 BPB (worse than A0).
- **Conclusion:** 2,000 steps × 4,096 tokens is about 4 passes over the corpus and the model is still under-fitting, so dropout only slows learning. Rejected.

## A4. Muon optimizer

- **Hypothesis:** Muon's orthogonalised updates on the hidden weight matrices should converge faster per step than AdamW, which matters when steps are capped.
- **Change:** `--opt muon --muon_lr 0.02` (Muon on hidden matrices, AdamW on embeddings and norms).
- **Result:** 1.7652 BPB at 600 steps, against 2.0195 for AdamW.
- **Conclusion:** the largest single effect measured at 600 steps. The obvious next run is a full 2,000-step Muon run at vocab 2,944; it was not part of this submission.

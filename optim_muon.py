"""Muon: momentum SGD whose update is orthogonalised with Newton-Schulz.

Pure PyTorch, no custom kernels. Intended for 2-D hidden weight matrices only;
embeddings, the tied output head and all 1-D parameters (norm gains) stay on
AdamW, because orthogonalising a vocab-shaped matrix is both expensive and
wrong (its rows are not a shared feature basis).

Idea: Adam's per-coordinate scaling lets a few directions dominate the update.
Replacing the momentum buffer M by its nearest orthogonal matrix makes every
singular direction take an equal-size step, which empirically buys a lot of
progress in the first couple of thousand steps -- exactly our regime.

Quintic Newton-Schulz coefficients follow Keller Jordan's public write-up of
the nanoGPT speedrun; the implementation here is written from the definition.
"""
import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      nesterov=nesterov, ns_steps=ns_steps,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(g)
                buf = st["m"]
                buf.mul_(group["momentum"]).add_(g)
                upd = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                upd = zeropower_via_newtonschulz5(upd, group["ns_steps"])
                # keep the RMS of the update comparable to Adam's ~1 per entry
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(upd.type_as(p), alpha=-group["lr"] * scale)

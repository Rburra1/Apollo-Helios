"""
Muon optimizer.

Muon = Adam-like momentum on flat gradients, then Newton-Schulz orthogonalization
of the matrix-shaped update before applying. Plus AdamW-style decoupled weight decay.

Reference: Keller Jordan et al. (https://kellerjordan.github.io/posts/muon/)
Scaling fixes: Moonshot AI "Muon is Scalable for LLM Training" (arXiv:2502.16982)

Usage pattern (this is the standard one):
    Use Muon for 2D matrix params in transformer blocks (qkv, proj, w1/w2/w3, fc1/fc2).
    Use AdamW for everything else (embeddings, output head, norms, biases, scalars).

The split is handled by partition_params() below. The training loop instantiates
both optimizers and steps both each iteration.

MPS notes:
    Newton-Schulz uses bf16 matmuls on CUDA in the reference impl. On MPS we use fp32
    because bf16 matmul on Apple Silicon is supported but flaky on smaller models.
    The performance hit is modest (~10-15%) at 25M-param scale.
"""

import torch
from torch.optim.optimizer import Optimizer


# Newton-Schulz coefficients optimized in Keller Jordan's writeup.
# 5 iterations is enough for a good orthogonalization at fp32.
NS_COEFFS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = NS_STEPS) -> torch.Tensor:
    """
    Approximate orthogonalization via Newton-Schulz iteration.

    Given gradient matrix G of shape (m, n), compute U @ V.T where G ~ U @ S @ V.T.
    The result has the same shape as G but with all singular values clamped to ~1.

    Works for any 2D shape; we transpose if m > n so the polynomial works in the
    smaller dim.
    """
    assert G.dim() == 2
    a, b, c = NS_COEFFS
    X = G.float()  # fp32 for numerical stability on MPS

    # Spectral normalize so largest singular value is ~1 before the polynomial
    X = X / (X.norm() + 1e-7)

    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon optimizer for matrix-shaped parameters.

    Args:
        params: iterable of 2D tensor parameters
        lr: learning rate (Muon scale; usually similar to AdamW lr)
        momentum: SGD momentum (default 0.95)
        nesterov: whether to use Nesterov momentum (default True)
        weight_decay: AdamW-style decoupled weight decay (default 0.1)
        ns_steps: Newton-Schulz iterations (default 5)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.1,
        ns_steps: int = NS_STEPS,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            wd = group['weight_decay']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.dim() != 2:
                    raise ValueError(
                        f"Muon requires 2D params; got shape {p.shape}. "
                        f"Use AdamW for non-matrix params."
                    )

                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                update = g + momentum * buf if nesterov else buf

                # Orthogonalize the update via Newton-Schulz
                ortho = newton_schulz(update, steps=ns_steps)

                # Per-parameter scale adjustment from Moonshot's scaling paper:
                # scale by max(1, m/n)^0.5 to match per-element update magnitude
                # across different matrix shapes.
                fan_out, fan_in = p.shape
                scale = max(1.0, fan_out / fan_in) ** 0.5

                # Decoupled weight decay (AdamW style)
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                p.add_(ortho, alpha=-lr * scale)

        return loss


# ----------------------------------------------------------------------
# Param partitioning helper
# ----------------------------------------------------------------------

def partition_params(model: torch.nn.Module):
    """
    Split params into (muon_params, adamw_params).

    Muon: 2D weight tensors inside transformer blocks. Specifically the
        attention QKV and proj, and FFN matrices.
    AdamW: embeddings (incl. tied output head, since its weight IS tok_emb.weight),
        norm scales, biases, and any 1D params.
    """
    muon_params = []
    adamw_params = []
    seen_ids = set()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen_ids:
            continue  # weight-tied; only count once
        seen_ids.add(id(p))

        is_block_matrix = (
            ('blocks.' in name)
            and p.dim() == 2
            and ('norm' not in name)
        )

        if is_block_matrix:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    return muon_params, adamw_params


if __name__ == '__main__':
    # Quick sanity check on a tiny matrix
    G = torch.randn(64, 128)
    O = newton_schulz(G)
    # Check that singular values are roughly ~1
    s = torch.linalg.svdvals(O)
    print(f"Orthogonalized singular values: min={s.min():.3f}, max={s.max():.3f}, mean={s.mean():.3f}")

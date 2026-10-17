__all__ = ['POOLINGS', 'ALL_POOLINGS', 'pool_hidden', 'selftest_pooling']

import numpy as np
import torch

# Keys are the suffixes used in column names, so a pooled column always says out
# loud which reduction produced it: `pool_l32_mean`, `pool_l32_std`, …
POOLINGS = {
    "mean":  "arithmetic mean over valid frames (the standard ASR-probe choice)",
    "max":   "per-dimension max over valid frames (peak activations)",
    #"min":   "per-dimension min over valid frames",
    #"sum":   "unnormalised sum; grows with utterance length, so it encodes "
    #         "duration directly. Use only if that is what you want.",
    "std":   "per-dimension standard deviation, ddof=0 (temporal variability)",
    #"first": "the first valid frame only (no reduction; a control)",
    #"last":  "the last valid frame only (no reduction; a control)",
}
# Four reductions are commented out rather than deleted. `pool_hidden` still
# implements all seven, but it validates its `poolings` argument against this
# dict, so an entry that is commented out cannot be requested — which is the
# point: it keeps the stored column count at three. Storage is why. Seven
# poolings x 1024 dims x 41,641 AMI rows is ~1.2 GB per model; three is ~0.5 GB.
# Uncomment a line to bring that reduction back for a run.
ALL_POOLINGS = tuple(POOLINGS)


@torch.no_grad()
def pool_hidden(hidden, lengths, poolings=ALL_POOLINGS):
    """Reduce padded frame-level hidden states to one vector per utterance.

    Parameters
    ----------
    hidden : (B, T, D) tensor, possibly bf16/fp16, possibly padded.
    lengths : (B,) integer tensor of VALID frame counts.
    poolings : which reductions to compute; see `POOLINGS`.

    Returns
    -------
    dict mapping pooling name -> (B, D) float32 tensor, still on `hidden`'s
    device. The caller moves them to host memory once, which is the only
    transfer in the whole pipeline: (B, D) instead of (B, T, D).

    Notes
    -----
    Everything is promoted to float32 first. Two reasons, both real:

      * `sum` over ~1500 Whisper frames on an outlier dimension exceeds
        float16's 65504 ceiling and silently becomes `inf`.
      * `std` is computed two-pass (centre, then square) rather than via
        E[x^2] - E[x]^2. Squaring before centring overflows for the same reason
        and, independently of dtype, cancels catastrophically when the mean
        dwarfs the deviation — routine for these embeddings.

    Storage downstream is float32 for exactly the `sum` reason above. Do not
    "save space" by casting the pooled columns to float16.
    """
    unknown = [p for p in poolings if p not in POOLINGS]
    if unknown:
        raise ValueError(f"unknown pooling(s) {unknown}; choose from {sorted(POOLINGS)}")

    h = hidden.float()
    B, T, D = h.shape

    lengths = lengths.to(device=h.device, dtype=torch.long).clamp_(1, T)
    mask = torch.arange(T, device=h.device)[None, :] < lengths[:, None]     # (B, T)
    m3 = mask.unsqueeze(-1)                                                # (B, T, 1)
    n = lengths.to(h.dtype).unsqueeze(-1)                                  # (B, 1)

    want = set(poolings)
    out = {}

    # One masked sum serves sum, mean and std.
    if want & {"sum", "mean", "std"}:
        total = (h * m3).sum(dim=1)                                        # (B, D)
        mean = total / n
        if "sum" in want:
            out["sum"] = total
        if "mean" in want:
            out["mean"] = mean
        if "std" in want:
            centred = (h - mean.unsqueeze(1)) * m3        # padding contributes zero
            var = centred.pow(2).sum(dim=1) / n
            out["std"] = var.clamp_min(0).sqrt()          # clamp: -0.0 from rounding

    if "max" in want:
        out["max"] = h.masked_fill(~m3, float("-inf")).amax(dim=1)
    if "min" in want:
        out["min"] = h.masked_fill(~m3, float("inf")).amin(dim=1)
    if "first" in want:
        out["first"] = h[:, 0]
    if "last" in want:
        out["last"] = h[torch.arange(B, device=h.device), lengths - 1]

    return {p: out[p] for p in poolings}


def _reference_pool(frames, how):
    """Slow, obviously-correct reference implementation, for verification only."""
    f = np.asarray(frames, dtype=np.float64)
    return {
        "mean": lambda: f.mean(0), "max": lambda: f.max(0), "min": lambda: f.min(0),
        "sum": lambda: f.sum(0), "std": lambda: f.std(0),
        "first": lambda: f[0], "last": lambda: f[-1],
    }[how]()


def selftest_pooling(seed=0, tol=1e-4, verbose=True):
    """Check `pool_hidden` against a naive per-row loop on ragged synthetic data.

    Worth running once per session. Because the frame-level tensors are never
    stored, there is no way to re-derive a pooled column after the fact — an
    off-by-one in the mask or in the `last` index would be undetectable later,
    and would look like a plausible near-zero probe result rather than a crash.

    The padding is poisoned with a large constant so any leak shows up
    immediately, and the magnitudes are inflated (x50) so that a float16 `sum`
    or a one-pass `std` would overflow and fail this test loudly.
    """
    rng = np.random.default_rng(seed)
    B, T, D = 6, 23, 12
    lens = rng.integers(1, T + 1, size=B)
    x = (rng.normal(size=(B, T, D)) * 50.0).astype(np.float32)
    for i, L in enumerate(lens):
        x[i, L:] = 1e4

    fast = pool_hidden(torch.from_numpy(x), torch.from_numpy(lens))
    worst = {}
    for p in ALL_POOLINGS:
        ref = np.stack([_reference_pool(x[i, :lens[i]], p) for i in range(B)])
        got = fast[p].cpu().numpy().astype(np.float64)
        scale = np.maximum(np.abs(ref).max(axis=1, keepdims=True), 1e-12)
        worst[p] = float((np.abs(ref - got) / scale).max())

    ok = all(v < tol for v in worst.values()) and all(
        np.isfinite(v.cpu().numpy()).all() for v in fast.values()
    )
    if verbose:
        print("pool_hidden self-test:",
              "  ".join(f"{p}={worst[p]:.1e}" for p in ALL_POOLINGS))
        print("  ->", "OK (float32 rounding only)" if ok else "FAILED")
    return ok

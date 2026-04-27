"""
data.py
=======

Spherical code data generation, augmentation, and batched loading.

Provides:
  - sample_spherical_code(N, D)  — draw N uniform points on S^{D-1}
  - pad_batch(codes, D_max, N_max)  — collate variable-size codes into a
    padded (B, N_max, D_max) tensor with masks
  - ArchiveCache  — load all .pt files from a directory into RAM
  - SphereCodeMixedDataset  — mixed-source dataset with augmentations
  - BucketBatchSampler  — groups samples by (N, D) bucket to minimize padding
  - make_loader(length, cfg)  — convenience DataLoader builder (reads archive_dirs from cfg)

Data sources (selected randomly per sample):
  1. Pure random uniform codes on S^{D-1}
  2. Pre-computed semi-optimized codes loaded from .pt archive
  3. Perturbed archive codes (local or global noise)
  4. Quick-optimized random codes (a few repulsion steps)
  5. Furthest-point-sampled subsets of archive codes, optionally perturbed

Augmentations applied on top:
  - Random rotation (uniform SO(D) matrix)
  - Re-normalization to unit sphere

All functions accept duck-typed ``cfg`` objects (any namespace with the
expected attributes), so they are decoupled from the Config dataclass.
"""

from __future__ import annotations

import glob
import math
import os
import re
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


from tqdm import tqdm



# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_spherical_code(N: int, D: int, device: str = "cpu") -> torch.Tensor:
    """N uniform points on S^{D-1}."""
    x = torch.randn(N, D, device=device)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def normalize_to_sphere(x: torch.Tensor) -> torch.Tensor:
    """Project rows of x onto the unit sphere."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def random_rotation(D: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample a uniform random element of SO(D) via QR decomposition."""
    A = torch.randn(D, D, generator=generator)
    Q, R = torch.linalg.qr(A)
    # Ensure det(Q) = +1 (proper rotation)
    Q = Q * torch.diag(R).sign()
    return Q


def perturb_local(x: torch.Tensor, sigma: float,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """Apply a spatially-localized perturbation centered at a random point.

    A random center is chosen uniformly on S^{D-1}.  Each point receives
    Gaussian noise scaled by a weight that decays with geodesic distance
    from the center:  w_i = exp(-arccos(<x_i, c>)^2 / (2 * bandwidth^2)).
    The bandwidth is drawn uniformly from [0.3, 1.5] radians so the
    perturbation ranges from tightly focused to hemisphere-scale.

    ``sigma`` controls the peak noise magnitude.
    """
    N, D = x.shape
    # Random center on S^{D-1}
    center = torch.randn(D, generator=generator, dtype=x.dtype, device=x.device)
    center = center / center.norm().clamp_min(1e-12)
    # Geodesic distance from each point to the center
    cos_dist = (x @ center).clamp(-1.0, 1.0)       # (N,)
    geo_dist = cos_dist.acos()                       # (N,)  in [0, pi]
    # Random bandwidth (radians)
    bw = 0.3 + torch.rand(1, generator=generator, dtype=x.dtype, device=x.device).item() * 1.2
    weights = torch.exp(-0.5 * (geo_dist / bw) ** 2)  # (N,)  in [0, 1]
    # Weighted noise
    noise = torch.randn(N, D, generator=generator, dtype=x.dtype, device=x.device)
    noise = noise * (sigma * weights.unsqueeze(-1))
    return normalize_to_sphere(x + noise)


def perturb_global(x: torch.Tensor, sigma: float,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    """Apply a shared random displacement to all points, then re-normalize.

    Simulates a slight rotation or bias shift.
    """
    D = x.shape[-1]
    shift = torch.randn(D, generator=generator) * sigma
    return normalize_to_sphere(x + shift)


def perturb_tangent(x: torch.Tensor, sigma: float,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """Add tangent-space noise to each point independently.

    For each point, project isotropic Gaussian noise onto the tangent plane of
    the sphere at that point, then retract via normalization.  This stays
    closer to the manifold than naive ambient noise.
    """
    N, D = x.shape
    noise = torch.randn(N, D, generator=generator, dtype=x.dtype, device=x.device) * sigma
    # Project out the radial component: noise_tan = noise - <noise, x> x
    noise = noise - (noise * x).sum(-1, keepdim=True) * x
    return normalize_to_sphere(x + noise)


def perturb_swap(x: torch.Tensor, frac: float = 0.1,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Randomly swap pairs of points, then add small tangent noise.

    Swaps ``frac`` of point pairs, creating a combinatorial perturbation that
    is qualitatively different from continuous noise.
    """
    N, D = x.shape
    n_swaps = max(1, int(frac * N / 2))
    perm = torch.randperm(N, generator=generator)
    x = x.clone()
    for i in range(n_swaps):
        a, b = perm[2 * i].item(), perm[2 * i + 1].item()
        x[a], x[b] = x[b].clone(), x[a].clone()
    return x


def apply_rotation(x: torch.Tensor,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    """Rotate all points by a random SO(D) matrix."""
    D = x.shape[-1]
    Q = random_rotation(D, generator=generator)
    return x @ Q.T


def furthest_point_sample(x: torch.Tensor, K: int,
                          generator: torch.Generator | None = None) -> torch.Tensor:
    """Greedy furthest-point sampling of K points from x (N, D).

    Returns a (K, D) tensor.
    """
    N = x.shape[0]
    K = min(K, N)
    idx0 = torch.randint(N, (1,), generator=generator).item()
    selected = [idx0]
    # min distance from each point to the selected set
    dists = torch.full((N,), float("inf"))
    for _ in range(K - 1):
        last = x[selected[-1]]
        d = 1.0 - (x @ last)  # cosine distance on the sphere
        dists = torch.minimum(dists, d)
        # pick the farthest
        nxt = dists.argmax().item()
        selected.append(nxt)
    return x[selected]


def quick_optimize(x: torch.Tensor, steps: int = 10,
                   lr: float = 0.05,
                   potential: str = "coulomb") -> torch.Tensor:
    """Run a few Riemannian repulsion steps to spread points on the sphere.

    Supports two potentials:
      - "coulomb":  E = sum_{i<j} 1 / ||x_i - x_j||
      - "log":      E = -sum_{i<j} log(||x_i - x_j||)

    Gradient is projected onto the tangent plane, then points are
    re-normalized.  LR decays linearly to lr/3 over the steps.
    """
    x = x.clone().detach().requires_grad_(False)
    N, D = x.shape
    if N <= 1:
        return x
    for s in range(steps):
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            # pairwise squared distances
            diff = x.unsqueeze(0) - x.unsqueeze(1)          # (N, N, D)
            dist2 = (diff * diff).sum(-1)                    # (N, N)
            # mask diagonal
            mask = ~torch.eye(N, dtype=torch.bool, device=x.device)
            dist = dist2[mask].clamp_min(1e-8).sqrt()
            if potential == "log":
                energy = -(dist.log()).sum()
            else:
                energy = (1.0 / dist).sum()
            energy.backward()
        with torch.no_grad():
            g = x.grad
            # project gradient onto tangent plane of sphere
            g = g - (g * x).sum(-1, keepdim=True) * x
            # linear LR decay
            step_lr = lr * (1.0 - 2.0 / 3.0 * s / max(steps - 1, 1))
            x = normalize_to_sphere(x - step_lr * g)
    return x.detach()


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def pad_batch(codes, D_max: int, N_max: int):
    """Pad a list of (N_i, D_i) codes into (B, N_max, D_max) plus masks.

    Always allocates the full (N_max, D_max) shape so that PyTorch's CUDA
    caching allocator can reuse freed blocks across batches without
    fragmentation.  The encoder and decoder crop to batch-local N/D
    internally, so compute cost is still proportional to the actual data.

    Returns
    -------
    x : (B, N_max, D_max) float tensor, zero-padded
    mask : (B, N_max) bool tensor — True for real points
    Ds : (B,) long tensor — ambient dimension of each code
    Ns : (B,) long tensor — number of points in each code
    """
    B = len(codes)
    x = torch.zeros(B, N_max, D_max)
    mask = torch.zeros(B, N_max, dtype=torch.bool)
    Ds = torch.zeros(B, dtype=torch.long)
    Ns = torch.zeros(B, dtype=torch.long)
    for i, c in enumerate(codes):
        N, D = c.shape
        x[i, :N, :D] = c
        mask[i, :N] = True
        Ds[i], Ns[i] = D, N
    return x, mask, Ds, Ns


# ---------------------------------------------------------------------------
# Archive cache — load all .pt files into RAM
# ---------------------------------------------------------------------------

class ArchiveCache:
    """Pre-load all .pt files from a directory into RAM for fast access.

    Files are expected to be named ``*_N{n}_D{d}.pt`` and contain a single
    (N, D) float tensor of unit vectors on S^{D-1}.
    """

    def __init__(self, directory: str, N_min: int = 0, N_max: int = 10000,
                 D_min: int = 0, D_max: int = 10000, verbose: bool = True):
        self.codes: list[torch.Tensor] = []
        self.by_nd: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.N_vals: list[int] = []
        self.D_vals: list[int] = []

        pattern = os.path.join(directory, "*.pt")
        files = sorted(glob.glob(pattern))
        if verbose:
            print(f"[ArchiveCache] loading {len(files)} files from {directory} ...")

        loaded = 0
        file_iter = tqdm(files, desc="Loading archive", unit="file", disable=not verbose)
        for f in file_iter:
            m = re.search(r"N(\d+)_D(\d+)", os.path.basename(f))
            if not m:
                continue
            n, d = int(m.group(1)), int(m.group(2))
            if not (N_min <= n <= N_max and D_min <= d <= D_max):
                continue
            t = torch.load(f, weights_only=True, map_location="cpu").float()
            if t.ndim != 2:
                continue
            # Use actual tensor shape — filename may be mismatched
            n_actual, d_actual = t.shape
            if not (N_min <= n_actual <= N_max and D_min <= d_actual <= D_max):
                continue
            n, d = n_actual, d_actual
            # Ensure unit norm
            t = normalize_to_sphere(t)
            idx = len(self.codes)
            self.codes.append(t)
            self.by_nd[(n, d)].append(idx)
            self.N_vals.append(n)
            self.D_vals.append(d)
            loaded += 1

        self.N_vals = sorted(set(self.N_vals))
        self.D_vals = sorted(set(self.D_vals))
        if verbose:
            print(f"[ArchiveCache] loaded {loaded} codes, "
                  f"N in [{self.N_vals[0]}..{self.N_vals[-1]}], "
                  f"D in [{self.D_vals[0]}..{self.D_vals[-1]}]")

    def __len__(self):
        return len(self.codes)

    def sample_random(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Return a random code from the archive."""
        idx = torch.randint(len(self.codes), (1,), generator=generator).item()
        return self.codes[idx]

    def sample_with_nd(self, N: int, D: int,
                       generator: torch.Generator | None = None) -> torch.Tensor | None:
        """Return a random code with exactly (N, D), or None."""
        pool = self.by_nd.get((N, D))
        if not pool:
            return None
        idx = pool[torch.randint(len(pool), (1,), generator=generator).item()]
        return self.codes[idx]


# ---------------------------------------------------------------------------
# Multi-archive cache — equal-probability mixing across multiple directories
# ---------------------------------------------------------------------------

class MultiArchiveCache:
    """Wraps multiple ArchiveCache instances and samples equally across them.

    On each call to ``sample_random``, one archive is chosen uniformly at
    random and then one code is drawn from that archive.  This ensures codes
    from larger archives are not over-represented relative to smaller ones.
    """

    def __init__(self, caches: list[ArchiveCache]):
        self.caches = [c for c in caches if len(c) > 0]

    def __len__(self) -> int:
        return sum(len(c) for c in self.caches)

    @property
    def N_vals(self) -> list[int]:
        vals: set[int] = set()
        for c in self.caches:
            vals.update(c.N_vals)
        return sorted(vals)

    @property
    def D_vals(self) -> list[int]:
        vals: set[int] = set()
        for c in self.caches:
            vals.update(c.D_vals)
        return sorted(vals)

    def sample_random(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Pick one archive uniformly at random, then return a random code from it."""
        idx = torch.randint(len(self.caches), (1,), generator=generator).item()
        return self.caches[idx].sample_random(generator=generator)

    def sample_with_nd(self, N: int, D: int,
                       generator: torch.Generator | None = None) -> torch.Tensor | None:
        """Return a random code with exactly (N, D) from any archive, or None."""
        pool = [c for c in self.caches if c.by_nd.get((N, D))]
        if not pool:
            return None
        idx = torch.randint(len(pool), (1,), generator=generator).item()
        return pool[idx].sample_with_nd(N, D, generator=generator)


# ---------------------------------------------------------------------------
# Mixed-source dataset with augmentations
# ---------------------------------------------------------------------------

# Source selection weights (unnormalized).  Adjusted at init time based on
# whether an archive is available.
_DEFAULT_SOURCE_WEIGHTS = {
    "random":            0.15,   # pure random uniform on S^{D-1}
    "archive":           0.50,   # raw archive code
    "fps":               0.35,   # furthest-point sample from archive
}

# Available perturbation types, selected uniformly at random per application.
_PERTURB_FNS = {
    "local":   perturb_local,
    "global":  perturb_global,
    "tangent": perturb_tangent,
}


class SphereCodeMixedDataset(Dataset):
    """Mixed-source dataset with a composable augmentation pipeline.

    Pipeline per sample:
      1. **Base code**: pick a source (random / archive / FPS subset).
      2. **Augmentation ops** (perturb, optimize): each is applied
         independently with a given probability.  The *order* is randomized
         per sample, and perturbation may be applied multiple times (the
         count is drawn from a geometric distribution).
      3. **Rotation**: always applied last (it does not change the geometry
         of the code, only the coordinate frame).
      4. **Final normalization**: always applied last to guarantee unit norm.

    Normalize-to-sphere is called after every individual augmentation step.
    """

    def __init__(
        self,
        length: int,
        D_min: int,
        D_max: int,
        N_min: int,
        N_max: int,
        archive: ArchiveCache | None = None,
        source_weights: dict[str, float] | None = None,
        perturb_prob: float = 0.4,
        max_perturb_rounds: int = 3,
        optimize_prob: float = 0.15,
        perturb_sigma_range: tuple[float, float] = (0.001, 0.15),
        quick_opt_steps_range: tuple[int, int] = (3, 15),
        quick_opt_lr: float = 0.05,
        seed: int | None = None,
    ):
        self.length = length
        self.D_min, self.D_max = D_min, D_max
        self.N_min, self.N_max = N_min, N_max
        self.archive = archive
        self.perturb_prob = perturb_prob
        self.max_perturb_rounds = max_perturb_rounds
        self.optimize_prob = optimize_prob
        self.perturb_sigma_range = perturb_sigma_range
        self.quick_opt_steps_range = quick_opt_steps_range
        self.quick_opt_lr = quick_opt_lr
        self.seed = seed
        self._perturb_names = list(_PERTURB_FNS.keys())

        # Build cumulative source weights
        weights = dict(_DEFAULT_SOURCE_WEIGHTS)
        if source_weights is not None:
            weights.update(source_weights)
        if archive is None or len(archive) == 0:
            for k in ("archive", "fps"):
                weights[k] = 0.0
        self._source_names = list(weights.keys())
        w = torch.tensor([weights[k] for k in self._source_names], dtype=torch.float)
        self._source_cum = (w / w.sum()).cumsum(0)

    def __len__(self):
        return self.length

    # ---- internal helpers --------------------------------------------------

    def _gen(self, idx) -> torch.Generator | None:
        if self.seed is not None:
            g = torch.Generator()
            g.manual_seed(self.seed + idx)
            return g
        return None

    def _rand_ND(self, g):
        D = torch.randint(self.D_min, self.D_max + 1, (1,), generator=g).item()
        N = torch.randint(self.N_min, self.N_max + 1, (1,), generator=g).item()
        return N, D

    def _rand_sigma(self, g):
        lo, hi = self.perturb_sigma_range
        log_lo, log_hi = math.log(lo), math.log(hi)
        u = torch.rand(1, generator=g).item()
        return math.exp(log_lo + u * (log_hi - log_lo))

    def _rand_opt_steps(self, g):
        lo, hi = self.quick_opt_steps_range
        return torch.randint(lo, hi + 1, (1,), generator=g).item()

    def _pick_source(self, g) -> str:
        u = torch.rand(1, generator=g).item()
        for i, c in enumerate(self._source_cum):
            if u < c.item():
                return self._source_names[i]
        return self._source_names[-1]

    def _get_archive_code(self, g):
        if self.archive is None or len(self.archive) == 0:
            return None, None, None
        code = self.archive.sample_random(generator=g)
        n, d = code.shape
        return code, n, d

    # ---- augmentation steps ------------------------------------------------

    def _apply_perturb(self, code, g):
        """Apply one random perturbation type with a random sigma."""
        sigma = self._rand_sigma(g)
        kind = self._perturb_names[
            torch.randint(len(self._perturb_names), (1,), generator=g).item()
        ]
        fn = _PERTURB_FNS[kind]
        if kind == "local" or kind == "global":
            code = fn(code, sigma, generator=g)
        else:
            code = fn(code, sigma, generator=g)
        return normalize_to_sphere(code)

    def _apply_optimize(self, code, g):
        """Apply a few Riemannian repulsion steps."""
        steps = self._rand_opt_steps(g)
        potential = "log" if torch.rand(1, generator=g).item() < 0.3 else "coulomb"
        code = quick_optimize(code, steps=steps, lr=self.quick_opt_lr,
                              potential=potential)
        return normalize_to_sphere(code)

    # ---- main generation ---------------------------------------------------

    def __getitem__(self, idx):
        g = self._gen(idx)

        # --- 1. Base code ---
        source = self._pick_source(g)

        if source == "archive":
            code, N, D = self._get_archive_code(g)
            if code is None:
                N, D = self._rand_ND(g)
                code = sample_spherical_code(N, D)
        elif source == "fps":
            code, N_full, D = self._get_archive_code(g)
            if code is None:
                N, D = self._rand_ND(g)
                code = sample_spherical_code(N, D)
            else:
                K = torch.randint(
                    max(self.N_min, 3),
                    min(N_full, self.N_max) + 1,
                    (1,), generator=g,
                ).item()
                code = furthest_point_sample(code, K, generator=g)
        else:  # "random" or fallback
            N, D = self._rand_ND(g)
            code = sample_spherical_code(N, D)

        code = normalize_to_sphere(code)

        # --- 2. Build augmentation ops list and shuffle order ---
        ops = []  # list of callables: fn(code, g) -> code

        # Perturbation: geometric number of rounds (1..max), each independent
        if torch.rand(1, generator=g).item() < self.perturb_prob:
            n_rounds = 1
            for _ in range(1, self.max_perturb_rounds):
                if torch.rand(1, generator=g).item() < 0.4:
                    n_rounds += 1
                else:
                    break
            for _ in range(n_rounds):
                ops.append(self._apply_perturb)

        # Optimization
        if torch.rand(1, generator=g).item() < self.optimize_prob:
            ops.append(self._apply_optimize)

        # Shuffle the order so perturb and optimize can interleave freely
        if len(ops) > 1:
            perm = torch.randperm(len(ops), generator=g).tolist()
            ops = [ops[i] for i in perm]

        # --- 3. Apply augmentation ops ---
        for op in ops:
            code = op(code, g)

        # --- 4. Always rotate last (does not change the spherical code) ---
        code = apply_rotation(code, generator=g)

        # --- 5. Final normalization ---
        code = normalize_to_sphere(code)

        # Attach (N, D) metadata for the bucket sampler
        code._nd = code.shape
        return code

    def predict_shape(self, idx) -> tuple[int, int]:
        """Return the (N, D) that __getitem__(idx) will produce.

        Replays the same RNG draws that __getitem__ uses to select the
        source and determine the code shape, without doing any expensive
        augmentation or I/O.
        """
        g = self._gen(idx)
        source = self._pick_source(g)

        if source == "archive":
            code, N, D = self._get_archive_code(g)
            if code is not None:
                return N, D
            N, D = self._rand_ND(g)
            return N, D
        elif source == "fps":
            code, N_full, D = self._get_archive_code(g)
            if code is not None:
                K = torch.randint(
                    max(self.N_min, 3),
                    min(N_full, self.N_max) + 1,
                    (1,), generator=g,
                ).item()
                return K, D
            N, D = self._rand_ND(g)
            return N, D
        else:
            N, D = self._rand_ND(g)
            return N, D


# ---------------------------------------------------------------------------
# Bucket batch sampler
# ---------------------------------------------------------------------------

class BucketBatchSampler(Sampler):
    """Groups dataset indices into batches where samples have similar (N, D),
    reducing padding waste.

    Strategy: predict approximate (N, D) for each index, sort indices by a
    composite key ``(D_bucket, N_bucket)`` with local jitter, then slice into
    consecutive batches.  This ensures every batch is full (except possibly the
    last) and that samples within a batch have similar sizes.
    """

    def __init__(self, dataset: SphereCodeMixedDataset, batch_size: int,
                 drop_last: bool = True, seed: int | None = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        if self.seed is not None:
            g.manual_seed(self.seed + torch.randint(0, 2**31, (1,)).item())
        else:
            g.manual_seed(torch.randint(0, 2**31, (1,)).item())

        ds = self.dataset
        n_items = len(ds)

        # Group indices by D bucket, then sort by N within each group.
        # This guarantees same-D batches and avoids the boundary mixing that
        # occurs when a linear sort places high-N D=k items next to low-N D=k+1
        # items in the same batch.
        by_D: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for i in range(n_items):
            N, D = ds.predict_shape(i)
            by_D[D].append((N, i))

        all_batches = []
        for D in sorted(by_D.keys()):
            items = by_D[D]
            # Sort by N with small per-item jitter so batches aren't identical
            # across epochs. Jitter is tiny relative to N_range so it only
            # breaks ties, not the overall ordering.
            jittered = sorted(
                items,
                key=lambda x: x[0] + torch.rand(1, generator=g).item() * 0.5,
            )
            indices = [x[1] for x in jittered]
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        # Shuffle batch order so the model doesn't see D/N in sorted order
        batch_perm = torch.randperm(len(all_batches), generator=g)
        for i in batch_perm.tolist():
            yield all_batches[i]

    def __len__(self):
        n_items = len(self.dataset)
        if self.drop_last:
            return n_items // self.batch_size
        return (n_items + self.batch_size - 1) // self.batch_size


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------

# Module-level cache keyed by the sorted tuple of archive directories, so
# different archive_dirs configs coexist without re-loading.
_archive_caches: dict[tuple[str, ...], ArchiveCache | MultiArchiveCache] = {}


def load_archives(
    directories: list[str], cfg
) -> "ArchiveCache | MultiArchiveCache | None":
    """Load (or return cached) archives from a list of directories.

    Each directory is loaded into its own :class:`ArchiveCache`.  When more
    than one valid directory is provided the caches are wrapped in a
    :class:`MultiArchiveCache` so each directory is sampled with equal
    probability regardless of how many codes it contains.
    """
    global _archive_caches
    valid_dirs = [d for d in directories if os.path.isdir(d)]
    if not valid_dirs:
        return None
    key = tuple(sorted(valid_dirs))
    if key not in _archive_caches:
        caches = [
            ArchiveCache(
                d,
                N_min=cfg.N_min, N_max=cfg.N_max,
                D_min=cfg.D_min, D_max=cfg.D_max,
            )
            for d in valid_dirs
        ]
        _archive_caches[key] = MultiArchiveCache(caches) if len(caches) > 1 else caches[0]
    return _archive_caches[key]


def make_loader(length: int, cfg, seed: int | None = None) -> DataLoader:
    """Build a DataLoader of mixed-source spherical codes with bucket batching.

    ``cfg`` must expose: D_min, D_max, N_min, N_max, batch_size, num_workers.
    Archive directories and all augmentation / source-weight parameters are
    read from cfg (with safe getattr fallbacks for backward compatibility):
      archive_dirs, perturb_prob, max_perturb_rounds, optimize_prob,
      perturb_sigma_lo, perturb_sigma_hi, quick_opt_steps_lo,
      quick_opt_steps_hi, source_weight_random, source_weight_archive,
      source_weight_fps.
    """
    archive_dirs = getattr(cfg, "archive_dirs", ["combined_points_archive"])
    archive = load_archives(archive_dirs, cfg) if archive_dirs else None

    source_weights = {
        "random":  getattr(cfg, "source_weight_random",  0.15),
        "archive": getattr(cfg, "source_weight_archive", 0.50),
        "fps":     getattr(cfg, "source_weight_fps",     0.35),
    }

    ds = SphereCodeMixedDataset(
        length=length,
        D_min=cfg.D_min,
        D_max=cfg.D_max,
        N_min=cfg.N_min,
        N_max=cfg.N_max,
        archive=archive,
        source_weights=source_weights,
        perturb_prob=getattr(cfg, "perturb_prob", 0.4),
        max_perturb_rounds=getattr(cfg, "max_perturb_rounds", 3),
        optimize_prob=getattr(cfg, "optimize_prob", 0.15),
        perturb_sigma_range=(
            getattr(cfg, "perturb_sigma_lo", 0.001),
            getattr(cfg, "perturb_sigma_hi", 0.15),
        ),
        quick_opt_steps_range=(
            getattr(cfg, "quick_opt_steps_lo", 3),
            getattr(cfg, "quick_opt_steps_hi", 15),
        ),
        seed=seed,
    )

    sampler = BucketBatchSampler(
        ds, batch_size=cfg.batch_size, drop_last=True, seed=seed,
    )

    return DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True, # Optimization: speed up host to device transfer
        collate_fn=lambda batch: pad_batch(batch, cfg.D_max, cfg.N_max),
    )

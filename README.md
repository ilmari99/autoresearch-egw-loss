# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The setup here has been retargeted from short-horizon model training to short-horizon **loss-function research**. An agent edits [loss.py](loss.py), runs [test.py](test.py), checks whether the aggregate `loss_suitability` score improved, and keeps or discards the change. The evaluation harness also prints per-test diagnostics so the agent can see *why* a loss improved or regressed.

## How it works

The repo is deliberately kept small and only really has four files that matter:

- **`loss.py`** — the single file the agent edits. It contains the loss implementation and solver details. **This file is edited and iterated on by the agent**.
- **`data.py`** — shared data-loading, sampling, perturbation, and optimisation helpers used by the evaluation harness. **This file is read for context but not modified during the experiment loop**.
- **`test.py`** — the fixed evaluation harness. It runs a suite of loss-property tests, prints a scorecard, and emits the single optimization target `loss_suitability`. **This file is not modified during the experiment loop**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, evaluation runs under a **fixed 10-minute time budget**. The metric is **loss_suitability** — lower is better. It is a bounded weighted aggregate over the full property suite in [test.py](test.py), with fundamental properties like identity, differentiability, bounded gradients, optimisation recovery, permutation equivariance, padding invariance, and batch invariance carrying the highest weight.

The per-test scorecard includes a `score` column showing each test's weighted contribution to the final `loss_suitability`, so the agent can see exactly where the aggregate is coming from without relying on VRAM bookkeeping.

## Quick start

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU with a CUDA 12.8 compatible driver for the default setup. The current [pyproject.toml](pyproject.toml) pins a CUDA-enabled PyTorch wheel.

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Manually run the evaluation harness
uv run test.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

Before starting the autonomous keep/discard loop, make sure the retargeted loss-harness files are tracked by git and the repo is in a state where resetting `loss.py` will actually revert your experiment. The workflow assumes `loss.py`, `data.py`, and `test.py` are versioned files, not untracked scratch files.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
loss.py         — loss implementation and solver (agent modifies this)
data.py         — shared benchmark helpers (read-only during experiments)
test.py         — fixed evaluation harness and aggregate scoring
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `loss.py`. `data.py` and `test.py` are fixed support files during the experiment loop. This keeps the scope manageable and diffs reviewable.
- **Fixed evaluation budget.** `test.py` enforces a hard 10-minute wall-clock budget. This makes experiments directly comparable and prevents the agent from "winning" by making evaluation arbitrarily expensive.
- **Weighted bounded scoring.** `loss_suitability` is a capped weighted aggregate, so one catastrophic subtest cannot dominate the whole score while still being visible in the scorecard.
- **Per-test diagnostics.** The harness prints the raw property measurements, per-test penalty breakdown, and per-test score contribution, so the agent can optimize with signal rather than blindly hill-climbing a scalar.

## Platform support

The compact benchmark profile in [test.py](test.py) is meant to complete within the 10-minute budget on a reasonable GPU. If you want a broader but slower sweep, run `uv run test.py --stress` manually, but the autonomous loop should normally optimize the default compact profile.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## License

MIT

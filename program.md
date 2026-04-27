# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Verify git preconditions**: The keep/discard loop assumes git can actually revert your experiments. Before starting, make sure `loss.py`, `data.py`, `test.py`, `README.md`, `program.md`, and `pyproject.toml` are tracked in this repo and that the working tree is in a state where resetting `loss.py` will restore the baseline.
4. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `loss.py` — the file you modify. This is where you implement your experimental ideas.
   - `data.py` — read-only support code used by the harness. Do not modify it during the loop.
   - `test.py` — the fixed evaluation harness. Do not modify it during the experiment loop.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment evaluates a candidate `loss.py` on a **fixed time budget of 10 minutes**. You launch it simply as: `uv run test.py`.

**What you CAN do:**
- Modify `loss.py` — this is the only file you edit during the loop. Everything inside the loss implementation is fair game.

**What you CANNOT do:**
- Modify `data.py`. It is read-only support code for fixtures, perturbations, and helper routines.
- Modify `test.py`. It is the fixed evaluation harness and scoring function.
- Modify `program.md`, `README.md`, or `pyproject.toml` as part of the search loop.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Hack the metrics. Do not alter score extraction, printed summaries, log parsing, `results.tsv`, CLI scope, or any other bookkeeping around the evaluation. Improve only the loss implementation.

**The goal is simple: get the lowest loss_suitability.** Lower is better. The evaluation harness prints a per-test scorecard with each test's weighted score contribution and a final aggregate score; use the per-test diagnostics to decide what to improve next.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the evaluation as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
loss_suitability: 21.278545
total_seconds:    97.2
status:           ok
tests_run:        17/17
```

The script also prints a detailed scorecard above this summary. The `score` column is each test's weighted contribution to `loss_suitability`, so the positive-weight rows add up to the final metric. The key metric can be extracted from the log file with:

```
grep "^loss_suitability:\|^total_seconds:\|^status:\|^tests_run:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 4 columns:

```
commit	loss_suitability	status	description
```

1. git commit hash (short, 7 chars)
2. loss_suitability achieved (e.g. 21.278545) — use 999.000000 for crashes/timeouts
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	loss_suitability	status	description
a1b2c3d	31.482100	keep	baseline
b2c3d4e	24.903700	keep	deterministic sorted-row init
c3d4e5f	26.781900	discard	sharper epsilon with fewer outer steps
d4e5f6g	999.000000	crash	removed masking guard
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `loss.py` with an experimental idea by directly editing the code.
3. git commit
4. Run the experiment: `uv run test.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the summary: `grep "^loss_suitability:\|^total_seconds:\|^status:\|^tests_run:" run.log`
6. If the grep output is empty, the run crashed before the final summary. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If the summary exists but `tests_run` is not the full count, inspect the scorecard for rows with `crash`, `timeout`, or `skipped` — `test.py` records per-test failures there even when the run still prints a final summary.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If loss_suitability improved (lower), you "advance" the branch, keeping the git commit
9. If loss_suitability is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: `test.py` enforces the global 10-minute evaluation budget internally. If the summary says `status: timeout`, treat the run as a failure and revert unless the user explicitly wants to study that regime.

**Crashes**: If a run crashes (OOM, bug, numerical issue, etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo or a missing mask), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log `crash` in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Each evaluation prints which properties are still weak, so you should use that signal to guide the next `loss.py` change instead of searching blindly.

# M03 branch and remote-machine checkout

M03 is developed on **`lens-methods-m-03`** in:

```text
https://github.com/Adefioye/affect-geometry.git
```

## Published branch verified on 2026-11-19

The local branch tracks **`origin/lens-methods-m-03`**. The implementation is committed and
published on GitHub at:

```text
8bde57005a61b258eac3b5f694efa5490d416daf
Add implementation for m03 experiment
```

The source, configurations, tests, existing experiment guides, and `m-03-docs.md` are tracked
in that revision. View the [published branch](https://github.com/Adefioye/affect-geometry/tree/lens-methods-m-03).
The clone and switch instructions below can retrieve the implementation directly.

To verify the currently published branch from an existing clone:

```bash
git ls-remote --heads origin refs/heads/lens-methods-m-03
```

At the time of verification it returned:

```text
8bde57005a61b258eac3b5f694efa5490d416daf  refs/heads/lens-methods-m-03
```

## Fresh clone on a remote GPU machine

Choose a working directory on the remote machine, then run:

```bash
git clone --branch lens-methods-m-03 --single-branch \
  https://github.com/Adefioye/affect-geometry.git
cd affect-geometry
```

This creates a checkout of the M03 branch and configures its remote tracking branch.
If Git reports that the branch does not exist, check the repository URL, account access,
and branch spelling against the values above.

## Switch an existing clone on a remote machine

Enter the existing `affect-geometry/` repository. Review `git status` and commit or stash
local edits before switching branches. Register M03 among the remote's fetched branches and
fetch its explicit reference so tracking also works for a clone originally restricted to `main`.

```bash
cd /path/to/affect-geometry
git status --short --branch
git remote set-branches --add origin lens-methods-m-03
git fetch origin refs/heads/lens-methods-m-03:refs/remotes/origin/lens-methods-m-03

if git show-ref --verify --quiet refs/heads/lens-methods-m-03; then
  git switch lens-methods-m-03
  git branch --set-upstream-to=origin/lens-methods-m-03 lens-methods-m-03
else
  git switch --create lens-methods-m-03 --track origin/lens-methods-m-03
fi

git merge --ff-only origin/lens-methods-m-03
```

`--ff-only` updates a branch that is behind the published branch without creating a merge
commit. A divergence error means that the local and remote histories need to be reconciled
before running that version of the experiment. The commands use `origin` as listed above;
check `git remote -v` if the remote machine uses a different remote name or repository URL.

## Verify the checkout and install

```bash
git branch --show-current
git rev-parse HEAD
git log -1 --format='%h %s'
git ls-files src/affect_geometry/m03 configs/m03 tests/m03 experiments/m03

test -f src/affect_geometry/m03/__main__.py
test -f configs/m03/gemma2-all-label-free.json
test -f experiments/m03/README.md
```

The branch should be `lens-methods-m-03`, and the file listing should include the implementation,
configurations, tests, and run guide. Record `git rev-parse HEAD` when setting up the run so
the code revision can be matched across machines. Commit `8bde570` is the verified
implementation revision; the branch can include later documented changes.

Install from the checked-out branch and run the CPU integration check:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,m03]'
export MPLCONFIGDIR="$PWD/outputs/m03_lens_comparison/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/outputs/m03_lens_comparison/.cache"

python -m pytest tests/m03 -q
python -m affect_geometry.m03 smoke --run-id remote-smoke-01
```

For the actual Gemma run, follow the [run guide](README.md), including a CUDA-compatible
PyTorch environment, Gemma license access, and `HF_TOKEN` for the model and private EXP2
baseline. The runner downloads its pinned artifacts when the local copies are absent.
Git does not transfer the ignored `.env`, weights, baseline snapshots, caches, or run outputs.

After the environment and access are ready, the command for all implemented label-free arms is:

```bash
python -m affect_geometry.m03 run \
  --config configs/m03/gemma2-all-label-free.json \
  --run-id m03-gemma2-all-01
```

Keep the checked-out code, configuration, and run files fixed while an experiment is running
or being resumed. The pipeline validates the implementation/configuration hashes and extraction
software versions against its frozen manifests. See the run guide for resume behavior and
the separate stages used to inspect the GPU pilot before full extraction.

## Publish later local documentation updates

New local edits reach other machines after they are committed and pushed. For this guide
and its index link, review and publish those two paths from the development checkout:

```bash
git add experiments/m03/README.md experiments/m03/REMOTE.md
git diff --cached --stat
git diff --cached
git commit -m "Document M03 branch and remote-machine checkout"
git push origin lens-methods-m-03
```

Writing this guide does not itself commit or push these documentation changes.

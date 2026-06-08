# Contributing to LGVSC

Thanks for your interest in LGVSC. This is a research code release; contributions that
improve clarity, reproducibility, or portability are very welcome.

## Repository layout

The repo is organized by **pipeline stage** (`01_data_prep` … `07_downstream`), plus
`docs/`, `environment/`, and `env.sh`. See `README.md` for the full map and `docs/` for the
architecture walkthrough, data sources, smoke test, and reproducibility notes.

## Setup

1. Copy your machine paths into `env.sh`, then `source env.sh`. Every stage resolves its
   inputs/outputs from those variables (`DATA_ROOT`, `OPENSORA_DIR`, `INTERNVL_DIR`, …).
2. Each stage runs in its own conda environment (the large models have incompatible
   dependencies). Create one with `conda env create -f environment/<name>.yml`; the
   stage↔environment mapping is in the `README.md` table.

## Smoke test

Before opening a PR, run the bundled end-to-end smoke test on the sample under `assets/`
and confirm your output matches the recorded CBR/PSNR. See `docs/SMOKE_TEST.md` for the
exact commands and expected numbers.

## Code conventions

- **Paths**: never hardcode absolute paths in executable code. Route everything through the
  `env.sh` variables (Python reads them via `os.environ` defaults; shell scripts reference
  them directly) and allow per-run CLI overrides.
- **No vendored upstreams**: keep large external models out of the repo. If you must change
  an upstream file, ship a `.patch` (as we do for NTSCC / TimeSformer), not a copy.
- **No committed artifacts**: no `__pycache__/`, `*.log`, weights, or dataset media. These
  are covered by `.gitignore`.

## Before submitting

Run the same lightweight checks our CI runs:

```bash
# byte-compile all Python
find . -name '*.py' -not -path './*/__pycache__/*' -print0 | xargs -0 -n1 python -m py_compile

# syntax-check all shell scripts
find . -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

Then open a pull request describing **what** changed and **why**, and note any change to the
reproduction commands so the README / stage READMEs stay in sync.

## License

By contributing, you agree that your contributions are licensed under the repository's
`LICENSE` (MIT). Note that the third-party components LGVSC builds on retain their own
licenses — see `THIRD_PARTY_NOTICES.md`.

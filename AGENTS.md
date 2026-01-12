# Repository Guidelines

## Project Structure & Module Organization
- Root scripts: `llama_topk_mass.py` collects top‑k probability mass from a llama.cpp server; `plot.py` turns a CSV into summaries and plots.
- Data and artifacts live at the repo root (`mass.csv`, `mass.json`, `mass.xlsx`) and in output folders like `400-tok/` and `1k-tok/` (plots and summary CSVs).
- `LICENSE` and `.gitattributes` live at the top level.

## Build, Test, and Development Commands
- Generate per‑step mass data:
  `python llama_topk_mass.py --prompt "..." --klist 100,128,200 --max-tokens 400 --endpoint completion --csv mass.csv --json mass.json`
  This calls a local llama.cpp server (default `127.0.0.1:8080`).
- Create plots and summaries from the CSV:
  `python plot.py`
  Note: `plot.py` currently reads `/mnt/data/mass.csv`. Update `csv_path` or copy your CSV to that location for local runs.
- Dependencies are plain Python packages (e.g., `requests`, `pandas`, `numpy`, `matplotlib`).

## Coding Style & Naming Conventions
- Python style: 4‑space indentation, snake_case for variables/functions, and clear docstrings for scripts.
- Data/asset naming uses `mass_*` prefixes and `k`-based suffixes (e.g., `mass_summary_1k.csv`, `mass_hist.png`).
- Keep generated artifacts grouped in a directory per run or token budget (e.g., `400-tok/`, `1k-tok/`).

## Testing Guidelines
- There is no automated test suite in this repo.
- Validate changes by running the scripts and confirming expected outputs (CSV/JSON plus plots) are produced and readable.
- If you add tests, keep fixtures small and prefer `pytest` with deterministic inputs.

## Commit & Pull Request Guidelines
- Git history is minimal and uses short, plain‑English messages (e.g., “Initial commit”, “Add stuff to repo!”). Keep messages concise and imperative.
- PRs should include:
  - A brief summary of the change and why it matters.
  - Reproduction steps (commands and parameters used).
  - Any new or updated artifacts (plots/CSVs) and their locations.

## Configuration Notes
- `llama_topk_mass.py` assumes a running llama.cpp server; adjust `--host`, `--port`, and `--endpoint` as needed.
- Large `k` values produce large payloads; expect slower runs and bigger output files.

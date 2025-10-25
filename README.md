# PyCCM

[![License: GNU GPL 3.0](https://img.shields.io/badge/License-GNUGPL3.0-green.svg)](#license)
![Python](https://img.shields.io/badge/Python-3.7_|_3.8_|_3.9_|_3.10_|_3.11-blue)
![Tests](https://img.shields.io/badge/Tests-211_passing-brightgreen)
![Quality](https://img.shields.io/badge/Quality-4.7%2F5_stars-yellow)
![OS](https://img.shields.io/badge/OS-Linux_|_Windows_|_macOS-red)

Population projection pipeline with single-year ages ("unabridged") and joint parameter sweeps for fertility & mortality assumptions. Supports parallelized scenario runs and consolidated outputs (life tables, ASFR, Leslie matrices, projections).

> **Repo name:** `PyCCM`  
> **Entry point:** `src/main_compute.py`  
> **Config:** `config.yaml` (root)  
> **Data dir:** `data/` (CSV inputs)  
> **Outputs:** `results_unabridged/` (default)  
> **Status:** Production Ready (211/211 tests passing)


---

## Table of contents

- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Parameter sweeps & parallelism](#parameter-sweeps--parallelism)
- [Inputs & expected files](#inputs--expected-files)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Testing/Documentation Quick Links](#testingdocumentation-quick-links)
- [Citation](#citation)
- [License](#license)

---

## Features

- **Unabridged** single-year age structure & annual projections
- **Joint sweeps** (Cartesian product) over:
  - Target TFR range
  - Mortality improvement totals
  - Life-table smoothing MA window
- **Parallel execution** across grid points/tasks
- **Consolidated outputs** per artifact type (CSV + Parquet)
- Reproducible seeds per scenario label
- Optional midpoints blending (EEVV vs. Census) for deaths

---

## Installation

### 1) Create a fresh environment (recommended)

Using **conda**:

```bash
conda create -n pyccm python=3.11 -y
conda activate pyccm
```

### 2) Install package and dependencies

From the repo root:

```bash
pip install -e .
```

If you don’t have a `pyproject.toml` yet, you can simply install runtime deps manually:

```bash
pip install numpy pandas tqdm pyarrow
```

---

## Quickstart

1) Put your inputs under `data/` (see [Inputs & expected files](#inputs--expected-files)).

2) Adjust `config.yaml` (ranges, run mode, parallel processes, output paths).

3) Run:

```bash
cd src
python main_compute.py
```

You’ll see logs about detected ranges, tasks, and the global progress bar.  
Final combined outputs land in `results_unabridged/` (by default).

## Configuration

All behavior is controlled via `config.yaml`. Key sections:

- `paths`: input/output locations (`data_dir`, `results_dir`, CSV paths)
- `unabridging.enabled`: single-year ages + annual steps when `true`
- `projections`: years, death sources, flows year
- `fertility`:
  - `default_tfr_target`
  - `convergence_years`
  - `default_tfr_target_range` → **sweep** bounds
- `mortality`:
  - `improvement_total` (default, if not in CSV)
  - `improvement_total_range` → **sweep** bounds
  - `ma_window` and `ma_window_range` → **sweep** bounds
- `runs`:
  - `mode`: `"no_draws"` or `"draws"`
  - `no_draws_tasks`: list of omission scenarios
  - `draws`: `num_draws`, distributions, label pattern
- `parallel.processes`: worker threads per grid point
- `housekeeping.clean_run`: remove results directory before run

---

## Parameter sweeps & parallelism

PyCCM builds a **Cartesian grid** over the configured ranges:

- `fertility.default_tfr_target_range`
- `mortality.improvement_total_range`
- `mortality.ma_window_range`

For each grid point, the configured **tasks** (e.g., `mid/low/high_omissions`) are run, optionally **in parallel**.

- Total scenarios = `#grid_points × #tasks`
- Progress bar reflects `#grid_points × #tasks × #death_choices × #years`

Set worker count in `config.yaml`:

```yaml
parallel:
  processes: 4
```

> Tip: On heavy runs, start with smaller ranges to validate I/O and shapes.

---

## Inputs & expected files

Place these (optional but recommended) under `data/` or point via `paths`:

- **Mortality improvements** (per DPTO; optional):  
  `mortality_improvements.csv` with columns like:
  - `DPTO_NOMBRE`
  - `improvement_total` (fraction or %)
  - `convergence_years`, `kind` (`exp`/`logistic`), `converge_frac`, `mid_frac`, `steepness`
- **Target TFRs** (per DPTO; optional):  
  `target_tfrs.csv` (see `fertility.default_tfr_target` for global fallback)
- **Midpoint weights** (optional):  
  `midpoints.csv` with `DPTO_NOMBRE` → EEVV weight in `[0,1]`

Raw population/migration/defunction data are loaded via `data_loaders.load_all_data` from `paths.data_dir`.

---

## Outputs

All consolidated results are written to `results_dir`:

- `all_lifetables.{csv,parquet}`
- `all_asfr.{csv,parquet}`
- `all_leslie_matrices.{csv,parquet}`
- `all_projections.{csv,parquet}`

During the run, scenarios may write to `_scenarios/<name>/` and are aggregated at the end.

---

## Troubleshooting

- **“Life tables not initialized before projection.”**  
  Ensure deaths exist for the relevant `death_choice` **and** year used to build life tables:
  - `EEVV`: requires deaths up to `projections.last_observed_year_by_death.EEVV`
  - `censo_2018` & `midpoint`: require deaths at `projections.start_year`
- **0 scenarios detected**  
  Check that ranges produce at least one value each (inclusive step logic) and `runs.no_draws_tasks` isn’t empty.
- **Slow runs**  
  Reduce range sizes, restrict `DEATH_CHOICES`, or increase `parallel.processes`.
- **Missing input CSVs**  
  You’ll see warnings; the pipeline falls back to YAML defaults.
 
---

## Testing/Documentation Quick Links

**Documentations:**
- [docs/README.md](docs/README.md) - Complete documentation index
- [TESTING_SUMMARY_QUICK.md](docs/tests/TESTING_SUMMARY_QUICK.md) - 1-page executive summary
- [DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md) - Central navigation hub

**For Developers/QA:**
- [COMPREHENSIVE_TESTING_REPORT.md](docs/tests/COMPREHENSIVE_TESTING_REPORT.md) - Full test analysis (25K words)
- [docs/tests/README.md](docs/tests/README.md) - Test suite navigation
- Run tests: `pytest tests/ -v`

**For Demographers:**
- Module assessments in `docs/modules/[module]/DEMOGRAPHIC_ASSESSMENT_*.md`


**Quality Assurance:**
- **211 tests** (100% passing)
- **Python 3.7-3.11** compatibility verified

---
## Citation

If this work supports your research, please cite the repo:

```
@software{PyCCM,
  title = {PyCCM: Population projection pipeline with joint parameter sweeps},
  author = {Rahal, C., Jaramillo, J.},
  year = {2025},
  url = {https://github.com/your-org/PyCCM}
}
```

---

## License

This project is licensed under the **GNU GENERAL PUBLIC LICENSE**. See [LICENSE](LICENSE).

---

### Notes on badges

- Replace `your-org` with your actual GitHub org/user.
- Ensure `actions/workflows/ci.yml` exists for the CI badge.
- Configure Codecov (or remove the coverage badge) if not in use.
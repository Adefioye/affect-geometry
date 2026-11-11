# Research notebooks

`01_compat_error_exploration.ipynb` explores saved EXP3 first-word classification errors for
Gemma 2, OLMo 2 and Apertus. It includes confusion distributions, formatting breakdowns,
min/max/mean tables and boxplots. The executed notebook retains its tables and figures.

## Environment

The local `.venv` uses Python 3.12 and inherits the machine's existing scientific packages.
Its project-local kernel is named **Python (affect-geometry)**. From the project root:

```sh
.venv/bin/python -m jupyterlab notebooks/01_compat_error_exploration.ipynb
```

For a fresh, isolated installation (Python >=3.10):

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[notebooks]'
.venv/bin/python -m ipykernel install --prefix .venv --name affect-geometry --display-name 'Python (affect-geometry)'
.venv/bin/python -m jupyterlab notebooks/01_compat_error_exploration.ipynb
```

The notebook imports project sources directly, supports launch from the root or `notebooks/`,
and downloads only missing source tables at pinned revisions. Set `HF_TOKEN` in the project
`.env` if access requires authentication. All generated artifacts are under
`outputs/compat-errors/`, with plot-ready CSVs in `plot-ready/` and figures/report in
`analysis-report/`. These generated files and the environment are ignored by Git.

## Interpretation

The activation tables are balanced correct/incorrect samples. Response-level confusion and
formatting counts therefore describe those samples, not full-population performance.
Full-population failure counts and rates come from the saved `cohorts/support.csv` tables.
Absent sampled gold labels are explicitly marked unavailable. The first-word parser never
searches later words for an emotion label.

# Repository guide

## Project

- This is a Python 3.12 notebook project for VA measurements, including chamber and outdoor runs and timing experiments.
- The measurement workflows live in the root-level `*.ipynb` files. Measurement output belongs in `data/`, which is ignored by Git.
- `sow1_gui/` is the Python desktop replacement for the outdoor workflow. Launch with `uv run python -m sow1_gui`; use `--demo` for synthetic live data without hardware. The GUI uses PySide6 and pyqtgraph with one instrument worker.
- `pyproject.toml` defines the dependencies. `uv.lock` pins them. The VNA and triaxial positioner control packages are editable dependencies from sibling directories, `../mcg-lab-vna-control` and `../mcg-lab-triaxial-positioner-control`.

## Environment and commands

- Use **uv** to manage, run, and change the project. Run `uv sync` from the repository root to set up `.venv` from the lockfile. The sibling control packages must be present for this to succeed.
- Run Python and project tools through `uv run`, for example `uv run python -c "import numpy; print(numpy.__version__)"`. For notebook work, use the uv environment as the notebook kernel; if JupyterLab is needed locally, run `uv run --with jupyterlab jupyter lab`.
- Add or remove dependencies with `uv add <package>` or `uv remove <package>`, and commit both `pyproject.toml` and `uv.lock` when dependencies change. Do not manually edit `.venv`.

## Working conventions

- Put temporary files, scratch scripts, intermediate plots, and disposable outputs in `tmp/`. Create it if absent; it is ignored by Git.
- Put documentation requested by the user, or needed to explain project decisions and workflows, in `docs/`. Put Markdown reports in `docs/reports/`. Create these directories if absent; `docs/` is ignored by Git, so explicitly tell the user when a deliverable is stored there.
- Keep raw measurement data in `data/`. Do not commit measurement data or machine-specific configuration.
- Notebook cells may connect to and move laboratory hardware. Inspect connection settings, motion ranges, and measurement parameters before executing cells; do not run hardware-facing notebooks as a routine validation step.
- Keep notebook edits focused and avoid committing incidental execution output or metadata changes. For non-hardware changes, use targeted checks through `uv run` where practical.
- Run GUI and acquisition regression tests with `uv run python -m unittest discover -s tests -v`. Keep tests independent of real hardware. Preserve the notebook row format, measured masks, calibration guards, atomic checkpoints, and the manual positioning workflow when changing acquisition code.

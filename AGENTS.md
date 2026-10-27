# Affect Geometry Project Conventions

- Keep the project organized into clear, domain-oriented subdirectories. Do not accumulate unrelated code or artifacts at the repository root.
- Keep the codebase modular and follow clean-architecture boundaries: separate data access, scientific computation, experiment orchestration, persistence, and presentation. Keep reusable research logic out of notebooks and command-line entry points.
- Save all generated CSVs, checkpoints, plots, and other run artifacts under `outputs/`, using descriptive subdirectories where useful.
- Use the shared `affect_geometry.logging` utilities for descriptive `tqdm` progress on meaningful work units and structured run logging. Operational logs are the sole exception to the `outputs/` rule: centralize them under `logs/<model-and-run-identifier>/` and never create ad hoc log files elsewhere.
- For every essential plot, save a self-contained, tidy CSV containing all values, labels, identifiers, and experimental context required to recreate that plot independently. Keep these plot-ready CSVs separate from intermediate tables, and make plotting code consume the plot-ready CSV rather than hidden in-memory state.
- Use `$analyze-affect-outputs` when interpreting completed run outputs or producing a Markdown analysis report from saved local metrics and plots.

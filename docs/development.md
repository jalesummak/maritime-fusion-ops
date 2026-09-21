# Development and release checks

## Local environment used for release checks

Windows; Python 3.13.9; pandas 2.3.3; NumPy 2.2.6; Streamlit 1.51.0; Plotly 6.3.0; scikit-learn 1.7.2. The Dockerfile uses Python 3.12; local Python smoke checks do not substitute for a fresh Docker build.

The dependency files specify compatible ranges, not an exact environment lock. Record your installed versions before comparing numerical results.

## Commands

```bash
python -m pip install -r dashboard/requirements.txt
python -m unittest discover -s tests -v
python -m pip install -r requirements-notebooks.txt
python tools/build_simple_notebook.py
python tools/build_clean_notebook.py
python tools/build_research_figure.py
```

Notebook builders regenerate the two teaching artifacts without executing data collection or modelling. Do not use regeneration to overwrite a notebook containing your own unsaved analysis; save a separate copy first.

## Test coverage

Offline tests cover repeated collector pulls, requested-date filtering, review-rule branches, failed-request key redaction, and application startup with an explicitly empty data directory. They do not validate conflict labels, API uptime, licensed data access, model performance, or a public deployment.

Historical metrics in documentation are recorded observations from the research. API downloads, model fitting and manual case reviews are not silently rerun by tests.

## Source control

Only code, teaching notebooks without outputs, aggregate figures and documentation are published. Raw data, private environment files, logs, local checkpoints, generated archive files and marketing drafts are ignored.

No third-party data is licensed by this repository. No general reuse license for project code has been selected in this initial release.

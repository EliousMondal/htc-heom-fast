# Contributing

This repository is currently a research-code package.  The preferred workflow is:

1. Create a small issue describing the bug, benchmark, or feature.
2. Add or update a test in `tests/`.
3. Keep large simulation outputs out of git.
4. Run

```bash
pytest
python -m compileall src tests
```

before opening a pull request.

For performance-sensitive changes, include a small benchmark case and report `Nmol`, `K-matsubara`, `L`, number of ADOs, number of Numba threads, CPU model, memory usage, and wall time.

# GitHub metadata files

This repository includes a few small metadata files that make the project easier to install, cite, test, and archive.

## `LICENSE.md`

The repository uses the MIT License. This is a permissive open-source license: users may use, copy, modify, redistribute, and build on the code, provided that they keep the copyright and license notice. The license also states that the software is provided without warranty.

For many small academic research-code packages, MIT is a reasonable default when the goal is to maximize reuse and make the code easy for other researchers to build on. If your advisor, collaborators, funding agency, or university requires a different license, replace `LICENSE.md` before the public release.

## `CITATION.cff`

`CITATION.cff` tells GitHub, Zenodo, and users how to cite the software. The current file contains the software title, author, version, release date, license, repository URL, abstract, and keywords.

After the first Zenodo archive is created, add the Zenodo DOI to `CITATION.cff`.

## `.zenodo.json`

`.zenodo.json` gives Zenodo additional metadata when it archives a GitHub release. It contains the title, software description, creator, license, keywords, and related repository URL.

## `pyproject.toml`

`pyproject.toml` is the standard Python packaging configuration file. It tells Python tools:

- how to build/install the package;
- the package name and version;
- the dependencies;
- the command-line scripts;
- the optional development dependencies;
- the test and formatting configuration.

For this package, `pyproject.toml` is what makes the following command work:

```bash
python -m pip install -e .[dev]
```

It also creates the command-line programs:

```bash
htc-heom-run
htc-heom-plot
```

## `.github/workflows/tests.yml`

This file runs the tests automatically on GitHub Actions whenever you push code or open a pull request. This is useful because users can see that the package installs and passes tests on clean Linux machines.

# Releasing the repository on GitHub and Zenodo

This file gives the exact release procedure for `htc-heom-fast`.

The first public release has three goals:

1. Put the cleaned repository on GitHub.
2. Make the repository citable by adding license and citation metadata.
3. Archive a versioned release on Zenodo so that Zenodo mints a DOI.

The current metadata assumes this repository URL:

```bash
https://github.com/eliousmondal/htc-heom-fast
```

If you use a different repository name or owner, update the following files before release:

- `pyproject.toml`
- `CITATION.cff`
- `.zenodo.json`
- this file


## 1. Check the package locally

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
pytest
```

You should see all tests pass.

Run a small command-line smoke test:

```bash
htc-heom-run \
  --Nmol 5 \
  --L 4 \
  --K-matsubara 0 \
  --dt-fs 0.25 \
  --tmax-fs 100 \
  --estimate-only
```


## 2. Create the GitHub repository

### Option A: using GitHub CLI

Install and log in to the GitHub CLI once:

```bash
gh auth login
```

Then, from the repository root:

```bash
git init
git add .
git commit -m "Initial public release"
git branch -M main
gh repo create eliousmondal/htc-heom-fast --public --source=. --remote=origin --push
```

### Option B: using the GitHub website

1. Go to GitHub and create a new empty repository named `htc-heom-fast`.
2. Do not initialize it with a README, license, or `.gitignore`, because those files already exist here.
3. From the local repository root, run:

```bash
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin git@github.com:eliousmondal/htc-heom-fast.git
git push -u origin main
```

If you use HTTPS instead of SSH, use:

```bash
git remote add origin https://github.com/eliousmondal/htc-heom-fast.git
git push -u origin main
```


## 3. Confirm that GitHub recognizes the metadata

After pushing:

1. Open the GitHub repository page.
2. Confirm that GitHub shows the MIT license.
3. Confirm that GitHub shows a citation option from `CITATION.cff`.
4. Open the `Actions` tab and confirm the test workflow passes.


## 4. Enable the repository in Zenodo

1. Log in to Zenodo using GitHub.
2. Open the Zenodo GitHub integration page.
3. Find `eliousmondal/htc-heom-fast`.
4. Toggle the repository on.

Zenodo archives GitHub releases, not arbitrary commits. Therefore, enabling the repository is not enough by itself. You must create a GitHub release.


## 5. Create the first GitHub release

Use a semantic version tag. For the first public release:

```bash
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
```

Then create a GitHub release from the tag:

```bash
gh release create v0.1.0 \
  --title "v0.1.0" \
  --notes "Initial public research-code release of htc-heom-fast."
```

Alternatively, create the release from the GitHub web interface using the tag `v0.1.0`.


## 6. Wait for Zenodo and copy the DOI

After the GitHub release is created, Zenodo should archive it automatically. Wait for Zenodo to finish processing the release, then copy the DOI.

Zenodo usually gives two useful DOI forms:

- a version-specific DOI for `v0.1.0`, which should be cited for exact reproducibility;
- a concept DOI that always points to the latest version of the software record.

For papers, cite the version-specific DOI when you need reproducibility. Use the concept DOI when you want to point to the evolving software project.


## 7. Add the DOI back into the repository

After Zenodo gives the DOI, update `CITATION.cff` by adding something like:

```yaml
doi: "10.5281/zenodo.xxxxxxxx"
identifiers:
  - type: doi
    value: "10.5281/zenodo.xxxxxxxx"
    description: "Zenodo DOI for version 0.1.0"
```

Also add a DOI badge near the top of `README.md`:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxxx)
```

Then commit and push:

```bash
git add CITATION.cff README.md
git commit -m "Add Zenodo DOI"
git push
```

This DOI update can be part of a later patch release if you want the archived source itself to contain the DOI badge.

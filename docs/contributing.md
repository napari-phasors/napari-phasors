# Contributing guide

As a community-maintained project, napari-phasors welcomes contributions in
the form of bug reports, bug fixes, feature implementations, documentation,
sample data, and enhancement proposals. This guide explains how to contribute.

The [Code of Conduct](code_of_conduct.md) applies to everyone participating
in the napari-phasors community.

## Ask for help

To ask questions about using napari-phasors, open a [GitHub
issue](https://github.com/napari-phasors/napari-phasors/issues).

## Propose enhancements

To suggest a new feature or other improvement, open a [GitHub
issue](https://github.com/napari-phasors/napari-phasors/issues) describing
the use case before starting work on a pull request.

## Report bugs

To report a bug, please open a [GitHub
issue](https://github.com/napari-phasors/napari-phasors/issues) and include:

- A minimal, self-contained script or set of steps reproducing the problem.
- A traceback, if available.
- The file(s) needed to reproduce the issue (or a sample file of the same
  format), if the bug is related to reading or writing data.
- An explanation of why the current behavior is wrong and what is expected
  instead.
- The napari-phasors, napari, and Python versions in use, and how
  napari-phasors was installed (napari plugin manager, standalone installer,
  conda/pip, or development version).

## Contribute code or documentation

The source code for napari-phasors is hosted in a GitHub repository at
<https://github.com/napari-phasors/napari-phasors>.

napari-phasors uses GitHub's [fork and pull collaborative development
model](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests).
All contributions should be developed in a personal fork of the repository
and submitted as [pull
requests](https://github.com/napari-phasors/napari-phasors/pulls).

### Fork and clone the repository

Fork the repository by pressing the "Fork" button at
<https://github.com/napari-phasors/napari-phasors>, then clone the personal
fork:

```bash
git clone https://github.com/your-user-name/napari-phasors.git
cd napari-phasors
git remote add upstream https://github.com/napari-phasors/napari-phasors.git
```

There are now two remote repositories: `upstream`, which refers to the
napari-phasors repository, and `origin`, which refers to the personal fork.

### Create a development environment

We recommend using [miniforge](https://conda-forge.org/download/) to create
an isolated environment, then installing napari-phasors in editable mode
with its testing dependencies:

```bash
mamba create -y -n napari-phasors-dev napari pyqt6 python=3.12
conda activate napari-phasors-dev
pip install -e ".[testing]"
```

Verify that the development environment is working by running the tests:

```bash
pytest -v
```

### Create a branch

Before implementing any changes, consider opening a [GitHub
issue](https://github.com/napari-phasors/napari-phasors/issues) to discuss
the bug fix or feature being worked on.

Synchronize the personal fork with the upstream repository, then create a
new branch for each bug fix or feature:

```bash
git checkout main
git fetch upstream
git rebase upstream/main
git push
git checkout -b my-new-feature
```

### Set up pre-commit hooks

napari-phasors uses [pre-commit](https://pre-commit.com/) to run **black**,
**isort**, and **ruff** automatically on every commit:

```bash
pip install pre-commit
pre-commit install
```

From then on, every `git commit` auto-formats and lints the changed files
before the commit goes through. The hooks can also be run manually on all
files:

```bash
pre-commit run --all-files
```

### Run the tests

Tests are run with [pytest](https://docs.pytest.org/) and, across the
supported Python and Qt backend combinations, with
[tox](https://tox.readthedocs.io/en/latest/):

```bash
pytest -v --cov=napari_phasors
tox
```

Please ensure the test coverage at least stays the same before submitting a
pull request.

### Submit a pull request

Push the branch to the personal fork and open a [pull
request](https://github.com/napari-phasors/napari-phasors/pulls) against the
`main` branch of the upstream repository. Describe the motivation and
content of the change, and link any related issue.

A maintainer will review the pull request, which may involve requesting
changes before it is merged. The
[tests](https://github.com/napari-phasors/napari-phasors/actions/workflows/run-tests.yml)
and [codecov](https://codecov.io/gh/napari-phasors/napari-phasors) checks
must pass.

## Share data files

Consider sharing sample datasets for testing and use in tutorials by opening
a [GitHub issue](https://github.com/napari-phasors/napari-phasors/issues).

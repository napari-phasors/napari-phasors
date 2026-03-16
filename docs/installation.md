# Installation

napari-phasors can be installed in several ways depending on your experience level. Choose the option that best fits your needs.

---

## Option 1: napari's built-in plugin manager (easiest)

If you already have [napari](https://napari.org) installed, this is the simplest way:

1. Open napari
2. Go to **Plugins → Install/Uninstall Plugins...**
3. Search for **napari-phasors**
4. Click **Install**
5. Restart napari

That's it — no terminal or command line needed.

```{note}
If you don't have napari yet, see [Option 2](#option-2-using-conda--pip-recommended) to install both napari and napari-phasors together.
```

---

## Option 2: Using conda + pip (recommended)

This is the recommended method for most users. It creates an isolated
environment so napari-phasors won't interfere with other software on your
computer.

### Step 1: Install miniforge

Download and install [miniforge](https://conda-forge.org/download/) for your
operating system. Miniforge provides the `mamba` package manager.

```{tip}
- **Windows**: After installing, open **Miniforge Prompt** from the Start menu.
- **macOS**: Open **Terminal** (found in Applications → Utilities).
- **Linux**: Open any terminal application.
```

If you already have Anaconda or Miniconda, you can use `conda` instead of
`mamba` in all commands below.

### Step 2: Create an environment

```bash
mamba create -y -n napari-phasors-env napari pyqt python=3.12
```

### Step 3: Activate the environment

```bash
mamba activate napari-phasors-env
```

```{important}
You need to activate the environment **every time** you open a new terminal
before using napari-phasors.
```

### Step 4: Install napari-phasors

```bash
pip install napari-phasors
```

### Step 5: Launch

```bash
napari
```

Then find the plugin under **Plugins → napari-phasors**.

---

## Option 3: Standalone installer (no Python required)

Pre-built installers for Windows, macOS, and Linux are available on the
[GitHub Releases](https://github.com/napari-phasors/napari-phasors/releases)
page. Download the installer for your platform, run it, and you're ready to
go — no Python installation needed.

```{note}
Standalone installers are generated automatically for each release. If no
installer is available for the latest version yet, use Option 1 or 2.
```

---

## Updating

To update to the latest version:

```bash
mamba activate napari-phasors-env
pip install --upgrade napari-phasors
```

Or through napari's plugin manager: **Plugins → Install/Uninstall Plugins...**
→ click **Update** next to napari-phasors.

---

## Development installation

For contributors who want to modify the code:

```bash
git clone https://github.com/napari-phasors/napari-phasors.git
cd napari-phasors
pip install -e ".[testing]"
```

### Pre-commit hooks

The project uses [pre-commit](https://pre-commit.com/) to enforce **black**,
**isort**, and **ruff** on every commit:

```bash
pip install pre-commit
pre-commit install
```

### Running tests

```bash
pytest src/napari_phasors/_tests/
```

Or in parallel:

```bash
pytest src/napari_phasors/_tests/ -n auto
```

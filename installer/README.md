# Installer assets

Scripts and assets used by [constructor](https://github.com/conda/constructor)
to build standalone installers for napari-phasors.

## Layout of a built bundle

The installer creates two conda environments, mirroring
[napari's own installer](https://github.com/napari/packaging):

```
<install root>/                     base env: python + conda tooling only
<install root>/envs/napari-phasors  app env:  napari + napari-phasors + Qt
<install root>/napari-phasors       launcher (napari-phasors.bat on Windows)
<install root>/launcher-env.sh      shared environment setup for the launcher
<install root>/conda-launcher       relocatable `conda` entry point (Unix)
<install root>/.condarc             channel pinning for plugin installs
<install root>/.napari_is_bundled_constructor
```

Two things depend on that split:

* `napari.utils.misc.running_as_constructor_app()` looks for
  `.napari_is_bundled_constructor` at `sys.prefix.parent.parent`, which only
  resolves to the install root when the app lives under `envs/`. It gates the
  "installing from PyPI can break your bundle" warning and the tagging of
  plugins that have no conda-forge build.
* `napari-plugin-manager` shells out to `conda` to install and update plugins.
  Keeping conda in the base env means it is never rewriting the environment it
  is itself running from. A single-environment bundle with no conda inside is
  why plugin installs used to fail silently.

`launcher-env.sh` / `launcher-env.bat` are the single source of truth for the
runtime environment: they point `CONDA_EXE` at the bundled conda, clear
`MAMBA_EXE`/`CONDA` so a conda active in the user's shell can never operate on
this prefix, and set the Qt plugin paths. The CI smoke tests source the same
files, so they check the environment users actually get.

## Offline installation

Everything in `specs` and `extra_envs` is resolved and embedded in the
installer at build time, and both the shell and NSIS installers link packages
with `conda install --offline`. Installing therefore needs no network; the CI
smoke tests run the installers with the proxy pointed at a closed port to keep
it that way. Installing or upgrading plugins later from inside the app does
need a connection.

## Files

| File | Role |
| --- | --- |
| `make_construct.py` | Generates `construct.yaml` for the current platform |
| `post_install.sh` / `post_install.bat` | Shortcuts and desktop entries |
| `napari-phasors` / `napari-phasors.bat` | Application launcher |
| `launcher-env.sh` / `launcher-env.bat` | Runtime environment setup |
| `conda-launcher` | Relocation-proof `conda` entry point (Unix) |
| `verify_installation.py` | Post-install checks run by CI on all platforms |

`construct.yaml`, `LICENSE.ascii.txt`, `icon.ico`, `app.icns` and
`napari_is_bundled_constructor` are generated at build time and git-ignored.

## macOS notes

The macOS job installs the `.sh` installer, moves the whole install root into
`napari-phasors.app/Contents/Resources/env`, and ships that in a DMG. Console
scripts such as `bin/conda` bake the absolute install-time prefix into their
shebang and break once the tree moves, so nothing invokes them: napari and
conda are both started as `python -m`, which resolves the interpreter by path.

The app is not signed or notarized. Drag it to `/Applications` (or
`~/Applications` without admin rights) before first launch - running it from
the mounted DMG can trigger Gatekeeper app translocation, which puts it on a
read-only path where plugin installs cannot write.

## Adding an installer icon

Place an `icon.png` file in this directory; `make_construct.py` detects it and
converts it to the per-platform formats (`icon.ico` on Windows, `app.icns` on
macOS).

- **Format:** PNG (constructor expects PNG for `icon_image`).
- **Recommended size:** 256 x 256 pixels or larger.
- If you only have a Windows `.ico` file, convert it to PNG first
  (e.g. with ImageMagick: `magick icon.ico icon.png`).

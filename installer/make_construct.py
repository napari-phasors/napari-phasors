"""Generate ``construct.yaml`` for the napari-phasors standalone installers.

The bundle uses the same two-environment layout as napari's own installer::

    <root>/                     base env: python + conda tooling only
    <root>/envs/napari-phasors  app env:  napari + napari-phasors + Qt

Two things depend on that layout:

* ``napari.utils.misc.running_as_constructor_app()`` looks for
  ``.napari_is_bundled_constructor`` at ``sys.prefix.parent.parent``, which
  only resolves to the install root when the app lives under ``envs/``. It
  gates the "installing from PyPI can break your bundle" warning and the
  tagging of plugins that have no conda-forge build.
* ``napari-plugin-manager`` shells out to ``conda`` to install and update
  plugins. Keeping conda in the base env means it is never rewriting the
  environment it is itself running from.

Everything listed in ``specs`` and ``extra_envs`` is resolved and embedded in
the installer at build time, and both the shell and NSIS installers link the
packages with ``conda install --offline``. Installation therefore needs no
network; only plugin installs and upgrades made later from inside the app do.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import yaml

HERE = Path(__file__).parent
WINDOWS = sys.platform.startswith('win')
MACOS = sys.platform == 'darwin'

# Must match the path the launcher scripts compute.
APP_ENV_NAME = 'napari-phasors'
PY_VER = '3.12'

# Kept in the base env only. `conda` is required there anyway for
# `extra_envs` to work at all, and `--solver=libmamba` is always passed by
# napari-plugin-manager, so the solver plugin has to be alongside it.
CONDA_TOOL_SPECS = ['conda >=23.10', 'conda-libmamba-solver']

# Written to <root>/.condarc. The `#!final` markers stop a user-level
# ~/.condarc from redirecting the channels plugin installs are solved against.
CONDARC = """\
channels:  #!final
  - conda-forge
channel_priority: strict  #!final
auto_update_conda: false  #!final
notify_outdated_conda: false  #!final
"""


def app_specs(version: str) -> list[str]:
    """Specs for the environment napari actually runs from."""
    return [
        f'python={PY_VER}',
        'napari',
        # Provides the Plugins > Install/Uninstall Plugins... dialog. It comes
        # in via the `napari` metapackage today; pinned here so a change to
        # that recipe cannot silently remove the menu entry.
        'napari-plugin-manager',
        'pyqt6',
        # The PyPI backends of the plugin manager: it prefers uv and falls
        # back to pip, and resolves both from sys.prefix (this env).
        'pip',
        'uv',
        f'napari-phasors={version}',
    ]


def write_ascii_license() -> str:
    """Shell installers require a plain-ASCII license file."""
    text = (HERE.parent / 'LICENSE').read_text(encoding='utf-8')
    ascii_text = (
        unicodedata.normalize('NFKD', text)
        .encode('ascii', 'ignore')
        .decode('ascii')
    )
    target = HERE / 'LICENSE.ascii.txt'
    target.write_text(ascii_text, encoding='ascii')
    return target.name


def build_extra_files() -> list[str | dict[str, str]]:
    """Files copied verbatim into the root of the install prefix."""
    marker = HERE / 'napari_is_bundled_constructor'
    marker.touch()
    extra_files: list[str | dict[str, str]] = [
        {marker.name: '.napari_is_bundled_constructor'}
    ]

    if WINDOWS:
        extra_files += ['launcher-env.bat', 'napari-phasors.bat']
    else:
        extra_files += ['launcher-env.sh', 'napari-phasors', 'conda-launcher']

    icon = HERE / 'icon.png'
    if icon.is_file():
        # Used as-is for the Linux .desktop entry; converted for the Windows
        # shortcuts and the macOS .app, so only icon.png lives in the repo.
        extra_files.append(icon.name)
        if WINDOWS or MACOS:
            from PIL import Image

            image = Image.open(icon)
            converted = 'icon.ico' if WINDOWS else 'app.icns'
            image.save(HERE / converted, format='ICO' if WINDOWS else 'ICNS')
            extra_files.append(converted)

    return extra_files


def definitions(version: str) -> dict:
    defs = {
        'name': 'napari-phasors',
        'version': version,
        'channels': ['conda-forge'],
        'specs': [f'python={PY_VER}', *CONDA_TOOL_SPECS],
        'extra_envs': {
            APP_ENV_NAME: {
                'specs': app_specs(version),
                # Shortcuts are created by the post-install scripts.
                'menu_packages': [],
            }
        },
        'menu_packages': [],
        # Never touch the user's shell profiles, PATH or registry: the bundle
        # is an application, not a conda installation to work from.
        'initialize_conda': False,
        'initialize_by_default': False,
        'register_envs': False,
        'condarc': CONDARC,
        'license_file': write_ascii_license(),
        'post_install': 'post_install.bat' if WINDOWS else 'post_install.sh',
        'extra_files': build_extra_files(),
    }

    if (HERE / 'icon.png').is_file():
        defs['icon_image'] = 'icon.png'

    if WINDOWS:
        defs.update(
            {
                'register_python': False,
                'register_python_default': False,
                # Default to a user-writable location: plugin installs write
                # into the prefix, so an elevated one would need admin rights
                # for every update.
                'default_prefix': r'%LOCALAPPDATA%\napari-phasors',
                'default_prefix_domain_user': r'%LOCALAPPDATA%\napari-phasors',
                'default_prefix_all_users': r'%ALLUSERSPROFILE%\napari-phasors',
            }
        )

    return defs


def _str_presenter(dumper, data):
    """Emit multi-line strings as block scalars so the file stays readable."""
    style = '|' if '\n' in data else None
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style=style)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--version', required=True, help='napari-phasors version to bundle'
    )
    args = parser.parse_args()

    yaml.add_representer(str, _str_presenter, Dumper=yaml.SafeDumper)
    target = HERE / 'construct.yaml'
    target.write_text(
        yaml.safe_dump(definitions(args.version), sort_keys=False),
        encoding='utf-8',
    )
    print(target.read_text(encoding='utf-8'))
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Verify a built standalone installer produces a working, updatable bundle.

Run this with the *app environment's* python, after sourcing/calling the
launcher environment (``launcher-env.sh`` or ``launcher-env.bat``), so it sees
exactly what napari sees when a user starts the app::

    . "${PREFIX}/launcher-env.sh"
    "${NAPARI_PHASORS_ENV}/bin/python" installer/verify_installation.py

It fails loudly if the plugin manager would be unable to install or update
plugins, which is otherwise invisible until a user tries it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = '') -> bool:
    print(f'{"PASS" if ok else "FAIL"}  {name}')
    if detail:
        print(f'        {detail}')
    if not ok:
        FAILURES.append(name)
    return ok


def check_layout(prefix: Path, root: Path) -> None:
    check(
        'app env lives in <root>/envs/napari-phasors',
        prefix.parent.name == 'envs' and prefix.name == 'napari-phasors',
        f'got {prefix}',
    )
    check('base env is a conda prefix', (root / 'conda-meta').is_dir())
    check('bundled .condarc present', (root / '.condarc').is_file())
    check(
        'constructor bundle marker present',
        (root / '.napari_is_bundled_constructor').is_file(),
        'napari looks for this at sys.prefix.parent.parent',
    )


def check_app() -> None:
    import napari

    import napari_phasors

    print(f'napari  : {napari.__version__}')
    print(f'phasors : {napari_phasors.__version__}')

    from napari.utils.misc import running_as_constructor_app

    check(
        'napari detects the constructor bundle',
        running_as_constructor_app(),
        'gates the PyPI-install warning and conda availability tagging',
    )

    try:
        from napari.utils._env_detection import (
            Environment,
            detect_environment,
        )
    except ImportError:  # napari < 0.8
        pass
    else:
        check(
            'napari detects a conda environment',
            detect_environment() == Environment.conda,
            f'got {detect_environment()}',
        )


def check_qt() -> None:
    """Qt must find its platform plugin inside the bundle.

    A wrong QT_PLUGIN_PATH is the classic "installer builds fine but the app
    never opens" failure, and it is silent: Qt aborts the process before
    anything can report it. Creating a QApplication is the assertion - if the
    plugin cannot be loaded, this call kills the interpreter and the check
    never prints, failing the job.

    Deliberately no OpenGL here. `napari --info` probes GL through vispy,
    which segfaults under the offscreen platform on headless runners.
    """
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    check(
        'Qt platform plugin loads',
        bool(app.platformName()),
        f'platform={app.platformName()!r} '
        f'QT_QPA_PLATFORM={os.environ.get("QT_QPA_PLATFORM", "<unset>")} '
        f'QT_PLUGIN_PATH={os.environ.get("QT_PLUGIN_PATH", "<unset>")}',
    )


def check_plugin_manager(prefix: Path, root: Path) -> None:
    check(
        'napari_plugin_manager importable (Plugins menu entry)',
        find_spec('napari_plugin_manager') is not None,
    )

    from napari_plugin_manager.base_qt_package_installer import (
        InstallerActions,
    )
    from napari_plugin_manager.qt_package_installer import (
        NapariCondaInstallerTool as Conda,
    )
    from napari_plugin_manager.qt_package_installer import (
        NapariInstallerQueue,
    )
    from napari_plugin_manager.qt_package_installer import (
        NapariPipInstallerTool as Pip,
    )
    from napari_plugin_manager.qt_package_installer import (
        NapariUvInstallerTool as Uv,
    )
    from napari_plugin_manager.utils import is_conda_package

    check(
        'napari is recognised as a conda package',
        is_conda_package('napari'),
        'this is what makes the dialog default to the Conda backend',
    )

    conda_exe = Path(Conda.executable())
    check(
        'CONDA_EXE points inside the bundle',
        conda_exe.is_file() and root in conda_exe.resolve().parents,
        f'CONDA_EXE={os.environ.get("CONDA_EXE", "<unset>")}',
    )
    check(
        'MAMBA_EXE / CONDA are not inherited from the host',
        not os.environ.get('MAMBA_EXE') and not os.environ.get('CONDA'),
        f'MAMBA_EXE={os.environ.get("MAMBA_EXE", "")!r} '
        f'CONDA={os.environ.get("CONDA", "")!r}',
    )
    if check('conda backend runs', Conda.available(), str(conda_exe)):
        proc = subprocess.run(
            [str(conda_exe), '--version'], capture_output=True, text=True
        )
        print(f'        {proc.stdout.strip()}')

    # The job the dialog would actually queue for a plugin install: it must
    # target the app env, not the base env conda itself runs from.
    try:
        args = Conda(
            action=InstallerActions.INSTALL, pkgs=('napari-plugin-example',)
        ).arguments()
    except ValueError as exc:  # no conda-meta in sys.prefix
        check('conda installs target the app env', False, str(exc))
    else:
        check(
            'conda installs target the app env',
            '--prefix' in args
            and Path(args[args.index('--prefix') + 1]).resolve() == prefix,
            ' '.join(args),
        )

    check('pip backend runs', Pip.available(), Pip.executable())
    # Uv.executable() only probes <prefix>/bin/uv (Scripts\uv.exe on Windows)
    # and otherwise falls back to a bare 'uv'. conda-forge puts uv in
    # Library\bin on Windows, so there the bundled one is found through PATH
    # rather than by path - resolve it the same way the subprocess would, and
    # confirm what wins is ours and not some unrelated uv on the host.
    uv_exe = shutil.which(Uv.executable())
    check(
        'uv backend runs from the app env',
        Uv.available()
        and uv_exe is not None
        and prefix in Path(uv_exe).resolve().parents,
        str(uv_exe),
    )
    check(
        'PyPI installs use uv',
        NapariInstallerQueue.PYPI_INSTALLER_TOOL_CLASS is Uv,
        f'got {NapariInstallerQueue.PYPI_INSTALLER_TOOL_CLASS.__name__}',
    )


def main() -> int:
    prefix = Path(sys.prefix).resolve()
    root = prefix.parent.parent

    print(f'python  : {sys.executable}')
    print(f'prefix  : {prefix}')
    print(f'root    : {root}')
    print()

    check_layout(prefix, root)
    check_app()
    check_qt()
    check_plugin_manager(prefix, root)

    print()
    if FAILURES:
        print(f'{len(FAILURES)} check(s) failed: {", ".join(FAILURES)}')
        return 1
    print('All checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

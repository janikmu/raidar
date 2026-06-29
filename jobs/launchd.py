"""Install/remove the macOS launchd agents that run enrich + digest on schedule.

Thin wrapper around infra/install_launchd.sh — that script stays the source of
truth (and what infra/smoke.sh lints), this just makes it reachable from the
`raidar` CLI so it isn't a one-off step you can only run right after cloning.

Usage:
    raidar install-launchd               # install / reinstall
    raidar install-launchd --uninstall   # unload + remove plists
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)

_PLISTS = ("com.airadar.enrich.plist", "com.airadar.digest.plist")
EXPECTED_LABELS = tuple(p[: -len(".plist")] for p in _PLISTS)


def repo_root() -> Path | None:
    """Locate the cloned repo containing infra/install_launchd.sh.

    Works for the editable install the README recommends (`uv tool install
    --editable .`), since jobs/ then still resolves into the source checkout.
    """
    candidate = Path(__file__).resolve().parent.parent
    if (candidate / "infra" / "install_launchd.sh").is_file():
        return candidate
    return None


def installed_plists() -> list[str]:
    """Labels (no .plist suffix) of the agents currently present in LaunchAgents."""
    la_dir = Path.home() / "Library" / "LaunchAgents"
    return sorted(label for label, plist in zip(EXPECTED_LABELS, _PLISTS) if (la_dir / plist).is_file())


def run_install_script(*args: str) -> int:
    root = repo_root()
    if root is None:
        print(
            "ERROR: could not find infra/install_launchd.sh next to this install.\n"
            "install-launchd needs an editable install from the cloned repo\n"
            "(uv tool install --editable .) — reinstall that way, or run\n"
            "./infra/install_launchd.sh directly from the repo.",
            file=sys.stderr,
        )
        return 2
    result = subprocess.run(["bash", str(root / "infra" / "install_launchd.sh"), *args])
    return result.returncode


@app.command("install-launchd")
def install_launchd(
    uninstall: bool = typer.Option(
        False, "--uninstall", help="Unload and remove the agents instead of installing them.",
    ),
) -> None:
    """Install (or remove) the macOS launchd agents for enrich + digest."""
    if platform.system() != "Darwin":
        print("launchd is macOS-only — on Linux, configure cron or systemd instead.", file=sys.stderr)
        raise typer.Exit(code=1)
    code = run_install_script("uninstall" if uninstall else "install")
    raise typer.Exit(code=code)


if __name__ == "__main__":
    app()

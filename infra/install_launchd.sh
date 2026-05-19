#!/usr/bin/env bash
# Install AI Radar launchd agents (enrich + digest).
#
# Substitutes __TOOL_DIR__ and __UV__ in the plist templates and copies them
# to ~/Library/LaunchAgents. Bootstraps them via launchctl. Re-running is
# safe — old versions are unloaded first.
#
# Usage:
#   ./infra/install_launchd.sh             # install / reinstall
#   ./infra/install_launchd.sh uninstall   # unload + remove plists
#
# Verify with:
#   launchctl list | grep airadar

set -euo pipefail

TOOL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UV_PATH="$(command -v uv || true)"
LA_DIR="$HOME/Library/LaunchAgents"
PLISTS=("com.airadar.enrich.plist" "com.airadar.digest.plist")

if [[ -z "${UV_PATH}" ]]; then
    echo "error: uv not found in PATH" >&2
    exit 1
fi

uninstall() {
    for plist in "${PLISTS[@]}"; do
        target="${LA_DIR}/${plist}"
        if [[ -f "${target}" ]]; then
            label="${plist%.plist}"
            echo "unloading ${label}"
            launchctl bootout "gui/$(id -u)/${label}" 2>/dev/null || true
            rm -f "${target}"
        fi
    done
    echo "uninstalled."
}

install() {
    mkdir -p "${LA_DIR}"
    mkdir -p "${TOOL_DIR}/logs"
    for plist in "${PLISTS[@]}"; do
        src="${TOOL_DIR}/infra/launchd/${plist}"
        target="${LA_DIR}/${plist}"
        label="${plist%.plist}"
        echo "installing ${label}"
        # Bootout any existing instance first so updates take effect.
        launchctl bootout "gui/$(id -u)/${label}" 2>/dev/null || true
        # Substitute placeholders and write to LaunchAgents.
        sed -e "s|__TOOL_DIR__|${TOOL_DIR}|g" -e "s|__UV__|${UV_PATH}|g" "${src}" > "${target}"
        launchctl bootstrap "gui/$(id -u)" "${target}"
    done
    echo
    echo "installed. verify with: launchctl list | grep airadar"
    echo "next enrich fires Sunday 20:00; digest at 21:00."
}

cmd="${1:-install}"
case "${cmd}" in
    install) install ;;
    uninstall) uninstall ;;
    *)
        echo "usage: $0 [install|uninstall]" >&2
        exit 2
        ;;
esac

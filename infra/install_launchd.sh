#!/usr/bin/env bash
# Install AI Radar launchd agents (enrich + digest).
#
# Safeguard: checks if the OS is macOS (Darwin).
# Detects if a global `raidar` binary is available in PATH. If so, templates
# the plists to call it directly. Otherwise, falls back to `uv run raidar`
# inside the cloned repository.
#
# Copies plist files to ~/Library/LaunchAgents and bootstraps them via launchctl.
#
# Usage:
#   ./infra/install_launchd.sh             # install / reinstall
#   ./infra/install_launchd.sh uninstall   # unload + remove plists
#
# Verify with:
#   launchctl list | grep airadar

set -euo pipefail

# 1. OS Platform Safeguard
if [[ "$(uname)" != "Darwin" ]]; then
    echo "error: launchd is macOS-only. For Linux or other systems, please configure cron or systemd." >&2
    exit 1
fi

TOOL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UV_PATH="$(command -v uv || true)"
RAIDAR_PATH="$(command -v raidar || true)"
LA_DIR="$HOME/Library/LaunchAgents"
PLISTS=("com.airadar.enrich.plist" "com.airadar.digest.plist")

if [[ -z "${UV_PATH}" ]]; then
    echo "error: uv not found in PATH" >&2
    exit 1
fi

# Load vault path from active configuration using Python
VAULT_PATH=""
if [[ -f "${TOOL_DIR}/lib/config.py" ]]; then
    VAULT_PATH="$(uv run --project "${TOOL_DIR}" python -c "from lib import config; print(config.load().vault_path)" 2>/dev/null || true)"
fi

if [[ -z "${VAULT_PATH}" ]]; then
    VAULT_PATH="$HOME/raidar-vault"
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
    mkdir -p "${VAULT_PATH}/logs"
    
    local exec_args=""
    local working_dir=""
    
    if [[ -n "${RAIDAR_PATH}" ]]; then
        echo "Found global raidar at: ${RAIDAR_PATH}"
        working_dir="${VAULT_PATH}"
    else
        echo "Global raidar not found in PATH. Using local uv-run fallback inside: ${TOOL_DIR}"
        working_dir="${TOOL_DIR}"
    fi

    for plist in "${PLISTS[@]}"; do
        src="${TOOL_DIR}/infra/launchd/${plist}"
        target="${LA_DIR}/${plist}"
        label="${plist%.plist}"
        echo "installing ${label}"
        
        # Build execution arguments
        if [[ "${label}" == "com.airadar.enrich" ]]; then
            if [[ -n "${RAIDAR_PATH}" ]]; then
                exec_args="        <string>${RAIDAR_PATH}</string>\n        <string>enrich</string>"
            else
                exec_args="        <string>${UV_PATH}</string>\n        <string>run</string>\n        <string>raidar</string>\n        <string>enrich</string>"
            fi
        else
            if [[ -n "${RAIDAR_PATH}" ]]; then
                exec_args="        <string>${RAIDAR_PATH}</string>\n        <string>digest</string>"
            else
                exec_args="        <string>${UV_PATH}</string>\n        <string>run</string>\n        <string>raidar</string>\n        <string>digest</string>"
            fi
        fi

        # Bootout any existing instance first so updates take effect.
        launchctl bootout "gui/$(id -u)/${label}" 2>/dev/null || true
        
        # Substitute placeholders and write to LaunchAgents.
        # Uses python to handle multiline substitution elegantly without sed parsing issues.
        python3 -c "
import sys
content = open('$src').read()
content = content.replace('<string>__PROGRAM_ARGUMENTS__</string>', '''$exec_args''')
content = content.replace('__WORKING_DIRECTORY__', '$working_dir')
content = content.replace('__LOG_DIR__', '$VAULT_PATH/logs')
open('$target', 'w').write(content)
"
        launchctl bootstrap "gui/$(id -u)" "${target}"
    done
    echo
    echo "installed. verify with: launchctl list | grep airadar"
    echo "launchd logs will be written to: ${VAULT_PATH}/logs/"
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

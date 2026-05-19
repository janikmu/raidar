#!/usr/bin/env bash
# Make the repo's SKILL.md available to Claude Code / Cowork globally by
# symlinking it into ~/.claude/skills/ai-radar/. Since it's a symlink, every
# edit to the repo's SKILL.md is immediately reflected — no re-install.
#
# Cowork auto-discovers SKILL.md at a project root, so this script is only
# needed if you want the skill available in OTHER Claude projects too.
#
# Usage:
#   ./infra/install_skill.sh             # install / re-install (symlink)
#   ./infra/install_skill.sh uninstall   # remove the symlink

set -euo pipefail

TOOL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${TOOL_DIR}/SKILL.md"
SKILL_NAME="ai-radar"
DEST_DIR="${HOME}/.claude/skills/${SKILL_NAME}"
DEST="${DEST_DIR}/SKILL.md"

if [[ ! -f "${SRC}" ]]; then
    echo "error: ${SRC} not found" >&2
    exit 1
fi

cmd="${1:-install}"
case "${cmd}" in
    install)
        mkdir -p "${DEST_DIR}"
        # `ln -sfn` replaces an existing symlink atomically; doesn't follow it.
        ln -sfn "${SRC}" "${DEST}"
        echo "installed: ${DEST} -> ${SRC}"
        ;;
    uninstall)
        if [[ -L "${DEST}" ]]; then
            rm "${DEST}"
            echo "removed: ${DEST}"
        else
            echo "no symlink at ${DEST}"
        fi
        # Remove the directory only if it's now empty.
        rmdir "${DEST_DIR}" 2>/dev/null || true
        ;;
    *)
        echo "usage: $0 [install|uninstall]" >&2
        exit 2
        ;;
esac

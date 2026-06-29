#!/usr/bin/env bash
# Offline smoke test for AI Radar.
#
# Runs every check that does NOT require API keys or live backends:
#   - All lib imports work.
#   - Each lib's __main__ smoke test passes (or skips gracefully when the
#     backend is offline — embeddings + llm).
#   - Each job CLI responds to --help.
#   - The capture job's --dry-run path assembles a prompt without writing.
#   - launchd plists pass plutil -lint and the install script is syntactically valid.
#
# Run from the tool repo root:  bash infra/smoke.sh

set -u  # NOT -e: we want to report failures, not abort on the first one.

TOOL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${TOOL_DIR}"

PASS=0
FAIL=0
SKIP=0

check() {
    local label="$1"; shift
    if "$@" >/tmp/airadar_smoke.out 2>&1; then
        printf "  \033[32mok\033[0m   %s\n" "${label}"
        PASS=$((PASS + 1))
    else
        printf "  \033[31mfail\033[0m %s\n" "${label}"
        sed 's/^/        /' /tmp/airadar_smoke.out | head -20
        FAIL=$((FAIL + 1))
    fi
}

check_skip_ok() {
    # Treats specific exit-zero "skipped" output as a skip rather than a pass.
    local label="$1"; shift
    if "$@" >/tmp/airadar_smoke.out 2>&1; then
        if grep -qiE "skipped|unreachable" /tmp/airadar_smoke.out; then
            printf "  \033[33mskip\033[0m %s (backend offline)\n" "${label}"
            SKIP=$((SKIP + 1))
        else
            printf "  \033[32mok\033[0m   %s\n" "${label}"
            PASS=$((PASS + 1))
        fi
    else
        printf "  \033[31mfail\033[0m %s\n" "${label}"
        sed 's/^/        /' /tmp/airadar_smoke.out | head -20
        FAIL=$((FAIL + 1))
    fi
}

echo "AI Radar offline smoke test"
echo "---------------------------"

echo "[imports]"
check "lib.config + lib.secrets + lib.logging_setup load" \
    uv run python -c "from lib import config, secrets, logging_setup; config.load()"
check "lib.vault imports" uv run python -c "from lib import vault"
check "lib.llm imports"   uv run python -c "from lib import llm"
check "lib.embeddings imports" uv run python -c "from lib import embeddings"
check "lib.github imports" uv run python -c "from lib import github"
check "lib.body imports" uv run python -c "from lib import body"
check "all jobs import"   uv run python -c "from jobs import capture, bulk_capture, enrich, digest, search, backfill, reevaluate, seed, cli"

echo "[lib smoke tests]"
check "lib.vault smoke" uv run python -m lib.vault
check_skip_ok "lib.embeddings smoke" uv run python -m lib.embeddings
check_skip_ok "lib.llm smoke" uv run python -m lib.llm
check "lib.github smoke (uses network)" uv run python -m lib.github

echo "[CLI wiring]"
check "raidar --help"  uv run raidar --help
check "capture --help" uv run python -m jobs.capture --help
check "enrich --help"  uv run python -m jobs.enrich --help
check "digest --help"  uv run python -m jobs.digest --help
check "search --help"  uv run python -m jobs.search --help
check "backfill --help" uv run python -m jobs.backfill --help
check "reevaluate --help" uv run python -m jobs.reevaluate --help
check "seed --help"    uv run python -m jobs.seed --help
check "seed --list (no LLM)" uv run python -m jobs.seed --list
check "health --help"  uv run python -m jobs.health --help
check "merge-concept --help" uv run python -m jobs.merge --help
check "rename-concept --help" uv run python -m jobs.rename --help
check "reindex --help" uv run python -m jobs.reindex --help
check "install-launchd --help" uv run python -m jobs.launchd --help
check "search list-concepts against empty vault" uv run python -m jobs.search list-concepts
check "health against empty vault" uv run python -m jobs.health

echo "[capture dry-run]"
check "capture --dry-run on free text" \
    uv run python -m jobs.capture --dry-run "smoke test note about dspy"

echo "[infra]"
check "install_launchd.sh syntax" bash -n infra/install_launchd.sh
check "enrich.plist validates"   plutil -lint infra/launchd/com.airadar.enrich.plist
check "digest.plist validates"   plutil -lint infra/launchd/com.airadar.digest.plist

echo
echo "Summary: ${PASS} ok, ${FAIL} fail, ${SKIP} skipped"
echo
if [[ ${SKIP} -gt 0 ]]; then
    echo "Skipped checks indicate a backend is offline (LMStudio or the academic"
    echo "proxy). That's fine for an initial setup smoke test — bring backends up"
    echo "and re-run to exercise the full path."
fi
[[ ${FAIL} -eq 0 ]]

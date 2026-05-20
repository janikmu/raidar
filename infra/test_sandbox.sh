#!/usr/bin/env bash
# Sandbox integration test for global config & context migration.
#
# Creates an isolated config and vault directory structure, initializes raidar,
# and verifies all paths and functions run correctly under sandboxed environment.
#
# Usage:
#   bash infra/test_sandbox.sh

set -euo pipefail

TOOL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${TOOL_DIR}"

SANDBOX_DIR="${TOOL_DIR}/sandbox"
export RAIDAR_CONFIG="${SANDBOX_DIR}/config/config.yaml"

echo "=== Raidar Sandbox Integration Test ==="
echo "Tool directory: ${TOOL_DIR}"
echo "Sandbox config: ${RAIDAR_CONFIG}"
echo "Sandbox vault : ${SANDBOX_DIR}/vault"
echo "---------------------------------------"

# 1. Clean previous sandbox run
rm -rf "${SANDBOX_DIR}"
mkdir -p "${SANDBOX_DIR}/config"

# 2. Run raidar init with sandbox paths
echo "[1] Initializing sandboxed vault and config..."
uv run raidar init --vault "${SANDBOX_DIR}/vault"

# 3. Assertions
echo -e "\n[2] Running assertions..."

assert_exists() {
    local path="$1"
    local desc="$2"
    if [[ -e "${path}" ]]; then
        echo "  ✓ ${desc} exists"
    else
        echo "  ✗ FAIL: ${desc} NOT found at ${path}" >&2
        exit 1
    fi
}

assert_contains() {
    local file="$1"
    local pattern="$2"
    local desc="$3"
    if grep -q "${pattern}" "${file}"; then
        echo "  ✓ ${desc} contains expected content"
    else
        echo "  ✗ FAIL: ${desc} ('${pattern}') NOT found in ${file}" >&2
        exit 1
    fi
}

# Verify configuration exists
assert_exists "${RAIDAR_CONFIG}" "sandbox config.yaml"
assert_contains "${RAIDAR_CONFIG}" "path: ${SANDBOX_DIR}/vault" "config.yaml vault.path"

# Verify vault structure and contents
assert_exists "${SANDBOX_DIR}/vault/context.md" "vault/context.md"
assert_exists "${SANDBOX_DIR}/vault/README.md" "vault/README.md"
assert_exists "${SANDBOX_DIR}/vault/.gitignore" "vault/.gitignore"

assert_contains "${SANDBOX_DIR}/vault/.gitignore" "embeddings/" "vault/.gitignore embeddings ignore"
assert_contains "${SANDBOX_DIR}/vault/.gitignore" "logs/" "vault/.gitignore logs ignore"

# Verify vault subdirectories
for subdir in concepts artifacts signals digests embeddings logs; do
    assert_exists "${SANDBOX_DIR}/vault/${subdir}" "vault/${subdir} directory"
done

# 4. Verify path resolution in config loader
echo -e "\n[3] Verifying config loader resolution..."
RESOLVED_VAULT="$(uv run python -c "from lib import config; print(config.load().vault_path)")"
RESOLVED_CONTEXT="$(uv run python -c "from lib import config; print(config.load().context_path)")"
RESOLVED_LOG="$(uv run python -c "from lib import config; print(config.load().log_file)")"

echo "  Resolved vault  : ${RESOLVED_VAULT}"
echo "  Resolved context: ${RESOLVED_CONTEXT}"
echo "  Resolved log    : ${RESOLVED_LOG}"

if [[ "${RESOLVED_VAULT}" != "${SANDBOX_DIR}/vault" ]]; then
    echo "  ✗ FAIL: resolved vault path mismatch" >&2
    exit 1
fi

if [[ "${RESOLVED_CONTEXT}" != "${SANDBOX_DIR}/vault/context.md" ]]; then
    echo "  ✗ FAIL: resolved context path mismatch" >&2
    exit 1
fi

if [[ "${RESOLVED_LOG}" != "${SANDBOX_DIR}/vault/logs/raidar.log" ]]; then
    echo "  ✗ FAIL: resolved log file path mismatch" >&2
    exit 1
fi
echo "  ✓ Config loader resolves relative paths against the vault successfully"

# 5. Run a dry run capture using the sandbox
echo -e "\n[4] Running dry-run capture in sandbox environment..."
# Let's seed a mock .env to config.parent so secrets loads it
echo "GITHUB_PAT=mock_pat_value" > "${SANDBOX_DIR}/config/.env"
uv run raidar capture --dry-run "https://github.com/astral-sh/uv"

echo -e "\n=== SANDBOX INTEGRATION TEST PASSED SUCCESSFULLY ==="

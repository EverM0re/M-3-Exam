#!/usr/bin/env bash
#
# setup_upstream.sh - Clone the upstream repositories required by each baseline.
#
# Usage:
#   bash scripts/setup_upstream.sh                  # install ALL upstream repos
#   bash scripts/setup_upstream.sh amem memverse    # only the listed baselines
#   bash scripts/setup_upstream.sh --list           # list available baselines
#   bash scripts/setup_upstream.sh --force amem     # re-clone (delete existing first)
#
# Notes:
#   - Always tracks each upstream's default branch (main / master). Versions
#     are NOT pinned; behaviour follows whatever the upstream ships today.
#   - This script ONLY clones the upstream source code. It does NOT install
#     Python packages: see baselines/README.md for the pip steps.
#   - mem0 and nano_graphrag are pip-installable PyPI packages and require
#     NO clone (handled via `pip install mem0ai` / `pip install nano-graphrag`).
#
set -euo pipefail

# Resolve the baselines/ directory regardless of where the script is invoked.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ------------------------------------------------------------------------------
# Baseline registry: name|github-url|post-clone-layout-fixup
#
# The "layout-fixup" column is a shell snippet executed inside the cloned
# directory (after `git clone`). Most baselines need no fixup ("noop"); a few
# need their inner sub-directory flattened to match what the adapter imports.
# ------------------------------------------------------------------------------
BASELINES=(
    "amem|https://github.com/WujiangXu/A-mem|noop"
    "memoryos|https://github.com/BAI-LAB/MemoryOS|memoryos_flatten"
    "mirix|https://github.com/Mirix-AI/MIRIX|noop"
    "memverse|https://github.com/KnowledgeXLab/MemVerse|noop"
    "ngm|https://github.com/StuckInTheNet/Neural-Graph-Memory-NGM|noop"
    "raganything|https://github.com/HKUDS/RAG-Anything|noop"
    "universalrag|https://github.com/wgcyeo/UniversalRAG|noop"
)

PIP_ONLY_BASELINES=("mem0" "nano_graphrag")

# Pretty-print helpers -------------------------------------------------------
COLOR_BLUE="\033[34m"
COLOR_GREEN="\033[32m"
COLOR_YELLOW="\033[33m"
COLOR_RED="\033[31m"
COLOR_RESET="\033[0m"
log_info()  { printf "${COLOR_BLUE}[info]${COLOR_RESET}  %s\n" "$*"; }
log_ok()    { printf "${COLOR_GREEN}[ok]${COLOR_RESET}    %s\n" "$*"; }
log_warn()  { printf "${COLOR_YELLOW}[warn]${COLOR_RESET}  %s\n" "$*"; }
log_error() { printf "${COLOR_RED}[error]${COLOR_RESET} %s\n" "$*" >&2; }

# Layout fixups --------------------------------------------------------------
# Each fixup runs inside the cloned repository (working directory = the clone).

fixup_noop() {
    :  # nothing to do
}

# BAI-LAB/MemoryOS hosts its Python package under memoryos-pypi/. Our adapter
# imports `from memoryos import Memoryos`, which is the *module* memoryos.py
# living at that subdirectory's root. Flatten so that becomes top-level here.
fixup_memoryos_flatten() {
    if [ -d memoryos-pypi ]; then
        # Move every file out of memoryos-pypi/ into the current directory.
        shopt -s dotglob
        mv memoryos-pypi/* ./
        shopt -u dotglob
        rmdir memoryos-pypi
    fi
}

# Public API: clone <name> <url> <fixup>
clone_one() {
    local name="$1" url="$2" fixup="$3"
    local target="${BASELINES_DIR}/${name}/upstream"

    if [ -d "$target" ] && [ -z "${FORCE:-}" ]; then
        log_warn "${name}: ${target} already exists. Use --force to re-clone."
        return 0
    fi
    if [ -d "$target" ] && [ -n "${FORCE:-}" ]; then
        log_info "${name}: --force given; removing existing ${target}"
        rm -rf "$target"
    fi

    log_info "${name}: cloning ${url} (depth=1) ..."
    git clone --depth=1 "$url" "$target"

    log_info "${name}: applying layout fixup [${fixup}]"
    ( cd "$target" && "fixup_${fixup}" )

    log_ok "${name}: ready at ${target}"
}

# CLI parsing ----------------------------------------------------------------
FORCE=""
SELECTED=()
LIST_ONLY=""

while [ $# -gt 0 ]; do
    case "$1" in
        --list)   LIST_ONLY=1 ;;
        --force)  FORCE=1 ;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -20
            exit 0
            ;;
        --*)
            log_error "unknown flag: $1"
            exit 2
            ;;
        *)
            SELECTED+=("$1")
            ;;
    esac
    shift
done

if [ -n "$LIST_ONLY" ]; then
    echo "Baselines that require git clone:"
    for entry in "${BASELINES[@]}"; do
        IFS='|' read -r name url _ <<< "$entry"
        printf "  %-15s  %s\n" "$name" "$url"
    done
    echo
    echo "Baselines available via pip (no clone needed):"
    for n in "${PIP_ONLY_BASELINES[@]}"; do
        printf "  %s\n" "$n"
    done
    exit 0
fi

# Build the list of baselines to install.
TARGETS=()
if [ "${#SELECTED[@]}" -eq 0 ]; then
    # No args -> install everything that needs cloning.
    for entry in "${BASELINES[@]}"; do
        IFS='|' read -r name _ _ <<< "$entry"
        TARGETS+=("$name")
    done
else
    for name in "${SELECTED[@]}"; do
        found=""
        for entry in "${BASELINES[@]}"; do
            IFS='|' read -r ename _ _ <<< "$entry"
            if [ "$ename" = "$name" ]; then
                found=1
                TARGETS+=("$name")
                break
            fi
        done
        if [ -z "$found" ]; then
            # Friendly hint for pip-only baselines.
            for p in "${PIP_ONLY_BASELINES[@]}"; do
                if [ "$p" = "$name" ]; then
                    log_warn "${name}: pip-installable baseline; no clone required."
                    log_warn "        Run 'pip install mem0ai' or 'pip install nano-graphrag>=0.0.6' (see README)."
                    found=1
                    break
                fi
            done
            if [ -z "$found" ]; then
                log_error "unknown baseline: $name (use --list to see available)"
                exit 2
            fi
        fi
    done
fi

if [ "${#TARGETS[@]}" -eq 0 ]; then
    log_info "Nothing to clone."
    exit 0
fi

# Run the clones.
log_info "Installing ${#TARGETS[@]} upstream(s): ${TARGETS[*]}"
log_info "Tracking each repo's default branch (no version pinning)."
echo

failed=()
for name in "${TARGETS[@]}"; do
    for entry in "${BASELINES[@]}"; do
        IFS='|' read -r ename url fixup <<< "$entry"
        if [ "$ename" = "$name" ]; then
            if ! clone_one "$name" "$url" "$fixup"; then
                failed+=("$name")
            fi
            break
        fi
    done
done

echo
if [ "${#failed[@]}" -gt 0 ]; then
    log_error "Some baselines failed to install: ${failed[*]}"
    exit 1
fi
log_ok "All requested upstream sources are ready."
log_info "Remember to install the required pip packages from baselines/README.md."

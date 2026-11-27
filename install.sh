#!/usr/bin/env bash
# Install linker-sim into ./.venv from uv.lock, then check the installation.
#
# Needs: macOS on Apple Silicon, or Linux on x86_64 (glibc >= 2.28) or ARM64
# (glibc >= 2.34); internet access; about 2 GB of disk. No GPU. Everything else (uv,
# Python 3.12, MuJoCo, the viewers) is installed by this script.
# Run it from anywhere: ./install.sh
set -euo pipefail

UV_VERSION="0.11.7"   # installed only if uv is missing
MIN_UV="0.11.0"

fail() { echo "install.sh: $*" >&2; exit 1; }
older() { [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" = "$1" ]; }

cd "$(dirname "$0")"

system="$(uname -s)"
machine="$(uname -m)"
case "${system}/${machine}" in
    Darwin/arm64) ;;
    Darwin/*) fail "linker-sim needs a Mac with Apple Silicon; MuJoCo 4 has no Intel Mac build." ;;
    Linux/x86_64) min_glibc=2.28 ;;
    Linux/aarch64) min_glibc=2.34 ;;
    *) fail "linker-sim supports macOS on Apple Silicon and Linux on x86_64 or ARM64, not ${system} ${machine}." ;;
esac
if [ "$system" = Linux ]; then
    glibc="$(ldd --version 2>/dev/null | head -n1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
    [ -n "$glibc" ] || fail "could not read the glibc version (linker-sim needs glibc >= ${min_glibc})."
    older "$glibc" "$min_glibc" && fail "glibc ${glibc} is too old; need >= ${min_glibc}."
fi

# uv fetches LW-v0's files with git. On a Mac without the command line tools, /usr/bin/git
# exists but fails (and offers to install them), so run it rather than look it up.
git --version >/dev/null 2>&1 || fail "git not found. Install it first (macOS: 'xcode-select --install'; Ubuntu: 'sudo apt install git'); uv fetches LW-v0's robot files with it."

if ! command -v uv >/dev/null; then
    echo "Installing uv ${UV_VERSION} (https://docs.astral.sh/uv/) into ~/.local/bin"
    url="https://astral.sh/uv/${UV_VERSION}/install.sh"
    if command -v curl >/dev/null; then curl -LsSf "$url" | sh
    elif command -v wget >/dev/null; then wget -qO- "$url" | sh
    else fail "need curl or wget to download uv."
    fi
    export PATH="$HOME/.local/bin:$PATH"
    installed_uv=1
fi
uv_version="$(uv --version | cut -d' ' -f2)"
older "$uv_version" "$MIN_UV" && fail "uv ${uv_version} is too old; run 'uv self update' (need >= ${MIN_UV})."

echo "Creating .venv from uv.lock (first run downloads up to 250 MB of packages and about 30 MB of LW-v0's files)"
uv sync --locked

status=0
uv run linker-sim doctor || status=$?

echo
echo "linker-sim is installed in $(pwd)/.venv."
echo "Run commands with 'uv run linker-sim <command>', or 'source .venv/bin/activate' first."
if [ -n "${installed_uv:-}" ]; then
    echo "uv was installed into ~/.local/bin: open a new terminal, or run 'source ~/.local/bin/env', so this one finds it."
fi
exit "$status"

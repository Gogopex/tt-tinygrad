#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PIN="$(cat "$REPO/TINYGRAD_PIN")"
TG="$REPO/tinygrad"
PATCH="$REPO/patches/0001-tt-device-hooks.patch"

if [ ! -d "$TG/.git" ]; then
  echo "cloning tinygrad@$PIN into $TG"
  git clone https://github.com/tinygrad/tinygrad.git "$TG"
fi

cd "$TG"
if [ "$(git rev-parse HEAD)" != "$PIN" ]; then
  echo "checking out pinned commit $PIN"
  git fetch --quiet
  git checkout --quiet "$PIN"
fi

if git apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "patch already applied"
else
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "tinygrad working tree is dirty and patch is not cleanly applied; aborting"
    exit 1
  fi
  echo "applying patch"
  git apply --check "$PATCH"
  git apply "$PATCH"
fi

echo "linking TT backend files"
ln -sf "$REPO/tt_renderer.py" "$TG/tinygrad/renderer/tt.py"
ln -sf "$REPO/tt_device.py"   "$TG/tinygrad/runtime/ops_tt.py"

echo "done. tinygrad is at $TG with the TT backend wired in."
echo "run:   python tt_runner.py"

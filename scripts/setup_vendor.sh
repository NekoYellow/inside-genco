#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/vendor/gridfm-graphkit"
REVISION="95b90eecc7f5c2c86a9dd776f0c3f40434def22f"

if [[ ! -d "$DEST/.git" ]]; then
  git clone https://github.com/gridfm/gridfm-graphkit "$DEST"
fi

git -C "$DEST" fetch origin "$REVISION"
git -C "$DEST" checkout --detach "$REVISION"
echo "gridfm-graphkit ready at $REVISION"

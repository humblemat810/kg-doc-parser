#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_kogwistar="$repo_root/kogwistar"

cd "$repo_root"

poetry install

if [ -f "$local_kogwistar/pyproject.toml" ]; then
  echo "Local kogwistar checkout detected; installing editable override from $local_kogwistar"
  poetry run python -m pip install -e "$local_kogwistar"
else
  echo "No local kogwistar checkout found; keeping GitHub dependency from pyproject.toml"
fi

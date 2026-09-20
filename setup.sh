#!/usr/bin/env bash
# Run from the project directory, or pass this script's absolute path.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$PROJECT_DIR/scripts/setup_server.sh" "$@"

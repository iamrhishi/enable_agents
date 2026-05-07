#!/usr/bin/env bash
# Convenience wrapper — real script lives in scripts/run.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/scripts/run.sh" "$@"

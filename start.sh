#!/usr/bin/env bash
# Build the frontend (if needed) and run the backend, which serves both the
# API and the web app at http://localhost:${PORT:-8000}.
# Extra arguments are passed to uvicorn, e.g. ./start.sh --reload
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
FRONTEND="$ROOT/frontend"
BACKEND="$ROOT/backend"

if [ ! -x "$BACKEND/.venv/bin/uvicorn" ]; then
  echo "Backend virtualenv not found. Set it up first:" >&2
  echo "  cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

if [ ! -d "$FRONTEND/node_modules" ]; then
  echo "Installing frontend dependencies..."
  (cd "$FRONTEND" && npm install)
fi

# Rebuild only when dist is missing or any frontend source is newer than it.
if [ ! -f "$FRONTEND/dist/index.html" ] || \
   [ -n "$(find "$FRONTEND/src" "$FRONTEND/public" "$FRONTEND/index.html" \
             "$FRONTEND/vite.config.ts" "$FRONTEND/package.json" \
             -newer "$FRONTEND/dist/index.html" -print -quit 2>/dev/null)" ]; then
  echo "Building frontend..."
  (cd "$FRONTEND" && npm run build)
fi

cd "$BACKEND"
echo "Starting app at http://localhost:${PORT:-8000}"
exec .venv/bin/uvicorn app.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" "$@"

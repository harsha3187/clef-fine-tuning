#!/usr/bin/env bash
# Runs once after the dev container is created: Git LFS, and a shell hook that reads the repository's .env.
set -eu

git lfs install --skip-repo >/dev/null 2>&1 || true

# .env (gitignored) holds the host token broker's address and secret; the repository is mounted, so the container can read it.
# tr drops the carriage returns a .env written on Windows can carry; a stray one ends up inside the secret.
hook='[ -f "$PWD/.env" ] && [ -f "$PWD/agent_demo/server.py" ] && { set -a; . <(tr -d "\r" < "$PWD/.env"); set +a; }'
grep -qF "$hook" ~/.bashrc 2>/dev/null || printf '\n# Adaptive Intelligence Engine\n%s\n' "$hook" >> ~/.bashrc

echo "Dev container ready."
echo "  tests:  python -m pytest -q"
echo "  app:    python -m agent_demo.server        (http://localhost:8765)"
if [ ! -f .env ]; then
  echo "  Azure:  no .env yet. On the host run: python -m routing_lab.token_broker --write-env .env, then reopen a terminal here."
fi
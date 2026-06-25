#!/usr/bin/env bash
# rk-agent start script
set -euo pipefail

C="\033[0;36m" G="\033[0;32m" R="\033[0;31m" Y="\033[0;33m" DIM="\033[2m" NC="\033[0m"
ok()   { echo -e "  \033[0;32m✓\033[0m $*"; }
err()  { echo -e "  \033[0;31m✗\033[0m $*"; }
info() { echo -e "  \033[2m→\033[0m $*"; }

echo ""
echo -e "  ${C}rk-agent${NC}"
echo -e "  ${DIM}─────────────────────────${NC}"

# Check .env
if [ ! -f ".env" ]; then
    err ".env not found. Run: bash scripts/setup.sh"
    exit 1
fi

# Source .env for validation
set -a; source .env; set +a

if [ -z "${TELEGRAM_BOT_TOKEN:-}" ] && [ -z "${API_HOST:-}" ]; then
    err "No platform configured. Set TELEGRAM_BOT_TOKEN or API_HOST in .env"
    exit 1
fi
if [ -z "${TELEGRAM_BOT_TOKEN:-}" ]; then
    info "Telegram interface disabled (no TELEGRAM_BOT_TOKEN)"
fi
if [ -z "${BOT_ENCRYPTION_KEY:-}" ]; then
    err "BOT_ENCRYPTION_KEY is not set in .env"
    exit 1
fi

# Activate venv
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    ok "Virtual environment active"
else
    err "No .venv found. Run: bash scripts/setup.sh"
    exit 1
fi

# Ensure workspace exists
mkdir -p "${RIKA_WORKSPACE:-$HOME/.Rika-Workspace}"
ok "Workspace: ${RIKA_WORKSPACE:-$HOME/.Rika-Workspace}"

# Migrations
info "Checking database migrations..."
python3 -m src.db.migrate
ok "Database up to date"

# Start
echo ""
echo -e "  ${G}Starting bot...${NC}  (Ctrl+C to stop)"
echo -e "  ${DIM}─────────────────────────${NC}"
echo ""
exec python3 -m src.bot.app

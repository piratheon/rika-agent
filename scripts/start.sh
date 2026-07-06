#!/usr/bin/env bash
# rika-agent start script
set -euo pipefail

C="\033[0;36m" G="\033[0;32m" R="\033[0;31m" Y="\033[0;33m" DIM="\033[2m" NC="\033[0m"
ok()   { echo -e "  \033[0;32m✓\033[0m $*"; }
err()  { echo -e "  \033[0;31m✗\033[0m $*" >&2; }
info() { echo -e "  \033[2m→\033[0m $*"; }
warn() { echo -e "  \033[0;33m!\033[0m $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo ""
echo -e "  ${C}rk-agent${NC}"
echo -e "  ${DIM}─────────────────────────${NC}"

# ── Duplicate-run guard ───────────────────────────────────────────────────────
PID_FILE="/tmp/rika-agent.pid"
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        err "Bot is already running (PID $OLD_PID)."
        err "Stop it first:  kill $OLD_PID   or   bash scripts/start.sh --force"
        [[ "${1:-}" == "--force" ]] || exit 1
        warn "--force: killing previous instance..."
        kill "$OLD_PID" 2>/dev/null && sleep 1
    fi
    rm -f "$PID_FILE"
fi

# ── .env ─────────────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    err ".env not found. Run:  bash scripts/setup.sh"
    exit 1
fi
set -a; source .env; set +a

if [ -z "${TELEGRAM_BOT_TOKEN:-}" ] && [ -z "${API_HOST:-}" ] && [ -z "${DISCORD_BOT_TOKEN:-}" ]; then
    err "No platform configured. Set TELEGRAM_BOT_TOKEN, DISCORD_BOT_TOKEN, or API_HOST in .env"
    exit 1
fi
if [ -z "${BOT_ENCRYPTION_KEY:-}" ]; then
    err "BOT_ENCRYPTION_KEY is not set in .env"
    exit 1
fi

# ── Virtual environment ───────────────────────────────────────────────────────
if [ -d ".venv" ]; then
    source .venv/bin/activate
    ok "Virtual environment active"
else
    err "No .venv found. Run:  bash scripts/setup.sh"
    exit 1
fi

# ── Directories ───────────────────────────────────────────────────────────────
RIKA_HOME="$HOME/.rika"
WORKSPACE="${WORKSPACE_PATH:-$RIKA_HOME/shared}"
mkdir -p "$WORKSPACE" "$RIKA_HOME/logs" "$RIKA_HOME/data/paused" "$RIKA_HOME/skills"
ok "Workspace: $WORKSPACE"

LOG_FILE="$RIKA_HOME/logs/rk-$(date '+%Y%m%d-%H%M%S').log"
info "Log: $LOG_FILE"

# ── Migrations ────────────────────────────────────────────────────────────────
info "Checking database migrations..."
python3 -m src.db.migrate
ok "Database up to date"

# ── Launch ────────────────────────────────────────────────────────────────────
echo ""
echo -e "  ${G}Starting bot...${NC}  (Ctrl+C to stop)"
echo -e "  ${DIM}─────────────────────────${NC}"
echo ""

cleanup() { rm -f "$PID_FILE"; }
trap cleanup EXIT

# Write PID after exec is not possible, so run in background then re-exec or
# use a wrapper that captures the PID before handing over
python3 -m src.bot.app &
BOT_PID=$!
echo "$BOT_PID" > "$PID_FILE"
ok "PID $BOT_PID"
wait "$BOT_PID"

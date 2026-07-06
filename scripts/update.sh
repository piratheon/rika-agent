#!/usr/bin/env bash
# rika-agent update script — pull, reinstall deps, migrate, restart if running
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
echo -e "  ${C}rika-agent update${NC}"
echo -e "  ${DIM}─────────────────────────${NC}"

# ── Git pull ──────────────────────────────────────────────────────────────────
if command -v git &>/dev/null && [ -d ".git" ]; then
    info "Pulling latest changes..."
    BEFORE=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    git pull --ff-only
    AFTER=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    if [ "$BEFORE" != "$AFTER" ]; then
        ok "Updated: ${BEFORE:0:7} → ${AFTER:0:7}"
    else
        ok "Already up to date (${BEFORE:0:7})"
    fi
else
    warn "Not a git repository — skipping pull. Copy new files manually if needed."
fi

# ── Virtual environment ───────────────────────────────────────────────────────
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    warn "No .venv found — creating one..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

info "Updating dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
ok "Dependencies up to date"

# ── Migrations ────────────────────────────────────────────────────────────────
info "Running database migrations..."
python3 -m src.db.migrate
ok "Database migrated"

# ── Skills bootstrap ──────────────────────────────────────────────────────────
info "Bootstrapping new bundled skills..."
python3 - << 'PYEOF'
from src.core.skill_registry import ensure_skills_bootstrapped
ensure_skills_bootstrapped()
print("  → skills ready")
PYEOF

# ── Restart running instance ──────────────────────────────────────────────────
PID_FILE="/tmp/rika-agent.pid"
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        warn "Bot is running (PID $OLD_PID) — restarting..."
        kill "$OLD_PID"
        sleep 2
        rm -f "$PID_FILE"
        bash "$SCRIPT_DIR/start.sh" &
        ok "Restarted in background"
    else
        rm -f "$PID_FILE"
        info "Previous instance not running — not restarting"
    fi
else
    info "Bot is not running — start with: bash scripts/start.sh"
fi

echo ""
echo -e "  ${G}Update complete.${NC}"

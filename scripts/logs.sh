#!/usr/bin/env bash
# rika-agent log viewer — tail latest log, or list/filter all logs
set -euo pipefail

C="\033[0;36m" G="\033[0;32m" DIM="\033[2m" NC="\033[0m"
err() { echo -e "  \033[0;31m✗\033[0m $*" >&2; }

LOGS_DIR="$HOME/.rika/logs"
ACTION="${1:-tail}"
LINES="${2:-80}"

if [ ! -d "$LOGS_DIR" ]; then
    err "No log directory at $LOGS_DIR — has the bot been started yet?"
    exit 1
fi

latest_log() {
    # Returns path to most recently modified log file
    ls -t "$LOGS_DIR"/rk-*.log 2>/dev/null | head -1
}

case "$ACTION" in
    tail|"")
        LOG=$(latest_log)
        if [ -z "$LOG" ]; then err "No log files found in $LOGS_DIR"; exit 1; fi
        echo -e "  ${C}Tailing:${NC} $LOG"
        echo -e "  ${DIM}──────────────────────────────────────${NC}"
        tail -f -n "$LINES" "$LOG"
        ;;
    list|ls)
        echo -e "  ${C}Log files in $LOGS_DIR${NC}"
        echo ""
        ls -lht "$LOGS_DIR"/rk-*.log 2>/dev/null | awk '{print "  " $6 " " $7 " " $8 "  " $5 "  " $9}' \
            || err "No log files found"
        ;;
    grep|search)
        PATTERN="${2:-ERROR}"
        LOG=$(latest_log)
        if [ -z "$LOG" ]; then err "No log files found"; exit 1; fi
        echo -e "  ${C}Searching '${PATTERN}' in:${NC} $LOG"
        echo ""
        grep --color=auto -n "$PATTERN" "$LOG" | tail -40 || echo "  No matches"
        ;;
    errors)
        LOG=$(latest_log)
        if [ -z "$LOG" ]; then err "No log files found"; exit 1; fi
        echo -e "  ${C}Errors and warnings in:${NC} $LOG"
        echo ""
        grep --color=auto -iE '"level"\s*:\s*"(error|warning|critical)"' "$LOG" | tail -40 \
            || grep --color=auto -iE '\b(ERROR|WARNING|CRITICAL)\b' "$LOG" | tail -40 \
            || echo "  No errors found"
        ;;
    clean)
        KEEP="${2:-10}"
        COUNT=$(ls "$LOGS_DIR"/rk-*.log 2>/dev/null | wc -l)
        if [ "$COUNT" -le "$KEEP" ]; then
            echo "  Only $COUNT log file(s) — nothing to clean (keeping $KEEP)"
            exit 0
        fi
        # Remove all but the newest KEEP logs
        ls -t "$LOGS_DIR"/rk-*.log | tail -n "+$((KEEP + 1))" | xargs rm -f
        REMOVED=$((COUNT - KEEP))
        echo -e "  ${G}Removed $REMOVED old log file(s), kept $KEEP${NC}"
        ;;
    *)
        echo "Usage: bash scripts/logs.sh [tail|list|errors|grep <pattern>|clean [keep_n]]"
        echo "  tail             Follow latest log (default)"
        echo "  list             List all log files"
        echo "  errors           Show only errors/warnings in latest log"
        echo "  grep <pattern>   Search pattern in latest log"
        echo "  clean [N]        Delete all but the N newest logs (default: 10)"
        exit 1
        ;;
esac

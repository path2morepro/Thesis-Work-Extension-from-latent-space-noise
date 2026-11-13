#!/usr/bin/env bash
# Catch the deaths run_all.sh cannot report itself.
#
# run_all.sh notifies on "finished" and on "interrupted", but both of those
# run INSIDE the pipeline process. An OOM-kill, a power loss or a hard reboot
# leaves status saying "state=running" forever and sends nothing, so the first
# you hear of it is when you check hours later.
#
# This script is the outside observer: if the status file says running but the
# recorded pid is gone, the run died. It pushes one notification and writes
# "state=crashed" so it never fires twice for the same run.
#
# Install (every 10 minutes):
#     ./tools/watchdog.sh --install
# Remove:
#     ./tools/watchdog.sh --uninstall
# Check by hand:
#     ./tools/watchdog.sh            # silent unless something is wrong
#     ./tools/watchdog.sh --verbose
#
# Safe to run when no pipeline has ever started, when one finished normally,
# and concurrently with itself.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATEST="$ROOT/logs/latest"
NOTIFY_PY="$ROOT/tools/notify.py"
NOTIFY_ENV="${NOTIFY_ENV:-$HOME/.config/thesis-notify.env}"
CRON_TAG='thesis-run_all-watchdog'

VERBOSE=0
say() { [[ $VERBOSE -eq 1 ]] && echo "$*"; return 0; }

resolve_py() {
    local c
    for c in "$ROOT/.venv/bin/python" "$(command -v python3 2>/dev/null || true)"; do
        [[ -n "$c" && -x "$c" ]] && { echo "$c"; return 0; }
    done
    return 1
}

cmd_install() {
    local py; py="$(resolve_py)" || { echo 'no python found' >&2; exit 1; }
    # Keep every other crontab line, drop any previous copy of ours.
    local cur; cur="$(crontab -l 2>/dev/null | grep -v "$CRON_TAG" || true)"
    printf '%s\n*/10 * * * * %s/tools/watchdog.sh >/dev/null 2>&1  # %s\n' \
        "$cur" "$ROOT" "$CRON_TAG" | sed '/^$/d' | crontab -
    echo "installed: every 10 min  ($CRON_TAG)"
    crontab -l | grep "$CRON_TAG"
}

cmd_uninstall() {
    crontab -l 2>/dev/null | grep -v "$CRON_TAG" | crontab -
    echo "removed ($CRON_TAG)"
}

case "${1:-}" in
    --install)   cmd_install;   exit 0 ;;
    --uninstall) cmd_uninstall; exit 0 ;;
    --verbose)   VERBOSE=1 ;;
    '')          ;;
    *)           echo "usage: $0 [--install|--uninstall|--verbose]" >&2; exit 2 ;;
esac

d="$(readlink -f "$LATEST" 2>/dev/null || true)"
[[ -n "$d" && -d "$d" ]] || { say 'no run directory'; exit 0; }

state=$(sed -n 's/^state=//p' "$d/status" 2>/dev/null || true)
stage=$(sed -n 's/^stage=//p' "$d/status" 2>/dev/null || true)
name=$(sed  -n 's/^name=//p'  "$d/status" 2>/dev/null || true)
started=$(sed -n 's/^started=//p' "$d/status" 2>/dev/null || echo 0)
pid=$(cat "$d/pid" 2>/dev/null || true)

say "state=$state pid=$pid stage=$stage/$name"

# Only "running" can be a silent death. finished / interrupted / failed /
# crashed have all been reported already.
[[ "$state" == running ]] || { say 'nothing to check'; exit 0; }
[[ -n "$pid" ]]           || { say 'no pid recorded';  exit 0; }

if kill -0 "$pid" 2>/dev/null; then
    say "pid $pid alive"
    exit 0
fi

# Mark it first. If the notification is slow or fails, we still never fire
# twice for the same run.
{ echo 'state=crashed'; echo "stage=$stage"; echo "name=$name"
  echo "started=$started"; } > "$d/status"

elapsed=0
[[ "$started" =~ ^[0-9]+$ ]] && elapsed=$(( $(date +%s) - started ))
el=$(printf '%dh %02dm' $((elapsed/3600)) $(((elapsed%3600)/60)))

echo "[$(date '+%F %T')] WATCHDOG: pid $pid gone while state=running" \
    >> "$d/pipeline.log"

if [[ -f "$NOTIFY_PY" ]]; then
    py="$(resolve_py)" || exit 0
    (
        set +e
        [[ -f "$NOTIFY_ENV" ]] && { set -a; . "$NOTIFY_ENV"; set +a; }
        "$py" "$NOTIFY_PY" --state failed --run "$(basename "$d")" \
            --stage "$stage ${name:-?}" --elapsed "$el" \
            --log "$d/pipeline.log" \
            --note "Process died without reporting (OOM-kill, power loss or hard reboot). Nothing was saved past the last completed window. Resume with: ./run_all.sh start"
    ) || true
fi
exit 0

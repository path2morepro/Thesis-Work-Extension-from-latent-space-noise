#!/usr/bin/env bash
#
# forecasting.sh — direct forecasting, four latent arms, detached.
#
#   ./forecasting.sh start          launch (returns immediately)
#   ./forecasting.sh status         which arm, how far, ETA
#   ./forecasting.sh follow         tail the live log (Ctrl-C is safe)
#   ./forecasting.sh stop           kill the run
#   ./forecasting.sh notify-test    send a test notification
#
# Leads 1..24 from one anchor per window, no ARCI, ens=10, Euler 100 steps,
# 50 windows per arm, test split. Every arm differs ONLY in the latent chain
#     z_k = a_k z_{k-1} + sqrt(1 - a_k^2) eps_k
# and they run in this order:
#
# STEP ONE — constant a, the two extremes:
#   uncorrelated_noise   a = 0   z_k = eps_k      a fresh latent each lead
#   fixed_noise          a = 1   z_k = z_{k-1}    one latent, every lead
# At a=1 the sqrt(1-a^2) factor is exactly 0, so the innovations are still
# drawn and then discarded -- which is what keeps the arms paired.
#
# STEP TWO — a_k = exp(-c2_k / 2) from a fitted time-varying OU model
# (Analysis/fit.py, c2 only, fitted on the Euler latents of data/invert/from_val):
#   ou_no_x0   time_varyingOU_withoutx0.pt   c2(lead, ch, y)       val nll 0.0870
#   ou_x0      c2_euler.pt                   c2(lead, ch, y, x0)   val nll 0.0115
# Box-mean a_k is the same for both (0.79 -> 0.74); what x0 buys is the spatial
# pattern (per-pixel a spans 0.35..0.92 against 0.767..0.774). These two write
# metrics.npz and meta.json only -- no plots.
#
# All four arms run on the SAME windows (choose_anchors is keyed on
# (seed, traj_id), never on the chain) with the SAME innovation stream (the
# torch seed is derived per window), so lead 1 is bit-identical across arms and
# everything downstream differs by the latent correlation and nothing else.
# Nothing is chained in the physical state: each lead is its own map
# Phi(x0, tau) from the same anchor.
#
# An arm whose windows are all on disk is SKIPPED, so starting again to add an
# arm does not rewrite a finished arm's metrics or meta.
#
# Detached via setsid: closing the terminal, logging out or dropping an SSH
# connection does not signal it. Only ./forecasting.sh stop, a reboot, or the
# OOM killer ends it.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MAIN_ROOT="$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
MAIN_ROOT="${MAIN_ROOT%/.git}"

resolve_py() {
    local c
    for c in "$ROOT/.venv/bin/python" "$MAIN_ROOT/.venv/bin/python" \
             "$(command -v python3 2>/dev/null || true)"; do
        if [[ -n "$c" && -x "$c" ]]; then echo "$c"; return 0; fi
    done
    return 1
}
PY="${PY:-$(resolve_py || true)}"
if [[ -z "$PY" || ! -x "$PY" ]]; then
    echo "forecasting.sh: no usable python found" >&2; exit 1
fi

# ── CONFIG ────────────────────────────────────────────────────────────────
# Step one. Each arm is "dirname:alpha", under data/forecasting/.
DF_ARMS=${DF_ARMS:-"uncorrelated_noise:0 fixed_noise:1"}
# Step two. Each arm is an --arm key of Condition_model.ou_forecast, which names
# both the fitted model and the output folder. Empty runs step one only.
DF_OU_ARMS=${DF_OU_ARMS:-"ou_no_x0 ou_x0"}
# Arms to recompute from scratch even though their windows are already on disk,
# as a space-separated list of names. The arm's windows and metric files are
# DELETED and regenerated -- same windows, same seeds, same innovations, since
# none of those depend on the model -- which is how you swap a model and
# replace that arm's output. Its old meta.json and curve are copied into the
# run's log directory first. Everything else still resumes.
DF_FORCE=${DF_FORCE:-}
DF_SPLIT=${DF_SPLIT:-test}    # the chains were fitted on val, so test is held out
DF_TRAJ=${DF_TRAJ:-50}        # windows per arm = DF_TRAJ * DF_ANCHORS
DF_ANCHORS=${DF_ANCHORS:-1}   # 1 = every window from a different trajectory
DF_ENS=${DF_ENS:-10}
DF_LEADS=${DF_LEADS:-24}
DF_STEPS=${DF_STEPS:-100}     # Euler
DF_SEED=${DF_SEED:-0}
# A window is ENS*LEADS rows at ~0.0144 GiB/row: 10*24 = 240 rows = 4.2 GiB in
# one call, which fits a 6 GB card with nothing spare. 5 members per call is
# 120 rows = 2.2 GiB. Members are independent given the latents, and the full
# ensemble is drawn BEFORE slicing, so this changes nothing numerically.
DF_MICRO_ENS=${DF_MICRO_ENS:-5}
# Measured throughput on this card: 185 row-evals/s (Euler-100 at 1.85 rows/s).
DF_RATE=${DF_RATE:-185}

# One internal list for every stage, as "name|kind|param". The two CONFIG
# variables keep their own natural spelling; everything below reads this.
ALL_ARMS=()
for _a in $DF_ARMS;    do ALL_ARMS+=("${_a%%:*}|const|${_a##*:}"); done
for _a in $DF_OU_ARMS; do ALL_ARMS+=("$_a|ou|$_a"); done
TOTAL=${#ALL_ARMS[@]}
# ──────────────────────────────────────────────────────────────────────────

LOCK="$ROOT/.forecasting.lock"
NOTIFY_PY="$ROOT/tools/notify.py"
NOTIFY_ENV="${NOTIFY_ENV:-$HOME/.config/thesis-notify.env}"
LATEST="$ROOT/logs/forecasting-latest"

# NOT `find ... | wc -l`: find exits 1 on a missing directory, pipefail
# propagates that through wc, and set -e then kills the run on what should be a
# count of zero -- which is every arm that has not started yet.
count_files() {   # $1 dir, $2 name pattern
    [[ -d "$1" ]] || { echo 0; return 0; }
    find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l
}

fmt() { local s=$1; printf '%dh %02dm %02ds' $((s/3600)) $(((s%3600)/60)) $((s%60)); }

arm_name()  { echo "${1%%|*}"; }
arm_kind()  { local r="${1#*|}"; echo "${r%%|*}"; }
arm_param() { echo "${1##*|}"; }
arm_dir()   { echo "data/forecasting/${1%%|*}"; }
arm_label() {     # what distinguishes this arm, for the log and for status
    case "$(arm_kind "$1")" in
        const) echo "alpha=$(arm_param "$1")" ;;
        ou)    echo "model=$(arm_param "$1")" ;;
    esac
}

# seconds for `left` windows at this arm's cost
arm_secs() { awk -v w="$1" -v e="$DF_ENS" -v l="$DF_LEADS" -v n="$DF_STEPS" \
                 -v q="$DF_RATE" 'BEGIN{printf "%d", w*e*l*n/q}'; }

arm_forced() { case " $DF_FORCE " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# ── inspection commands (run in the user's shell, not the job) ───────────
cmd_status() {
    local d; d="$(readlink -f "$LATEST" 2>/dev/null || true)"
    [[ -n "$d" && -d "$d" ]] || { echo 'no run found'; exit 1; }

    local state stage name started pid alive='no'
    state=$(sed -n 's/^state=//p'     "$d/status" 2>/dev/null || echo unknown)
    stage=$(sed -n 's/^stage=//p'     "$d/status" 2>/dev/null || echo ?)
    name=$(sed  -n 's/^name=//p'      "$d/status" 2>/dev/null || echo ?)
    started=$(sed -n 's/^started=//p' "$d/status" 2>/dev/null || echo 0)
    pid=$(cat "$d/pid" 2>/dev/null || echo '')
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && alive='yes'

    echo "run        : $(basename "$d")"
    echo "state      : $state   (process alive: $alive, pid ${pid:-none})"
    echo "stage      : $stage/$TOTAL  $name"
    [[ "$started" != 0 ]] && echo "elapsed    : $(fmt $(( $(date +%s) - started )))"
    [[ -f "$d/completed" ]] && echo "completed  : $(paste -sd', ' "$d/completed")"

    # One file pair per window, so counting them IS the progress bar, and it
    # survives a resume because finished windows stay on disk.
    local want; want=$(( DF_TRAJ * DF_ANCHORS ))
    printf 'arms       :\n'
    for _arm in "${ALL_ARMS[@]}"; do
        local dir n pct
        dir="$ROOT/$(arm_dir "$_arm")"
        n=$(count_files "$dir" '*_predicts.npy')
        pct=$(( want > 0 ? 100 * n / want : 0 ))
        printf '  %-22s %3d/%-3d windows (%d%%)  %s\n' \
               "$(arm_name "$_arm")" "$n" "$want" "$pct" "$(arm_label "$_arm")"
    done

    # ETA from this run's own rate, over the arm currently being written.
    if [[ "$state" == running && "$started" != 0 ]]; then
        local dir n el
        dir="$ROOT/data/forecasting/$name"
        for _arm in "${ALL_ARMS[@]}"; do
            [[ "$(arm_name "$_arm")" == "$name" ]] && dir="$ROOT/$(arm_dir "$_arm")"
        done
        n=$(count_files "$dir" '*_predicts.npy')
        el=$(( $(date +%s) - started ))
        if [[ $n -gt 0 && $want -gt $n ]]; then
            echo "             $(awk -v n="$n" -v e="$el" 'BEGIN{printf "%.0f", e/n}')s/window\
   ETA $(fmt $(awk -v w="$want" -v n="$n" -v e="$el" 'BEGIN{printf "%d", (w-n)*e/n}'))"
        fi
    fi

    command -v nvidia-smi >/dev/null && echo "gpu        : $(nvidia-smi \
        --query-gpu=memory.used,memory.total,utilization.gpu \
        --format=csv,noheader 2>/dev/null | head -1)"
    echo "logs       : $d"
    echo
    echo "--- last 12 lines of pipeline.log ---"
    tail -12 "$d/pipeline.log" 2>/dev/null || true
}

cmd_follow() {
    local d; d="$(readlink -f "$LATEST" 2>/dev/null || true)"
    [[ -n "$d" && -d "$d" ]] || { echo 'no run found'; exit 1; }
    echo "following $d/pipeline.log — Ctrl-C only stops watching, not the run"
    tail -f "$d/pipeline.log"
}

cmd_stop() {
    local d; d="$(readlink -f "$LATEST" 2>/dev/null || true)"
    local pid; pid=$(cat "$d/pid" 2>/dev/null || echo '')
    [[ -n "$pid" ]] || { echo 'no pid recorded'; exit 1; }
    kill -0 "$pid" 2>/dev/null || { echo "pid $pid is not running"; return 0; }
    kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    echo "sent TERM to process group $pid, waiting..."
    for _ in $(seq 30); do
        kill -0 "$pid" 2>/dev/null || { echo 'stopped cleanly'; return 0; }
        sleep 1
    done
    kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    echo 'killed'
}

# ── the job ──────────────────────────────────────────────────────────────
log() { echo "[$(date '+%F %T')] $*"; }

# Runs in a SUBSHELL so the SMTP app password is never inherited by the python
# process. Always returns 0: a notification problem must not change a run.
notify() {
    local state="$1" note="${2:-}"
    [[ -f "$NOTIFY_PY" ]] || return 0
    (
        set +e
        [[ -f "$NOTIFY_ENV" ]] && { set -a; . "$NOTIFY_ENV"; set +a; }
        "$PY" "$NOTIFY_PY" --state "$state" --run "${RUN_ID:-?}" \
            --stage "${CUR:-?}/$TOTAL ${CURNAME:-?}" \
            --elapsed "$(fmt ${SECONDS:-0})" \
            --log "${RUN_DIR:-}/pipeline.log" ${note:+--note "$note"}
    ) || true
    return 0
}

set_status() {
    { echo "state=$1"; echo "stage=$2"; echo "name=$3"
      echo "started=$RUN_STARTED"; } > "$RUN_DIR/status"
}

CHILD=''; CUR=0; CURNAME=init

on_term() {
    log 'received TERM/INT — shutting down'
    if [[ -n "$CHILD" ]] && kill -0 "$CHILD" 2>/dev/null; then
        kill -TERM "$CHILD" 2>/dev/null || true
        for _ in $(seq 20); do kill -0 "$CHILD" 2>/dev/null || break; sleep 0.5; done
        kill -0 "$CHILD" 2>/dev/null && kill -KILL "$CHILD" 2>/dev/null || true
    fi
    set_status interrupted "$CUR" "$CURNAME"
    notify interrupted 'Stopped by request. Finished windows are on disk; ./forecasting.sh start resumes.'
    log 'run TERMINATED'
    exit 143
}

run_stage() {
    local num="$1" name="$2" logname="$3"; shift 3
    CUR=$num; CURNAME=$name
    set_status running "$num" "$name"
    log "=== stage $num/$TOTAL  $name  START"
    log "    cmd: $*"
    local t0=$SECONDS rc=0
    "$@" >>"$RUN_DIR/$logname" 2>&1 &
    CHILD=$!
    set +e; wait "$CHILD"; rc=$?; set -e
    CHILD=''
    if [[ $rc -eq 0 ]]; then
        log "=== stage $num/$TOTAL  $name  DONE in $(fmt $((SECONDS-t0)))"
        echo "$name" >> "$RUN_DIR/completed"
    else
        set_status failed "$num" "$name"
        log "!!! stage $num/$TOTAL  $name  FAILED (exit $rc) after $(fmt $((SECONDS-t0)))"
        tail -25 "$RUN_DIR/$logname" | sed 's/^/    | /'
        notify failed "$(printf 'Arm %s exited %s.\n\n%s' "$name" "$rc" \
                        "$(tail -15 "$RUN_DIR/$logname")")"
        exit $rc
    fi
}

check_workspace() {
    local d="data/$DF_SPLIT"
    if [[ ! -d "$ROOT/$d" ]] || [[ -z "$(ls -A "$ROOT/$d" 2>/dev/null)" ]]; then
        log "FATAL: missing or empty: $d"
        log "       this checkout is $ROOT"
        [[ -n "$MAIN_ROOT" && "$MAIN_ROOT" != "$ROOT" ]] && \
            log "       the data is in $MAIN_ROOT — run it there"
        set_status failed 0 workspace
        notify failed "Wrong checkout: $ROOT has no $d."
        exit 1
    fi
}

# Every OU arm needs its checkpoint before anything starts: three hours into a
# run is a bad moment to find out a file is missing.
check_models() {
    local arms=''
    for _arm in "${ALL_ARMS[@]}"; do
        [[ "$(arm_kind "$_arm")" == ou ]] && arms+=" $(arm_param "$_arm")"
    done
    [[ -n "$arms" ]] || return 0
    # Ask ou_forecast itself where each checkpoint is. A second copy of those
    # paths here would go stale the moment an arm's model is swapped, and this
    # also proves the import works before a two-hour run starts.
    local out rc=0
    out=$("$PY" -c '
import sys
from Condition_model.ou_forecast import ARMS
bad = []
for a in sys.argv[1:]:
    if a not in ARMS:
        bad.append("unknown OU arm: " + a + " (have " + ", ".join(sorted(ARMS)) + ")")
    elif not ARMS[a]["ckpt"].exists():
        bad.append("missing checkpoint: " + str(ARMS[a]["ckpt"]))
    else:
        print("model              " + a + " -> " + ARMS[a]["ckpt"].name)
if bad:
    print("\n".join(bad))
    sys.exit(1)
' $arms 2>&1) || rc=$?
    while IFS= read -r _l; do [[ -n "$_l" ]] && log "$_l"; done <<< "$out"
    if [[ $rc -ne 0 ]]; then
        log "FATAL: OU arm check failed"
        set_status failed 0 models
        notify failed "OU arm check failed"
        exit 1
    fi
}

cmd_run() {
    exec 9>"$LOCK"
    flock -n 9 || { echo 'another forecasting.sh is already running'; exit 1; }
    echo $$ > "$RUN_DIR/pid"
    trap on_term TERM INT

    RUN_STARTED=$(date +%s)
    set_status starting 0 init
    log "forecasting start  run=$RUN_ID  pid=$$"
    log "config             split=$DF_SPLIT traj=$DF_TRAJ anchors=$DF_ANCHORS \
ens=$DF_ENS leads=1..$DF_LEADS steps=$DF_STEPS seed=$DF_SEED"
    log "                   no arci; each lead its own map from one anchor"

    local want total=0 left_total=0
    want=$(( DF_TRAJ * DF_ANCHORS ))
    for _arm in "${ALL_ARMS[@]}"; do
        local nm dir have left secs
        nm="$(arm_name "$_arm")"; dir="$(arm_dir "$_arm")"
        have=$(count_files "$ROOT/$dir" '*_predicts.npy')
        left=$(( want > have ? want - have : 0 ))
        arm_forced "$nm" && left=$want              # forced: recompute all of them
        secs=$(arm_secs "$left")
        total=$(( total + secs ))
        left_total=$(( left_total + left ))
        log "arm $nm   $(arm_label "$_arm")  $have/$want on disk, \
$left to do  ~$(fmt $secs)  -> $dir"
    done
    log "estimate           ~$(fmt $total) (a complete arm is skipped)"
    log "disk               $(df -h "$ROOT" | awk 'NR==2{print $4" free ("$5" used)"}')\
   needs ~$(awk -v w="$left_total" -v e="$DF_ENS" -v l="$DF_LEADS" \
            'BEGIN{printf "%d MB", w*(e+1)*l*2*64*64*4/1048576}')"
    log "python             $PY"
    ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
    check_workspace
    check_models

    local n=0
    for _arm in "${ALL_ARMS[@]}"; do
        n=$(( n + 1 ))
        local nm dir have
        nm="$(arm_name "$_arm")"; dir="$(arm_dir "$_arm")"
        have=$(count_files "$ROOT/$dir" '*_predicts.npy')
        # Every arm's metric of record is the temporal difference, so a
        # complete arm that lacks it gets a read-only pass over its saved
        # windows -- no GPU, no sampler, and metrics.npz is left alone.
        local tdonly=0
        if arm_forced "$nm"; then
            # Clearing beats --no-resume. Overwriting in place rewrites the
            # windows one at a time, so a crash halfway leaves an arm that is
            # part new and part old, while meta.json and the curve -- both
            # written at the END -- still describe the OLD model. The next run
            # would then see want/want windows and skip it. Clearing makes a
            # crash resumable and keeps the folder honest about what it holds.
            log "    $nm is FORCED: clearing $have windows and its metric files"
            mkdir -p "$RUN_DIR/prev-$nm"
            for _f in meta.json temporal_difference.npz metrics.npz; do
                [[ -e "$ROOT/$dir/$_f" ]] && cp -p "$ROOT/$dir/$_f" "$RUN_DIR/prev-$nm/" || true
            done
            rm -f "$ROOT/$dir"/*_predicts.npy "$ROOT/$dir"/*_truth.npy \
                  "$ROOT/$dir"/meta.json "$ROOT/$dir"/metrics.npz \
                  "$ROOT/$dir"/metrics.png "$ROOT/$dir"/temporal_difference.npz
            log "    old meta / curve kept in $(basename "$RUN_DIR")/prev-$nm"
            have=0
        elif [[ "$have" -ge "$want" ]]; then
            if [[ -e "$ROOT/$dir/temporal_difference.npz" ]]; then
                log "=== stage $n/$TOTAL  $nm  SKIP (complete: $have/$want on disk)"
                echo "$nm" >> "$RUN_DIR/completed"
                continue
            fi
            tdonly=1
            log "    $nm is complete; adding its temporal difference only (no gpu)"
        fi
        case "$(arm_kind "$_arm")" in
            const)
                # --tdiff always; --no-metrics only when metrics.npz is already
                # there, so a fresh arm still gets rmse / crps / ssr and its plot
                local extra=(--tdiff)
                [[ "$tdonly" -eq 1 ]] && extra+=(--no-metrics)
                run_stage "$n" "$nm" "$(printf '%02d' "$n")-$nm.log" \
                    "$PY" -m Condition_model.direct_forecast \
                        --split "$DF_SPLIT" --n-traj "$DF_TRAJ" --anchors "$DF_ANCHORS" \
                        --ens "$DF_ENS" --alpha "$(arm_param "$_arm")" \
                        --n-lead "$DF_LEADS" --steps "$DF_STEPS" \
                        --micro-ens "$DF_MICRO_ENS" --seed "$DF_SEED" \
                        --dirname "$nm" "${extra[@]}"
                ;;
            ou)
                # no --plot: these arms write metrics.npz and meta.json only
                run_stage "$n" "$nm" "$(printf '%02d' "$n")-$nm.log" \
                    "$PY" -m Condition_model.ou_forecast \
                        --arm "$(arm_param "$_arm")" \
                        --split "$DF_SPLIT" --n-traj "$DF_TRAJ" --anchors "$DF_ANCHORS" \
                        --ens "$DF_ENS" \
                        --n-lead "$DF_LEADS" --steps "$DF_STEPS" \
                        --micro-ens "$DF_MICRO_ENS" --seed "$DF_SEED" \
                        --dirname "$nm"
                ;;
        esac
    done

    set_status finished "$TOTAL" all
    log "FINISHED in $(fmt $((SECONDS)))"
    log 'deliverables:'
    local summary=''
    for _arm in "${ALL_ARMS[@]}"; do
        local nm dir have
        nm="$(arm_name "$_arm")"; dir="$(arm_dir "$_arm")"
        have=$(count_files "$ROOT/$dir" '*_predicts.npy')
        local need='temporal_difference.npz'          # the metric of record
        [[ "$(arm_kind "$_arm")" == const ]] && need+=' metrics.npz'
        log "  OK      $dir  ($have/$want windows)"
        for _f in $need; do
            [[ -e "$ROOT/$dir/$_f" ]] || log "  MISSING $dir/$_f"
        done
        [[ "$have" -eq "$want" ]] || log "  WARNING expected $want, found $have"
        summary+=" $nm:$have"
    done
    log 'all arms are on the same windows with the same innovations, so their'
    log 'metrics differ only by the latent chain. Next: temporal difference.'
    notify finished "Forecasting done:$summary"
}

FROM=1
case "${1:-}" in
    status) cmd_status ;;
    follow) cmd_follow ;;
    stop)   cmd_stop ;;
    notify-test)
        echo "config: $NOTIFY_ENV $([[ -f $NOTIFY_ENV ]] && echo '(found)' || echo '(MISSING)')"
        RUN_ID=notify-test RUN_DIR=/tmp notify finished 'Test from ./forecasting.sh'
        ;;
    start)
        RUN_ID="forecasting-$(date '+%Y%m%d-%H%M%S')"
        RUN_DIR="$ROOT/logs/$RUN_ID"
        mkdir -p "$RUN_DIR"
        ln -sfn "$RUN_DIR" "$LATEST"
        RUN_ID="$RUN_ID" RUN_DIR="$RUN_DIR" \
            setsid nohup "$0" _run >"$RUN_DIR/pipeline.log" 2>&1 < /dev/null &
        disown || true
        echo "started  $RUN_ID"
        echo "logs     $RUN_DIR/pipeline.log"
        echo
        echo "  ./forecasting.sh status   which arm it is on"
        echo "  ./forecasting.sh follow   watch it live"
        echo "  ./forecasting.sh stop     kill it"
        ;;
    _run)
        : "${RUN_ID:?}" "${RUN_DIR:?}"
        cmd_run
        ;;
    *) sed -n '3,43p' "$0"; exit 2 ;;
esac

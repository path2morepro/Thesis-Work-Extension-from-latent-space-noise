#!/usr/bin/env bash
#
# run_all.sh — the whole pipeline, detached, in order.
#
#   train -> forecast -> invert -> round trip
#
# Detached means a new session via setsid: closing the terminal, logging out or
# dropping an SSH connection does not signal it. Only `./run_all.sh stop`, a
# reboot, or the OOM killer ends it.
#
#   ./run_all.sh start              launch (returns immediately)
#   ./run_all.sh start --resume     continue an interrupted training run
#   ./run_all.sh status             which stage, how long, training progress
#   ./run_all.sh follow             tail the live log (Ctrl-C is safe)
#   ./run_all.sh stop               kill the run
#   ./run_all.sh notify-test        send a test notification (--dry to preview)
#
# Notifications on finish / failure / interruption go out through
# tools/notify.py, configured in ~/.config/thesis-notify.env (outside the repo,
# so an app password is never committed). With nothing configured the run is
# unaffected and the hooks just print "no channel configured".
#
# ONE STAGE IS ACTIVE: the RK4 inversion sector. Training and the Euler-100
# baseline inversion are DONE; forecasting and round trip are written out and
# commented in cmd_run, because their settings are not settled yet. To
# re-enable one: uncomment its run_stage line, uncomment the matching
# archive_for_stage line in archive_stale, and raise TOTAL.
#
# Stage parameters live in the CONFIG block; they are the values that fit a
# 6 GB card, so change them only with the VRAM note in mind.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# The venv and the data live in the MAIN checkout. Run from a git worktree,
# $ROOT/.venv does not exist and this used to die with a bare
# "No such file or directory", so resolve it: this checkout, then the main
# checkout, then whatever python3 is on PATH.
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
    echo "run_all.sh: no usable python found (looked in $ROOT/.venv, ${MAIN_ROOT:-?}/.venv, PATH)" >&2
    echo "            set it explicitly:  PY=/path/to/python ./run_all.sh ..." >&2
    exit 1
fi

# ── CONFIG ────────────────────────────────────────────────────────────────
# Forecasting: a window is ENS*24 rows at ~0.0144 GiB/row, so ens=20 is 480
# rows ~ 7 GiB in one call. N_MICRO=4 cuts that to 120 rows ~ 2.2 GiB. 4 was
# verified bit-identical to 2; 8 drifts ~2e-4 through the AR re-anchorings.
# Sampling costs 1.8 rows/s on this card (100 Euler steps per row, measured),
# and a trajectory is 99 leads * ENS rows, so forecasting scales brutally:
#   traj=10 ->  3.1 h    traj=50 -> 15.7 h
#   traj=20 ->  6.3 h    traj=100 -> 31.4 h   (the old hardcoded default)
# 20 keeps the metrics plot honest without costing half of the training time.
# alpha is MEASURED, not chosen: the pooled lag-1 correlation of the 1200
# inverted val latents. The directory no longer carries it (it is in that
# folder's meta.json), because 0.7133258747 is not a path component.
FC_TRAJ=${FC_TRAJ:-50}        # test trajectories to forecast
FC_ENS=${FC_ENS:-20}          # ensemble members each
FC_ALPHA=${FC_ALPHA:-0.7133258747}
FC_ANCHOR=${FC_ANCHOR:-0}     # 0 = forecast the whole trajectory, 99 leads
FC_NMICRO=${FC_NMICRO:-4}     # 480-row windows OOM; 4 gives 120 rows, 2.2 GiB
FC_DIRNAME=${FC_DIRNAME:-predict_constant_alpha}

# Invert. steps=100 only: the old 300 and 1000 directories came from the
# PREVIOUS checkpoint and mixing them with this model's latents would put two
# different models on one axis. They have been removed; meta.json in each
# output folder now records the model sha256 so the mix is detectable.
#
# 150 val trajectories x 8 anchors = 1200 windows x 24 leads = 28800 solves.
# At the measured 1.85 rows/s that is ~4.3 h, in eager fp32. torch.compile was
# tried and is NOT used: 0.90x (slower) because a single Euler step is already
# 650 ms of pure compute for 120 rows, leaving nothing to fuse.
# CURRENT RUN: a convergence probe, not a replacement for the baseline.
# The finished set (steps=100 order=1 Euler, 150 traj x 8 anchors) lives in
# data/invert/from_val and its per-lead variance runs 0.577 -> 0.943 instead
# of 1.0, which is what makes a time-varying OU hard to fit. This asks whether
# that deficit is an INTEGRATION error by redoing a small, independent subset
# with a far more accurate integrator: 300 steps of Heun instead of 100 of
# Euler, 599 network evals per row against 100.
#
# anchors=1 so the 30 windows come from 30 DIFFERENT trajectories and share no
# frames -- the distribution check needs independent draws. With anchors=1 no
# separate distribution_check_* directory is written, because the main set
# already is one window per trajectory.
#
# Output goes to data/invert/from_val_steps=300_order=2, kept apart from the
# baseline by name. Baseline values were: STEPS=100 ORDER=1 TRAJ=0 ANCHORS=8.
IV_SPLIT=${IV_SPLIT:-val}
IV_BATCH_PAIRS=${IV_BATCH_PAIRS:-5}   # 120 rows, 2.2 GiB. Raising it does NOT
                                      # go faster: throughput is flat in batch
                                      # size and 480 rows OOMs on a 6 GB card.
IV_SEED=${IV_SEED:-0}

# ── SECTOR: RK4 inversion of val ──────────────────────────────────────────
# The most accurate inversion this code can do, over the whole val split:
# RK4 at 500 steps is 1997 network evals per row against Euler-100's 100.
#
# Grew out of the 30-window probe that asked whether the variance deficit in
# data/invert/from_val (lead1 0.568, mean 0.862 against a target of 1.0) was an
# INTEGRATION error or a MODEL error. Caveat worth keeping: the final step of
# heunSample/rk4Sample drops to plain Euler, so the floor is O(dt^2) = 4e-6
# rather than machine precision -- still five orders below the 0.35 deficit.
#
# RK_ANCHORS IS THE SCALABILITY KNOB. Raise it and re-run; the folder is
# EXTENDED, not redrawn. generate_invert_data.choose_anchors() reads the
# anchors already on disk, keeps every one of them, and tops up from a
# per-trajectory permutation whose prefix does not change with k. So
# 1 -> 2 -> 3 -> 4 each add exactly RK_TRAJ windows and recompute nothing:
#
#   RK_ANCHORS=1   150 windows    30 on disk, 120 to do    ~ 8h 38m
#   RK_ANCHORS=2   300 windows   +150 to do                ~10h 48m
#   RK_ANCHORS=3   450 windows   +150 to do                ~10h 48m
#   RK_ANCHORS=4   600 windows   +150 to do                ~10h 48m
#
# At anchors=1 the 150 windows come from 150 DIFFERENT trajectories and share
# no frames, so the set is already the independent one a distribution check
# needs and no separate _check directory is written. From anchors=2 on, the
# windows of a trajectory overlap heavily (anchors 30 and 35 share 19 of their
# 24 target frames), so data/invert/from_val_rk4_check appears, holding the
# first anchor of each trajectory -- one window per trajectory, no shared
# frames. It costs no extra ODE solves.
#
# Output name is verbatim, so it carries no steps/order: meta.json is the only
# record of how these were made, and is what the resume guard compares. The 30
# windows already in from_val_rk4 were copied in by hand out of
# RK4_distribution_check (byte-identical, same model, same steps=500 order=4)
# and arrived without a meta.json, so the script prints a provenance NOTE for
# them on the first run and then writes one.
RK_ENABLE=${RK_ENABLE:-0}     # DONE: 150/150 windows in data/invert/from_val_rk4.
                              # Set to 1 only to extend it (see RK_ANCHORS).
RK_STEPS=${RK_STEPS:-500}
RK_ORDER=${RK_ORDER:-4}       # 4 = RK4
RK_TRAJ=${RK_TRAJ:-150}       # all of val
RK_ANCHORS=${RK_ANCHORS:-1}   # raise to 2,3,4... to extend. see above.
RK_OUTDIR=${RK_OUTDIR:-from_val_rk4}
# IV_BATCH_PAIRS and IV_SEED are set in the invert block above; both apply here.
# Measured row-evals/s: 185 at Euler-100 and Heun-300, 216 at RK4-500 (the
# 30-window run averaged 221.5 s/window = 24*1997/221.5). 185 is kept for the
# estimate so the printed number is an upper bound, not a hope.
IV_RATE=${IV_RATE:-185}

# ── SECTOR: round trip on test ────────────────────────────────────────────
# data -> latent -> data, leads 1..24, no ensemble. Two arms, run as four
# stages in this one invocation: invert euler-100, return euler-100, invert
# rk4-500, return rk4-500.
#
# The forward half MATCHES the backward half. A mixed pair (invert with RK4,
# return with Euler) measures neither integrator, and the old round_trip.py
# hardcoded smp.sample -- Euler -- for every arm.
#
# PAIRED BY CONSTRUCTION. choose_anchors keys on (seed, traj_id) only, never on
# steps or order, so both arms independently draw the SAME anchors from the same
# trajectories. The difference between the two curves is the integrator and
# nothing else.
#
# RT_TRAJ IS THE SCALABILITY KNOB, and trajectories are the right axis: at
# RT_ANCHORS=1 every window comes from a different trajectory and shares no
# frames. Raising RT_ANCHORS instead would add windows 8 frames apart that share
# 16 of their 24 target frames. Both halves resume by file, and the invert
# anchors are nested in k, so raising either number only costs the new windows:
#
#   RT_TRAJ=20   both arms  ~3h 01m      RT_TRAJ=100  both arms  ~15h 06m
#   RT_TRAJ=50   both arms  ~7h 33m      RT_TRAJ=150  both arms  ~22h 40m
#
# Cost is dominated by rk4-500 at 1997 evals/row against euler-100's 100, and
# a round trip pays it TWICE -- once each way. At RT_TRAJ=20: euler 8m, rk4 2h52m.
#
# No comparison arm. The old forecast_reference() read data/predict_constant_alpha,
# which does not exist, and comparison arms are out of scope. The run generates
# data and metrics.npz only; the figure is a separate opt-in step:
#     PYTHONPATH=. .venv/bin/python Analysis/round_trip.py --plot
RT_ENABLE=${RT_ENABLE:-1}
RT_SPLIT=${RT_SPLIT:-test}
RT_TRAJ=${RT_TRAJ:-20}        # windows per arm; raise to extend
RT_ANCHORS=${RT_ANCHORS:-1}   # 1 = every window from a different trajectory
RT_ARMS=${RT_ARMS:-"1:100 4:500"}   # order:steps, in run order

RT_NARMS=0
for _a in $RT_ARMS; do RT_NARMS=$(( RT_NARMS + 1 )); done

TOTAL=$(( ${RK_ENABLE:-0} + ( ${RT_ENABLE:-0} == 1 ? 2 * RT_NARMS : 0 ) ))
# ──────────────────────────────────────────────────────────────────────────

LOCK="$ROOT/.run_all.lock"
NOTIFY_PY="$ROOT/tools/notify.py"
NOTIFY_ENV="${NOTIFY_ENV:-$HOME/.config/thesis-notify.env}"
LATEST="$ROOT/logs/latest"
TRAIN_LOG_CSV="$ROOT/models/randomAnchor_n_24pred_formal_v2_training_log.csv"

# Mirror of Condition_model.generate_invert_data.out_dirs(): the baseline
# (100, 1) keeps the plain name, everything else is suffixed.
# evals per row: Euler steps, Heun 2s-1, RK4 4s-3 (last step drops to Euler)
# Mirror of generate_invert_data._suffix(): the baseline (100, 1) keeps the plain
# name, everything else carries steps and order. If that convention changes in
# python, it has to change here too -- this is the one place the shell guesses a
# path that python owns.
invert_dirname() {   # $1 split, $2 steps, $3 order
    if [[ "$2" == 100 && "$3" == 1 ]]; then echo "from_$1"
    else echo "from_$1_steps=$2_order=$3"; fi
}

invert_nfe() {       # $1 steps, $2 order
    case "$2" in
        1) echo "$1" ;;
        2) echo $(( 2 * $1 - 1 )) ;;
        4) echo $(( 4 * $1 - 3 )) ;;
        *) echo "$1" ;;
    esac
}

# Count matching files in a directory that may not exist yet.
#
# NOT `find ... | wc -l`. find exits 1 when the directory is missing, pipefail
# propagates that through wc, and set -e then kills the run on what should be a
# count of zero. That is a silent failure: the stage loop just stopped after the
# first arm, because the second arm's output directory did not exist yet.
count_files() {   # $1 dir, $2 name pattern
    [[ -d "$1" ]] || { echo 0; return 0; }
    find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l
}

fmt() {   # seconds -> 1h 02m 03s
    local s=$1
    printf '%dh %02dm %02ds' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

# ── status / inspection commands (run in the user's shell, not the job) ──
cmd_status() {
    local d; d="$(readlink -f "$LATEST" 2>/dev/null || true)"
    [[ -n "$d" && -d "$d" ]] || { echo 'no run found (logs/latest missing)'; exit 1; }

    local state stage name started pid
    state=$(sed -n 's/^state=//p'   "$d/status" 2>/dev/null || echo unknown)
    stage=$(sed -n 's/^stage=//p'   "$d/status" 2>/dev/null || echo ?)
    name=$(sed  -n 's/^name=//p'    "$d/status" 2>/dev/null || echo ?)
    started=$(sed -n 's/^started=//p' "$d/status" 2>/dev/null || echo 0)
    pid=$(cat "$d/pid" 2>/dev/null || echo '')

    local alive='no'
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then alive='yes'; fi

    echo "run        : $(basename "$d")"
    echo "state      : $state   (process alive: $alive, pid ${pid:-none})"
    echo "stage      : $stage/$TOTAL  $name"
    [[ "$started" != 0 ]] && echo "elapsed    : $(fmt $(( $(date +%s) - started )))"
    if [[ -f "$d/completed" ]]; then
        echo "completed  : $(paste -sd', ' "$d/completed")"
    fi

    # Per-stage progress. The invert stage writes one file per window, so
    # counting them IS the progress bar -- and it keeps working across a
    # resume, because finished windows stay on disk.
    if [[ "$name" == forecast ]]; then
        local fdir n pct
        fdir="$ROOT/data/$FC_DIRNAME"
        n=$(count_files "$fdir" 'sample*_predicts.npy')
        pct=$(( FC_TRAJ > 0 ? 100 * n / FC_TRAJ : 0 ))
        echo "forecast   : $n/$FC_TRAJ trajectories ($pct%)"
        if [[ "$started" != 0 && $n -gt 0 ]]; then
            local el
            el=$(( $(date +%s) - started ))
            echo "             $(awk -v n="$n" -v e="$el" 'BEGIN{printf "%.0f", e/n}')s/traj\
   ETA $(fmt $(awk -v w="$FC_TRAJ" -v n="$n" -v e="$el" \
                 'BEGIN{printf "%d", (w-n)*e/n}'))"
        fi
    fi

    # Both stage kinds write one file per window, so counting them IS the
    # progress bar -- and it keeps working across a resume, because finished
    # windows stay on disk. The job writes current_outdir and current_want
    # before each stage; reading those instead of recomputing from RK_*/RT_*
    # is what lets one block serve every arm.
    if [[ "$name" == invert* || "$name" == roundtrip* ]]; then
        local odir n want pct pat label
        odir="$ROOT/$(cat "$d/current_outdir" 2>/dev/null)"
        want=$(cat "$d/current_want" 2>/dev/null || echo 0)
        if [[ "$name" == invert* ]]; then
            pat='invert_traj*initial*.npy'; label='invert   '
        else
            pat='traj*initial*.npy';        label='roundtrip'
        fi
        n=$(count_files "$odir" "$pat")
        pct=$(( want > 0 ? 100 * n / want : 0 ))
        echo "$label  : $n/$want windows ($pct%)   $(basename "$odir")"
        if [[ "$started" != 0 && $n -gt 0 && $want -gt $n ]]; then
            local el rate
            el=$(( $(date +%s) - started ))
            rate=$(awk -v n="$n" -v e="$el" 'BEGIN{printf "%.1f", e/n}')
            echo "             ${rate}s/window (this run)   ETA $(fmt $(awk -v w="$want" \
                 -v n="$n" -v e="$el" 'BEGIN{printf "%d", (w-n)*e/n}'))"
        fi
    fi

    # Training progress is read from the CSV the trainer appends per epoch.
    # Only meaningful while the train stage is the one running: the CSV from
    # the finished 300-epoch run stays on disk forever.
    if [[ "$name" == train && -f "$TRAIN_LOG_CSV" ]]; then
        $PY - "$TRAIN_LOG_CSV" "$started" <<'PYSTAT'
import csv, sys, time
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print('training   : log open, no epoch finished yet'); raise SystemExit
ep  = int(rows[-1]['epoch'])
val = [float(r['val_loss']) for r in rows]
b   = min(val); bi = val.index(b) + 1
el  = time.time() - float(sys.argv[2] or 0)
per = el / ep if ep else 0
print(f'training   : epoch {ep}/300  train={float(rows[-1]["train_loss"]):.4f} '
      f'val={val[-1]:.4f}')
print(f'             best val={b:.4f} @ epoch {bi}   {per/60:.1f} min/epoch')
if ep < 300 and per:
    print(f'             ETA for 300 epochs: {(300-ep)*per/3600:.1f} h')
PYSTAT
    fi

    if command -v nvidia-smi >/dev/null; then
        echo "gpu        : $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
                              --format=csv,noheader 2>/dev/null | head -1)"
    fi
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
    [[ -n "$pid" ]] || { echo 'no pid recorded; nothing to stop'; exit 1; }
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "pid $pid is not running"; return 0
    fi
    # Negative pid = the whole process group (setsid made this pid the leader).
    kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    echo "sent TERM to process group $pid, waiting..."
    for _ in $(seq 30); do
        kill -0 "$pid" 2>/dev/null || { echo 'stopped cleanly'; return 0; }
        sleep 1
    done
    echo 'still alive after 30s, sending KILL'
    kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    sleep 1
    kill -0 "$pid" 2>/dev/null && echo 'WARNING: still running' || echo 'killed'
}

# ── the job itself ───────────────────────────────────────────────────────
log() { echo "[$(date '+%F %T')] $*"; }

# Runs in a SUBSHELL on purpose: sourcing the config there keeps the SMTP app
# password out of this shell's environment, so it is never inherited by the
# python training process. Always returns 0 -- a notification problem must not
# change the outcome of a 68 h run.
notify() {
    local state="$1" note="${2:-}"
    [[ -f "$NOTIFY_PY" ]] || return 0
    (
        set +e
        if [[ -f "$NOTIFY_ENV" ]]; then set -a; . "$NOTIFY_ENV"; set +a; fi
        "$PY" "$NOTIFY_PY" --state "$state" --run "${RUN_ID:-?}" \
            --stage "${CUR:-?}/$TOTAL ${CURNAME:-?}" \
            --elapsed "$(fmt ${SECONDS:-0})" \
            --log "${RUN_DIR:-}/pipeline.log" --csv "$TRAIN_LOG_CSV" \
            ${note:+--note "$note"}
    ) || true
    return 0
}

# state is what tools/watchdog.sh reads: "state=running" next to a pid that
# no longer exists means the run died without saying anything. A hard crash
# (OOM-kill, power loss) never reaches the notify hook, because the hook runs
# inside the dying process. The pid lives in $RUN_DIR/pid, written once at
# startup -- deliberately not duplicated here, to keep one source of truth.
set_status() {
    { echo "state=$1"; echo "stage=$2"; echo "name=$3"
      echo "started=$RUN_STARTED"; } > "$RUN_DIR/status"
}

# Stages overwrite by filename, so leftovers from an older model with different
# (traj, anchor) names survive and get read back as if they belonged to this
# run: round_trip.forecast_reference() globs *_predicts.npy and shared_windows()
# globs the invert dir, so neither notices the mix. These are MOVED, not
# deleted -- they cost hours to make.
#
# Only the outputs of stages that will actually run are touched, so
# `--from 3` leaves the forecast results from an earlier run alone.
ARCHIVED=0
archive_for_stage() {
    local num="$1"; shift
    [[ $num -lt $FROM ]] && return 0
    local arc="$ROOT/archive/$RUN_ID"
    for p in "$@"; do
        if [[ "$p" == *'*'* ]]; then
            compgen -G "$ROOT/$p" >/dev/null || continue
            mkdir -p "$arc/$(dirname "$p")"
            mv $ROOT/$p "$arc/$(dirname "$p")/"
        elif [[ -e "$ROOT/$p" ]]; then
            mkdir -p "$arc/$(dirname "$p")"
            mv "$ROOT/$p" "$arc/$p"
        else
            continue
        fi
        log "  archived $p"
        ARCHIVED=$((ARCHIVED+1))
    done
}

archive_stale() {
    # Neither the invert nor the forecast stage is archived, on purpose. Both
    # write one file per unit of work and both resume by skipping what is
    # already on disk, so moving the directory aside would throw away hours of
    # finished work on every restart -- 18 min per forecast trajectory.
    # Mixing generations is prevented a better way: each stage compares the
    # settings in its meta.json (model sha256, and for the forecast also
    # alpha / ens / anchor) and refuses to extend a folder that disagrees.
    #
    # archive_for_stage N "data/round_trip/steps=$IV_STEPS" \
    #                     "Analysis/Result/round_trip_rmse.png"
    if [[ $ARCHIVED -eq 0 ]]; then
        log '  nothing stale to archive'
    else
        log "  -> archive/$RUN_ID  ($ARCHIVED items)"
    fi
}

CHILD=''
CUR=0
CURNAME=init

on_term() {
    log 'received TERM/INT — shutting down'
    # The stage runs as a background child so it can be signalled by pid here.
    # Relying on the process group alone was not enough: the handler returned
    # and run_stage carried on as if nothing had happened.
    if [[ -n "$CHILD" ]] && kill -0 "$CHILD" 2>/dev/null; then
        kill -TERM "$CHILD" 2>/dev/null || true
        for _ in $(seq 20); do
            kill -0 "$CHILD" 2>/dev/null || break
            sleep 0.5
        done
        kill -0 "$CHILD" 2>/dev/null && kill -KILL "$CHILD" 2>/dev/null || true
        log "stage $CUR ($CURNAME) child $CHILD stopped"
    fi
    set_status interrupted "$CUR" "$CURNAME"
    notify interrupted 'Stopped by request (run_all.sh stop or a signal). Continue with: ./run_all.sh start --resume'
    log 'run TERMINATED'
    exit 143
}

run_stage() {
    local num="$1" name="$2" logname="$3"; shift 3
    if [[ $num -lt $FROM ]]; then
        log "--- stage $num/$TOTAL $name: SKIPPED (--from $FROM)"
        return 0
    fi
    CUR=$num; CURNAME=$name
    set_status running "$num" "$name"
    log "=== stage $num/$TOTAL  $name  START"
    log "    cmd: $*"
    log "    log: $RUN_DIR/$logname"
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
        log "!!! last 25 lines of $RUN_DIR/$logname:"
        tail -25 "$RUN_DIR/$logname" | sed 's/^/    | /'
        notify failed "$(printf 'Stage %s (%s) exited %s.\n\n%s' \
            "$num" "$name" "$rc" "$(tail -15 "$RUN_DIR/$logname")")"
        exit $rc
    fi
}

# Training reads data/train and data/val. A git worktree has neither (only the
# main checkout materialises them), so check before paying the compile warmup.
check_workspace() {
    local missing=''
    local d
    for d in data/train data/val; do
        if [[ ! -d "$ROOT/$d" ]] || [[ -z "$(ls -A "$ROOT/$d" 2>/dev/null)" ]]; then
            missing+=" $d"
        fi
    done
    [[ -z "$missing" ]] && return 0
    log "FATAL: missing or empty:$missing"
    log "       this checkout is $ROOT"
    if [[ -n "$MAIN_ROOT" && "$MAIN_ROOT" != "$ROOT" ]]; then
        log "       the data is in $MAIN_ROOT"
        log "       run it there:  cd $MAIN_ROOT && ./run_all.sh start"
    fi
    set_status failed 0 workspace
    notify failed "Wrong checkout: $ROOT has no data/train or data/val."
    exit 1
}

cmd_run() {
    exec 9>"$LOCK"
    flock -n 9 || { echo 'another run_all.sh is already running'; exit 1; }
    echo $$ > "$RUN_DIR/pid"
    trap on_term TERM INT

    RUN_STARTED=$(date +%s)
    set_status starting 0 init

    log "pipeline start   run=$RUN_ID  pid=$$  from stage $FROM${RESUME:+  (RESUMING)}"
    IV_NAMES=([1]=Euler [2]=Heun [4]=RK4)
    log "stages active    $TOTAL; train and the baseline invert are DONE"
    # Measured throughput is 185 row-evals/s on this card (Euler-100 at
    # 1.85 rows/s; the Heun-300 smoke run reproduced it at 185.5).
    IV_TOTSEC=0
    if [[ "${RK_ENABLE:-0}" == 1 ]]; then
        # Count what is already there. The estimate has to be for the work
        # that will actually run, otherwise raising RK_ANCHORS prints the cost
        # of the whole set every time and the ETA in `status` never matches.
        RK_WANT=$(( RK_TRAJ * RK_ANCHORS ))
        RK_HAVE=$(count_files "$ROOT/data/invert/$RK_OUTDIR" \
                                 'invert_traj*initial*.npy')
        RK_LEFT=$(( RK_WANT > RK_HAVE ? RK_WANT - RK_HAVE : 0 ))
        RK_ROWS=$(( RK_LEFT * 24 ))
        RK_NFE=$(invert_nfe "$RK_STEPS" "$RK_ORDER")
        RK_SEC=$(awk -v r="$RK_ROWS" -v n="$RK_NFE" -v q="$IV_RATE" \
                     'BEGIN{printf "%d", r*n/q}')
        IV_TOTSEC=$(( IV_TOTSEC + RK_SEC ))
        log "RK4 sector       split=$IV_SPLIT steps=$RK_STEPS order=$RK_ORDER \
(${IV_NAMES[$RK_ORDER]}) traj=$RK_TRAJ anchors=$RK_ANCHORS seed=$IV_SEED"
        log "                 $RK_WANT windows wanted, $RK_HAVE on disk, \
$RK_LEFT to compute -> data/invert/$RK_OUTDIR"
        log "                 $RK_ROWS rows x $RK_NFE evals/row @ $IV_RATE/s \
~$(fmt $RK_SEC)"
        if [[ $RK_LEFT -eq 0 ]]; then
            log "                 NOTE nothing to do at anchors=$RK_ANCHORS; \
raise RK_ANCHORS to extend the set"
        fi
    fi
    if [[ "${RT_ENABLE:-0}" == 1 ]]; then
        log "round trip       split=$RT_SPLIT traj=$RT_TRAJ anchors=$RT_ANCHORS \
arms='$RT_ARMS' seed=$IV_SEED"
        RT_WANT=$(( RT_TRAJ * RT_ANCHORS ))
        for _arm in $RT_ARMS; do
            _o=${_arm%%:*}; _st=${_arm##*:}
            _nm=${IV_NAMES[$_o]}; _lc=${_nm,,}
            _idir="data/invert/$(invert_dirname "$RT_SPLIT" "$_st" "$_o")"
            _rdir="data/round_trip/${_lc}_steps=$_st"
            # Each half is counted against what is already on disk, so the
            # printed number is the work this invocation will actually do.
            _ih=$(count_files "$ROOT/$_idir" 'invert_traj*initial*.npy')
            _rh=$(count_files "$ROOT/$_rdir" 'traj*initial*.npy')
            _il=$(( RT_WANT > _ih ? RT_WANT - _ih : 0 ))
            _rl=$(( RT_WANT > _rh ? RT_WANT - _rh : 0 ))
            _nfe=$(invert_nfe "$_st" "$_o")
            _sec=$(awk -v a="$_il" -v b="$_rl" -v n="$_nfe" -v q="$IV_RATE" \
                      'BEGIN{printf "%d", (a+b)*24*n/q}')
            IV_TOTSEC=$(( IV_TOTSEC + _sec ))
            log "  $_lc-$_st        $_nfe evals/row  invert $_il/$RT_WANT to do, \
return $_rl/$RT_WANT to do  ~$(fmt $_sec)"
        done
    fi
    log "estimate         ~$(fmt $IV_TOTSEC) (upper bound; resume skips \
finished windows)"
    log "python           $PY"
    # A systemd user session gives a soft limit of 1024 descriptors (hard
    # 524288). The dataset now bounds its memmap cache, so this is belt and
    # braces -- but the sampling stages open files too, and raising it costs
    # nothing.
    ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
    log "open files       soft $(ulimit -Sn)  hard $(ulimit -Hn)"
    check_workspace
    log 'archiving stale outputs from previous runs'
    archive_stale

    IV_N=0
    if [[ "${RK_ENABLE:-0}" == 1 ]]; then
        IV_N=$(( IV_N + 1 ))
        echo "data/invert/$RK_OUTDIR"           > "$RUN_DIR/current_outdir"
        echo "$(( RK_TRAJ * RK_ANCHORS ))"      > "$RUN_DIR/current_want"
        run_stage "$IV_N" "invert-rk4" \
            "$(printf '%02d' "$IV_N")-invert-$RK_OUTDIR.log" \
            "$PY" -m Condition_model.generate_invert_data \
                --split "$IV_SPLIT" --steps "$RK_STEPS" --order "$RK_ORDER" \
                --outdir "$RK_OUTDIR" \
                --n-traj "$RK_TRAJ" --anchors "$RK_ANCHORS" \
                --batch-pairs "$IV_BATCH_PAIRS" --seed "$IV_SEED"
    fi

    # ROUND TRIP. Two stages per arm, invert then return, in that order: the
    # return stage can then only ever read latents the stage before it finished,
    # so the two halves cannot drift apart. round_trip.py takes no --n-traj --
    # it uses every window in the invert directory, which makes the invert stage
    # the single knob for how many windows an arm has.
    if [[ "${RT_ENABLE:-0}" == 1 ]]; then
        RT_WANT=$(( RT_TRAJ * RT_ANCHORS ))
        for _arm in $RT_ARMS; do
            _o=${_arm%%:*}; _st=${_arm##*:}
            _nm=${IV_NAMES[$_o]}; _lc=${_nm,,}
            _idir="data/invert/$(invert_dirname "$RT_SPLIT" "$_st" "$_o")"
            _rdir="data/round_trip/${_lc}_steps=$_st"

            IV_N=$(( IV_N + 1 ))
            echo "$_idir"    > "$RUN_DIR/current_outdir"
            echo "$RT_WANT"  > "$RUN_DIR/current_want"
            run_stage "$IV_N" "invert-$_lc-$_st" \
                "$(printf '%02d' "$IV_N")-invert-$_lc-$_st.log" \
                "$PY" -m Condition_model.generate_invert_data \
                    --split "$RT_SPLIT" --steps "$_st" --order "$_o" \
                    --n-traj "$RT_TRAJ" --anchors "$RT_ANCHORS" \
                    --batch-pairs "$IV_BATCH_PAIRS" --seed "$IV_SEED"

            IV_N=$(( IV_N + 1 ))
            echo "$_rdir"    > "$RUN_DIR/current_outdir"
            echo "$RT_WANT"  > "$RUN_DIR/current_want"
            run_stage "$IV_N" "roundtrip-$_lc-$_st" \
                "$(printf '%02d' "$IV_N")-roundtrip-$_lc-$_st.log" \
                env PYTHONPATH="$ROOT" "$PY" Analysis/round_trip.py \
                    --split "$RT_SPLIT" --steps "$_st" --order "$_o"
        done
    fi


    # ----------------------------------------------------------------------
    # FORECAST: ready to run, waiting on the inversion question above. Flip
    # the two run_stage blocks to switch back to it.
    #
    # run_stage N forecast  0N-forecast.log \
    #     "$PY" -m Condition_model.forecasting \
    #         --eval-traj-num "$FC_TRAJ" --ens "$FC_ENS" --alpha "$FC_ALPHA" \
    #         --anchor "$FC_ANCHOR" --n-micro "$FC_NMICRO" \
    #         --steps 100 --dirname "$FC_DIRNAME"
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # INVERT: DONE. 1200 windows from val in 4.25 h, plus 150 independent
    # windows in data/invert/distribution_check_val. The pooled lag-1
    # correlation of those latents is FC_ALPHA above. Re-running is safe --
    # it resumes by skipping finished windows and refuses to add to a folder
    # written by a different model.
    #
    # The exact command that produced it (IV_STEPS/IV_TRAJ/IV_ANCHORS are no
    # longer defined above, so these are literals, not variables):
    #
    # run_stage N invert    0N-invert.log \
    #     "$PY" -m Condition_model.generate_invert_data \
    #         --split val --steps 100 --order 1 --n-traj 0 \
    #         --anchors 8 --batch-pairs 5 --seed 0
    #
    # TRAINING: DONE. 300 epochs, best val 0.1696, saved as
    # models/randomAnchor_n_24pred_formal_v2.pth. Left here so the command is
    # on record; re-enable only to train a NEW model, and move the existing
    # checkpoint aside first (train.py refuses to overwrite it).
    #
    # run_stage N train     0N-train.log \
    #     "$PY" -m Condition_model.train ${RESUME:+--resume}
    #
    # ROUND TRIP: settings still unconfirmed. It reads the forecast directory
    # through Analysis/round_trip.forecast_reference(), which now points at
    # data/predict_constant_alpha.
    #
    # run_stage N roundtrip 0N-roundtrip.log \
    #     env PYTHONPATH="$ROOT" "$PY" Analysis/round_trip.py \
    #         --steps "$IV_STEPS" --windows "$RT_WINDOWS"
    # ----------------------------------------------------------------------

    set_status finished $TOTAL all
    log "pipeline FINISHED in $(fmt $((SECONDS)))"
    log 'deliverables:'
    IV_SUMMARY=''
    if [[ "${RT_ENABLE:-0}" == 1 ]]; then
        RT_WANT=$(( RT_TRAJ * RT_ANCHORS ))
        for _arm in $RT_ARMS; do
            _o=${_arm%%:*}; _st=${_arm##*:}
            _nm=${IV_NAMES[$_o]}; _lc=${_nm,,}
            _rdir="data/round_trip/${_lc}_steps=$_st"
            _rh=$(count_files "$ROOT/$_rdir" 'traj*initial*.npy')
            log "  OK      $_rdir  ($_rh/$RT_WANT windows)"
            [[ -e "$ROOT/$_rdir/metrics.npz" ]] \
                || log "  MISSING $_rdir/metrics.npz"
            [[ "$_rh" -eq "$RT_WANT" ]] \
                || log "  WARNING expected $RT_WANT windows, found $_rh"
            IV_SUMMARY+=" ${_lc}-${_st}:$_rh"
        done
    fi
    if [[ "${RK_ENABLE:-0}" == 1 ]]; then
        _want=$(( RK_TRAJ * RK_ANCHORS ))
        _have=$(count_files "$ROOT/data/invert/$RK_OUTDIR" 'invert_traj*.npy')
        log "  OK      data/invert/$RK_OUTDIR  ($_have/$_want windows)"
        [[ -e "$ROOT/data/invert/$RK_OUTDIR/meta.json" ]] \
            || log "  MISSING data/invert/$RK_OUTDIR/meta.json"
        [[ "$_have" -eq "$_want" ]] \
            || log "  WARNING expected $_want windows, found $_have"
        IV_SUMMARY=" $RK_OUTDIR:$_have"
    fi
    if [[ "${RT_ENABLE:-0}" == 1 ]]; then
        log "next: draw the figure (no data is made, nothing is compared):"
        log "        PYTHONPATH=. .venv/bin/python Analysis/round_trip.py --plot"
        log "      to add windows:  RT_TRAJ=50 ./run_all.sh start"
        log "      NOTE a small round-trip residual does NOT prove the inversion"
        log "      is faithful -- backward-then-forward with one solver at one"
        log "      step count partially cancels its own local error."
    fi
    notify finished "Round trip done:$IV_SUMMARY. \
Figure: Analysis/round_trip.py --plot"
}

# ── dispatch ─────────────────────────────────────────────────────────────
ACTION="${1:-status}"; shift || true
# Must default, not assign: `start` exports FROM to the detached child, which
# re-enters this same dispatch block. A bare FROM=1 here silently threw that
# away and the child ran every stage regardless of --from.
FROM="${FROM:-1}"
RESUME="${RESUME:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)   FROM="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --dry)    DRY=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

case "$ACTION" in
    status) cmd_status ;;
    notify-test)
        RUN_ID='notify-test' RUN_DIR="$ROOT/logs" CUR=1 CURNAME=train
        if [[ "${1:-}" == '--dry' || "${DRY:-}" == 1 ]]; then
            ( set +e
              if [[ -f "$NOTIFY_ENV" ]]; then set -a; . "$NOTIFY_ENV"; set +a; fi
              "$PY" "$NOTIFY_PY" --state finished --run notify-test \
                  --stage '1/1 train' --elapsed '0h 00m 01s' \
                  --csv "$TRAIN_LOG_CSV" --dry-run )
        else
            echo "config: $NOTIFY_ENV $([[ -f $NOTIFY_ENV ]] && echo '(found)' || echo '(MISSING)')"
            notify finished 'This is a test notification from ./run_all.sh notify-test'
        fi
        ;;
    follow) cmd_follow ;;
    stop)   cmd_stop ;;
    start)
        RUN_ID="run-$(date '+%Y%m%d-%H%M%S')"
        RUN_DIR="$ROOT/logs/$RUN_ID"
        mkdir -p "$RUN_DIR"
        ln -sfn "$RUN_DIR" "$LATEST"
        # setsid puts the job in its own session, so no terminal can SIGHUP it.
        RUN_ID="$RUN_ID" RUN_DIR="$RUN_DIR" FROM="$FROM" RESUME="$RESUME" \
            setsid nohup "$0" _run >"$RUN_DIR/pipeline.log" 2>&1 < /dev/null &
        disown || true
        echo "started  $RUN_ID"
        echo "logs     $RUN_DIR/pipeline.log"
        echo
        [[ -n "$RESUME" ]] && echo 'mode     resuming from the saved epoch'
        echo "  ./run_all.sh status     which stage it is on"
        echo "  ./run_all.sh follow     watch it live"
        echo "  ./run_all.sh stop       kill it"
        ;;
    _run)
        : "${RUN_ID:?}" "${RUN_DIR:?}"
        cmd_run
        ;;
    *) sed -n '3,22p' "$0"; exit 2 ;;
esac

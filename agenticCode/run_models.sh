#!/bin/bash
# Runs the agent replay for each model separately, then compare + metrics for each.
# Classic is NOT re-run: its results are snapshotted once and reused.
#
#   nohup ./run_models.sh > runs.out 2>&1 &            # gemma, then qwen
#   nohup ./run_models.sh qwen3.8-27b > runs.out 2>&1 &  # one model only
#   SMOKE=0 ./run_models.sh                             # skip the 1-hour pre-check
#
# Per model, everything lives in ./runs/<model>/ on the host:
#   trace.jsonl timing.csv llm_calls.jsonl logs/ classic/ comparison/ agent.log ...
# Predictions go to their own Ditto prefix (historicVCIpredictions<Model>), so nothing
# has to be deleted and the runs cannot overwrite each other.
cd "$(dirname "$0")"
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(gemma-4-26b-a4b qwen3.8-27b)
QUICK="${QUICK:-0}"                 # QUICK=1: 4 sessions per model, separate dirs and prefixes
SMOKE="${SMOKE:-1}"
RUNS=runs
[ "$QUICK" = "1" ] && { RUNS=runs_quick; SMOKE=0; }
mkdir -p runs "$RUNS"
P="$RUNS/progress.log"
log() { echo "$(date '+%F %T') $*" | tee -a "$P"; }
dc() { docker compose run --rm "$@"; }

TEST_START="$(grep -E '^TEST_START=' .env | tail -1 | cut -d= -f2- | tr -d '"')"
[ -n "$TEST_START" ] || { log "ABORT: TEST_START not set in .env"; exit 1; }

QEXTRA=()
if [ "$QUICK" = "1" ]; then
  QEND="$(date -u -d "@$(( $(date -u -d "${TEST_START/T/ }" +%s) + 7200 ))" '+%Y-%m-%dT%H:%M:%S')"
  QEXTRA=(-e TEST_END="$QEND")
  log "QUICK mode: $TEST_START .. $QEND (+ final session)"
fi

# 1. snapshot the classic reference (once)
REF=runs/classic_ref
mkdir -p "$REF"
NCLASSIC=$(ls "$REF"/classic/hour_*.json 2>/dev/null | wc -l)
if [ "$NCLASSIC" -eq 0 ]; then   # try the output volume of this machine (where run_week.sh ran classic)
  dc --entrypoint sh -v "$PWD/$REF:/dst" coordinator -c "cp -r /app/output/classic /dst/" >> "$P" 2>&1
  NCLASSIC=$(ls "$REF"/classic/hour_*.json 2>/dev/null | wc -l)
fi
if [ "$NCLASSIC" -eq 0 ] && [ "${RUN_CLASSIC:-0}" = "1" ]; then   # or run classic here (~50 min, no LLM)
  log "running classic into $REF ..."
  dc -v "$PWD/$REF:/app/output" --entrypoint python coordinator classic_runner.py > "$REF/classic.log" 2>&1
  NCLASSIC=$(ls "$REF"/classic/hour_*.json 2>/dev/null | wc -l)
fi
[ "$NCLASSIC" -gt 0 ] || { log "ABORT: no classic results. Re-run with RUN_CLASSIC=1 to generate them (~50 min)."; exit 1; }
log "classic reference: $NCLASSIC hours"

# run one agent replay: $1=run dir  $2=model  $3=prefix  [extra docker -e options...]
run_agent() {
  local dir="$1" model="$2" prefix="$3"; shift 3
  dc -v "$PWD/$dir:/app/output" -v "$PWD/$dir/logs:/app/logs" \
     -e RUN_MODE=test -e GOOSE_MODEL="$model" -e PREDICTION_THING_PREFIX="$prefix" "$@" \
     coordinator
}

# does the MCP server really write under $2? (looks the first prediction up under that prefix)
prefix_check() {
  local dir="$1" prefix="$2"
  dc -v "$PWD/$dir:/app/output" -e CHK_PREFIX="$prefix" -e CHK_TS="$TEST_START" --entrypoint python coordinator -c "
import asyncio, os
from classic_runner import load_server
from fastmcp import Client
srv = load_server(experiment_tag='check', trace_file='/tmp/chk.jsonl', prediction_prefix=os.environ['CHK_PREFIX'])
async def main():
    async with Client(srv.mcp) as c:
        r = await c.call_tool('get_pending_prediction', {'sensor_id': '741', 'direction': '1.0', 'target_timestamp': os.environ['CHK_TS']})
        print('CHECK_RESULT', r.structured_content or r)
asyncio.run(main())
" 2>&1 | grep -a CHECK_RESULT
}

for M in "${MODELS[@]}"; do
  word="${M%%[-0-9.]*}"                       # gemma-4-26b-a4b -> gemma ; qwen3.8-27b -> qwen
  PREFIX="historicVCIpredictions${word^}"
  [ "$QUICK" = "1" ] && PREFIX="${PREFIX}Quick"
  DIR="$RUNS/$M"
  log "=== $M  prefix=$PREFIX  dir=$DIR"

  # 2. optional 1-hour pre-check: delegation happens and the prefix is honoured
  if [ "$SMOKE" = "1" ]; then
    SDIR="$RUNS/_smoke_$M"
    rm -rf "$SDIR"; mkdir -p "$SDIR/logs"
    run_agent "$SDIR" "$M" "${PREFIX}Smoke" -e TEST_END="$TEST_START" > "$SDIR/agent.log" 2>&1
    ND=$(grep -a -c "▸ delegate" "$SDIR/agent.log")
    CHK=$(prefix_check "$SDIR" "${PREFIX}Smoke")
    log "smoke $M: delegate calls=$ND  prefix check: $CHK"
    if [ "$ND" -eq 0 ] || ! echo "$CHK" | grep -q pending_evaluation; then
      log "SKIP $M: smoke check failed (no delegation or prefix not applied). Fix, or run with SMOKE=0."
      continue
    fi
  fi

  # 3. the real replay
  rm -rf "$DIR"; mkdir -p "$DIR/logs"
  if [ "$QUICK" = "1" ]; then   # only the first hours, so compare stays fast
    mkdir -p "$DIR/classic"
    ls "$REF"/classic/hour_*.json | sort | head -4 | xargs -I{} cp {} "$DIR/classic/"
  else
    cp -r "$REF/classic" "$DIR/classic"
  fi
  log "agent start $M"
  run_agent "$DIR" "$M" "$PREFIX" "${QEXTRA[@]}" > "$DIR/agent.log" 2>&1
  log "agent done $M (exit $?)"
  log "  delegate calls: $(grep -a -c '▸ delegate' "$DIR/agent.log")  llm requests: $(wc -l < "$DIR/llm_calls.jsonl" 2>/dev/null || echo 0)  usage from server: $(grep -c '"usage_source": "server"' "$DIR/llm_calls.jsonl" 2>/dev/null || echo 0)"

  # 4. compare + metrics, reading from the same run dir
  dc -v "$PWD/$DIR:/app/output" -v "$PWD/$DIR/logs:/app/logs" --entrypoint python coordinator \
     compare.py --agent-prefix "$PREFIX" > "$DIR/compare.log" 2>&1
  log "compare done $M (exit $?)"
  dc -v "$PWD/$DIR:/app/output" --entrypoint python coordinator \
     metrics.py --model "$M" > "$DIR/metrics.log" 2>&1
  log "metrics done $M (exit $?)"
done
log "ALL DONE"

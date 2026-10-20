#!/bin/sh
set -eu

MODE="${RUN_MODE:-production}"
RECIPE="${COORDINATOR_RECIPE:-/app/coordinator_recipe.yaml}"
MAX_TURNS="${GOOSE_MAX_TURNS:-150}"
OUT=/app/output
PROVIDER_JSON=/root/.config/goose/custom_providers/custom_skynetollama.json
PROVIDER_ORIG=/root/.config/goose/custom_providers/custom_skynetollama.json.orig
PROXY_PORT="${LLM_PROXY_PORT:-8089}"

mkdir -p "$OUT" /app/logs

# ---- LLM logging proxy: records tokens, latency and finish_reason per request ----
if [ "${LLM_PROXY:-1}" = "1" ]; then
    [ -f "$PROVIDER_ORIG" ] || cp "$PROVIDER_JSON" "$PROVIDER_ORIG"
    LLM_UPSTREAM_URL="$(python3 -c "import json;print(json.load(open('$PROVIDER_ORIG'))['base_url'])")"
    export LLM_UPSTREAM_URL
    # point goose at the proxy: same path, local host
    python3 - <<PY
import json
from urllib.parse import urlsplit
d = json.load(open("$PROVIDER_ORIG"))
path = urlsplit(d["base_url"]).path
d["base_url"] = "http://127.0.0.1:$PROXY_PORT" + path
json.dump(d, open("$PROVIDER_JSON", "w"), indent=2)
PY
    export LLM_LOG="$OUT/llm_calls.jsonl" LLM_HOUR_FILE="$OUT/current_hour.txt"
    python3 /app/llm_proxy.py &
    i=0
    until python3 -c "import socket;socket.create_connection(('127.0.0.1',$PROXY_PORT),1)" 2>/dev/null; do
        i=$((i+1)); [ "$i" -gt 20 ] && { echo "llm_proxy did not start"; exit 1; }
        sleep 0.5
    done
else
    # run without the proxy: restore the original provider file if a previous run replaced it
    [ -f "$PROVIDER_ORIG" ] && cp "$PROVIDER_ORIG" "$PROVIDER_JSON"
fi

[ -f "$OUT/timing.csv" ] || echo "model,hour,start_epoch,end_epoch,seconds,goose_exit" > "$OUT/timing.csv"

run_hour() {
    timestamp="$1"
    log_file="/app/logs/$(echo "$timestamp" | tr ':' '-').log"

    echo "- Hour: $timestamp"
    printf '%s' "$timestamp" > "$OUT/current_hour.txt"
    start=$(date +%s)
    rm -f /tmp/goose_rc
    {
        rc=0
        SIMULATION_START="$timestamp" goose run \
            --recipe "$RECIPE" \
            --no-profile \
            --with-builtin summon \
            --with-extension "traffic-detector:python /app/mcp_server.py" \
            --max-turns "$MAX_TURNS" \
            --max-tool-repetitions 2 \
            --debug || rc=$?
        echo "$rc" > /tmp/goose_rc
    } 2>&1 | tee "$log_file"
    end=$(date +%s)
    echo "${GOOSE_MODEL:-unknown},$timestamp,$start,$end,$((end-start)),$(cat /tmp/goose_rc 2>/dev/null || echo NA)" >> "$OUT/timing.csv"
}

if [ "$MODE" = "test" ]; then
    # TEST_START and TEST_END are inclusive prediction hours.
    # The extra final session evaluates the prediction made at TEST_END.
    if [ -z "${TEST_START:-}" ] || [ -z "${TEST_END:-}" ]; then
        echo "TEST_START and TEST_END must be set in .env for test mode"
        exit 1
    fi

    echo "Running in TEST mode: replaying $TEST_START through $TEST_END"

    current="$TEST_START"
    while [ "$(python3 -c "from datetime import datetime; print(datetime.fromisoformat('$current') <= datetime.fromisoformat('$TEST_END'))")" = "True" ]; do
        run_hour "$current"
        current="$(python3 -c "from datetime import datetime, timedelta; print((datetime.fromisoformat('$current') + timedelta(hours=1)).isoformat())")"
    done

    echo "- Final evaluation hour: $current"
    run_hour "$current"

    echo "Replay complete: $TEST_START through $TEST_END"

else
    echo "Running in PRODUCTION mode"

    while true; do
        current="$(date -u '+%Y-%m-%dT%H:00:00')"
        run_hour "$current"

        next_epoch="$(date -u -d "$current + 1 hour" +%s)"
        now_epoch="$(date -u +%s)"
        sleep_seconds=$((next_epoch - now_epoch))

        if [ "$sleep_seconds" -gt 0 ]; then
            sleep "$sleep_seconds"
        fi
    done
fi

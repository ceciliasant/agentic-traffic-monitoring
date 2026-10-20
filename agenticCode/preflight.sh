#!/bin/bash
# Checks machine before a long run.
cd "$(dirname "$0")"
ok(){ echo "  OK   $*"; }; bad(){ echo "  FAIL $*"; FAILED=1; }
FAILED=0
echo "== tools"
command -v docker >/dev/null && ok "docker" || bad "docker not found"
docker compose version >/dev/null 2>&1 && ok "docker compose" || bad "docker compose plugin missing"
command -v tmux >/dev/null && ok "tmux (use it so the run survives SSH)" || echo "  WARN tmux not installed; use nohup (see below)"
echo "== files"
for f in .env docker-compose.yaml Dockerfile entrypoint.sh llm_proxy.py metrics.py compare.py classic_runner.py mcp_server.py coordinator_recipe.yaml run_models.sh; do
  [ -f "$f" ] && ok "$f" || bad "missing $f"
done
grep -q "llm_proxy.py" Dockerfile && ok "Dockerfile copies llm_proxy.py" || bad "Dockerfile does not copy llm_proxy.py / metrics.py"
grep -q 'getenv("PREDICTION_THING_PREFIX"' mcp_server.py && ok "mcp_server.py reads PREDICTION_THING_PREFIX" || bad "mcp_server.py still hard-codes the prediction prefix"
grep -q "PREDICTION_THING_PREFIX" config.yaml && ok "config.yaml passes PREDICTION_THING_PREFIX" || echo "  WARN config.yaml env_keys lacks PREDICTION_THING_PREFIX"
grep -q "delegate" coordinator_recipe.yaml && ok "coordinator recipe mentions delegate" || bad "coordinator recipe does not mention delegate"
for k in TEST_START TEST_END OPENWEBUI_API_KEY USER_DITTO PASS_DITTO TOKEN_URL CLIENT_ID VCI_HISTORIC_DITTO_URL AWS_ENDPOINT_URL; do
  grep -qE "^$k=.+" .env && ok ".env $k" || bad ".env lacks $k"
done
echo "== classic reference"
if [ -d runs/classic_ref/classic ]; then ok "runs/classic_ref/classic: $(ls runs/classic_ref/classic/hour_*.json | wc -l) hours"
else bad "runs/classic_ref/classic missing (copy it from the machine that ran classic, or run classic here)"; fi
echo "== build + connectivity (from inside the container)"
docker compose build -q coordinator >/dev/null 2>&1 && ok "image builds" || bad "docker compose build failed"
docker compose run --rm --entrypoint python coordinator -c "
import os, sys, httpx, json
from APItoken import AuthenticationService
bad = 0
def rep(ok, msg):
    global bad
    print(('  OK   ' if ok else '  FAIL ') + msg); bad += (not ok)
try:
    AuthenticationService().get_token(); rep(True, 'Ditto token')
except Exception as e: rep(False, f'Ditto token: {e}')
try:
    r = httpx.get(os.environ['VCI_HISTORIC_DITTO_URL'] + 'things', params={'thing_id_prefix': 'historicFlowData'},
                  headers={'Authorization': 'Bearer ' + AuthenticationService().get_token()}, timeout=20)
    rep(r.status_code == 200, f'historic Ditto things -> {r.status_code}')
except Exception as e: rep(False, f'historic Ditto: {e}')
cfg = json.load(open('/root/.config/goose/custom_providers/custom_skynetollama.json.orig' if os.path.exists('/root/.config/goose/custom_providers/custom_skynetollama.json.orig') else '/root/.config/goose/custom_providers/custom_skynetollama.json'))
for m in [x['name'] for x in cfg['models']]:
    try:
        r = httpx.post(cfg['base_url'], headers={'Authorization': 'Bearer ' + os.environ['OPENWEBUI_API_KEY']},
                       json={'model': m, 'messages': [{'role': 'user', 'content': 'Reply with OK'}], 'max_tokens': 8}, timeout=120)
        u = (r.json() or {}).get('usage') if r.status_code == 200 else None
        rep(r.status_code == 200, f'LLM {m} -> {r.status_code} usage={u}')
    except Exception as e: rep(False, f'LLM {m}: {e}')
sys.exit(1 if bad else 0)
" 2>&1 | grep -a "OK  \|FAIL " ; [ "${PIPESTATUS[0]}" = "0" ] || FAILED=1
echo
[ "$FAILED" = "0" ] && echo "preflight passed" || echo "preflight FAILED: fix the lines above first"
exit $FAILED

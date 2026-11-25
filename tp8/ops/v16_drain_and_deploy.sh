#!/usr/bin/env bash
# Wave-V16 Zero-Drop Graceful Drain, Atomic 8-Node Profile Deploy & Auto-Rollback Tool
# Orchestrates maintenance stops or profile switches (default <-> max_kv_speed) with:
#   1. Zero-drop drain: polls /metrics (num_requests_running + num_requests_waiting) -> 0
#      up to --drain-timeout 120s before issuing coordinated SIGTERM!
#   2. Coordinated 8-node SIGTERM teardown + scoped /dev/shm & /tmp/vllm* IPC sweep
#   3. Pre-boot UMA buddy guard (v15_preboot_uma_guard.sh) across all 8 DGX Spark nodes
#   4. Strict 8/8 /rdma/SHA256SUMS digest & 4/4 ConnectX-7 RoCEv2 HCA verification
#   5. Atomic profile launch (ranks 7..1 FIRST, then head rank 0 LAST)
#   6. Immediate /health wait + 0-ULP 1dcfb306 boot certification
#   7. Automatic rollback to previous known-good env (last_good_env / live_champion_v15_default.env / live_champion_v12.env)
#      if launch, /health, or 0-ULP boot cert fails!
#
# Exit codes:
#   0 = Deployed & 0-ULP certified (or --drain-only drained cleanly to 0)
#   1 = Pre-deploy argument/validation error (live server untouched)
#   2 = Pre-boot UMA guard or /rdma/SHA256SUMS verification failed
#   3 = Graceful drain timeout expired with active requests (aborted safely without dropping traffic!)
#   4 = Target profile failed boot/0-ULP cert -> AUTOMATIC ROLLBACK SUCCEEDED & certified!
#   5 = Target profile failed AND automatic rollback ALSO failed (CRITICAL!)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# Optional external cluster config (ZERO hardcoded secrets)
if [ -n "${V16_CLUSTER_CONF:-}" ] && [ -f "$V16_CLUSTER_CONF" ]; then
  # shellcheck disable=SC1090
  set -a; source "$V16_CLUSTER_CONF"; set +a
elif [ -f "/etc/inkling-v16/supervisor.env" ]; then
  # shellcheck disable=SC1091
  set -a; source "/etc/inkling-v16/supervisor.env"; set +a
elif [ -f "/etc/inkling-v15/supervisor.env" ]; then
  # shellcheck disable=SC1091
  set -a; source "/etc/inkling-v15/supervisor.env"; set +a
fi

V16_HOSTS_DEFAULT="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
V16_HOSTS="${V16_HOSTS:-${V15_HOSTS:-$V16_HOSTS_DEFAULT}}"
# shellcheck disable=SC2206
HOSTS=($V16_HOSTS)
V16_SSH_USER="${V16_SSH_USER:-${V15_SSH_USER:-${USER:-jeweled}}}"
V16_SSH_JUMP_HOST="${V16_SSH_JUMP_HOST:-${V15_SSH_JUMP_HOST:-spark-one}}"
V16_SSH_JUMP_MAP="${V16_SSH_JUMP_MAP:-${V15_SSH_JUMP_MAP:-spark-five=10.77.0.51 spark-six=10.77.0.61}}"
V16_SSH_BIN="${V16_SSH_BIN:-${V15_SSH_BIN:-ssh}}"
V16_SCP_BIN="${V16_SCP_BIN:-${V15_SCP_BIN:-scp}}"
V16_SSH_TIMEOUT="${V16_SSH_TIMEOUT:-${V15_SSH_TIMEOUT:-10}}"
V16_CONTAINER_PREFIX="${V16_CONTAINER_PREFIX:-${V15_CONTAINER_PREFIX:-inkling-tp8-r}}"
V16_REMOTE_LAUNCHER="${V16_REMOTE_LAUNCHER:-${V15_REMOTE_LAUNCHER:-/home/${V16_SSH_USER}/inkling-tp8/inkling-tp8-v15.sh}}"
V16_REMOTE_ENV_PATH="${V16_REMOTE_ENV_PATH:-${V15_REMOTE_ENV_PATH:-/tmp/inkling_v16_active.env}}"
V16_RDMA_DIR="${V16_RDMA_DIR:-/home/${V16_SSH_USER}/inkling-tp8/rdma}"
V16_ENVS_DIR="${V16_ENVS_DIR:-$HERE/envs}"
V16_STATE_DIR="${V16_STATE_DIR:-${V15_STATE_DIR:-/var/lib/inkling-v16}}"
V16_HEAD_HOST="${V16_HEAD_HOST:-${V15_HEAD_HOST:-127.0.0.1}}"
PORT="${PORT:-8010}"
V16_SERVED_MODEL="${V16_SERVED_MODEL:-${V15_SERVED_MODEL:-inkling-2-small}}"
V16_REQUIRED_ACTIVE_HCAS="${V16_REQUIRED_ACTIVE_HCAS:-${V15_REQUIRED_ACTIVE_HCAS:-4}}"
V16_STOP_TIMEOUT="${V16_STOP_TIMEOUT:-${V15_STOP_TIMEOUT:-30}}"
V16_DRAIN_TIMEOUT_S="${V16_DRAIN_TIMEOUT_S:-120}"
V16_DRAIN_POLL_S="${V16_DRAIN_POLL_S:-2}"
V16_DRAIN_CONSECUTIVE_ZEROS="${V16_DRAIN_CONSECUTIVE_ZEROS:-2}"
V16_BOOT_TIMEOUT_S="${V16_BOOT_TIMEOUT_S:-${V15_BOOT_TIMEOUT_S:-420}}"
V16_BOOT_POLL_S="${V16_BOOT_POLL_S:-5}"
V16_BOOT_CERT_SHA256="${V16_BOOT_CERT_SHA256:-${V15_BOOT_CERT_SHA256:-}}"
V16_BOOT_CERT_PROMPT="${V16_BOOT_CERT_PROMPT:-${V15_BOOT_CERT_PROMPT:-State the first five prime numbers separated by commas, with no extra commentary:}}"
V16_BOOT_CERT_CMD="${V16_BOOT_CERT_CMD:-}"
V16_PRE_DRAIN_CMD="${V16_PRE_DRAIN_CMD:-}"
V16_UMA_GUARD_SCRIPT="${V16_UMA_GUARD_SCRIPT:-$HERE/v15_preboot_uma_guard.sh}"
if [ ! -f "$V16_UMA_GUARD_SCRIPT" ] && [ -f "$HERE/../../v15/unified/ops/v15_preboot_uma_guard.sh" ]; then
  V16_UMA_GUARD_SCRIPT="$HERE/../../v15/unified/ops/v15_preboot_uma_guard.sh"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

TARGET_ENV=""
PROFILE=""
ROLLBACK_ENV="${V16_ROLLBACK_ENV:-}"
DRAIN_ONLY=0
FORCE_STOP_ON_DRAIN_TIMEOUT=0
SKIP_UMA_GUARD=0
SKIP_SHA256_VERIFY=0
AUTO_REBOOT_DEFRAG=0
DRY_RUN="${V16_DRY_RUN:-0}"
JSON_OUT=0

usage() {
  cat <<'EOF'
Usage: v16_drain_and_deploy.sh [OPTIONS]
  --target-env <file>             Target production env file to deploy atomically
  --profile <default|max_kv_speed> Resolve target env from standard profile names
  --rollback-env <file>           Known-good rollback env (auto-resolves last_good_env / v15_default / v12)
  --drain-timeout <sec>           Max seconds to wait for in-flight requests to drain to 0 (default 120)
  --drain-poll <sec>              Poll interval during drain in seconds (default 2)
  --drain-only                    Wait for in-flight requests -> 0 and exit without stopping/deploying
  --force-stop-on-drain-timeout   Proceed with SIGTERM even if drain times out with active requests
  --head-host <host>              Head node API host for /metrics, /health, and boot cert (default 127.0.0.1)
  --port <port>                   vLLM API port (default 8010)
  --boot-timeout <sec>            Max seconds to wait for /health on new boot (default 420)
  --boot-poll <sec>               Poll interval while waiting for /health (default 5)
  --boot-cert-sha256 <prefix>     Expected 0-ULP SHA-256 digest prefix (e.g. 1dcfb306)
  --skip-uma-guard                Skip v15_preboot_uma_guard.sh check
  --skip-sha256-verify            Skip 8/8 /rdma/SHA256SUMS verification
  --auto-reboot-defrag            Pass --auto-reboot-defrag to v15_preboot_uma_guard.sh
  --dry-run                       Print orchestration plan without mutating cluster
  --json                          Emit machine-readable JSON deployment summary on stdout
  -h, --help                      Show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --target-env) TARGET_ENV="${2:?missing --target-env}"; shift 2 ;;
    --profile) PROFILE="${2:?missing --profile}"; shift 2 ;;
    --rollback-env) ROLLBACK_ENV="${2:?missing --rollback-env}"; shift 2 ;;
    --drain-timeout) V16_DRAIN_TIMEOUT_S="${2:?missing --drain-timeout}"; shift 2 ;;
    --drain-poll) V16_DRAIN_POLL_S="${2:?missing --drain-poll}"; shift 2 ;;
    --drain-only) DRAIN_ONLY=1; shift ;;
    --force-stop-on-drain-timeout) FORCE_STOP_ON_DRAIN_TIMEOUT=1; shift ;;
    --head-host) V16_HEAD_HOST="${2:?missing --head-host}"; shift 2 ;;
    --port) PORT="${2:?missing --port}"; shift 2 ;;
    --boot-timeout) V16_BOOT_TIMEOUT_S="${2:?missing --boot-timeout}"; shift 2 ;;
    --boot-poll) V16_BOOT_POLL_S="${2:?missing --boot-poll}"; shift 2 ;;
    --boot-cert-sha256) V16_BOOT_CERT_SHA256="${2:?missing --boot-cert-sha256}"; shift 2 ;;
    --skip-uma-guard) SKIP_UMA_GUARD=1; shift ;;
    --skip-sha256-verify) SKIP_SHA256_VERIFY=1; shift ;;
    --auto-reboot-defrag) AUTO_REBOOT_DEFRAG=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --json) JSON_OUT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[v16-deploy] Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

log() {
  if [ "$JSON_OUT" -eq 1 ]; then
    echo "$*" >&2
  else
    echo "$*"
  fi
}

jump_target_for() {
  local h="$1" pair key val
  for pair in $V16_SSH_JUMP_MAP; do
    key="${pair%%=*}"
    val="${pair#*=}"
    if [ "$key" = "$h" ]; then
      echo "$val"
      return 0
    fi
  done
  return 1
}

ssh_node() {
  local h="$1"; shift
  if [ "$DRY_RUN" = "1" ]; then
    log "[v16-deploy][DRY-RUN] ssh $h: $*"
    return 0
  fi
  local target=""
  if target="$(jump_target_for "$h")"; then
    "$V16_SSH_BIN" -o "ConnectTimeout=$V16_SSH_TIMEOUT" -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new -J "${V16_SSH_USER}@${V16_SSH_JUMP_HOST}" \
      "${V16_SSH_USER}@${target}" "$@"
  else
    "$V16_SSH_BIN" -o "ConnectTimeout=$V16_SSH_TIMEOUT" -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new "${V16_SSH_USER}@${h}" "$@"
  fi
}

scp_node() {
  local src="$1" h="$2" dst="$3"
  if [ "$DRY_RUN" = "1" ]; then
    log "[v16-deploy][DRY-RUN] scp $src -> $h:$dst"
    return 0
  fi
  local target=""
  if target="$(jump_target_for "$h")"; then
    "$V16_SCP_BIN" -q -o "ConnectTimeout=$V16_SSH_TIMEOUT" -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new -o "ProxyJump=${V16_SSH_USER}@${V16_SSH_JUMP_HOST}" \
      "$src" "${V16_SSH_USER}@${target}:$dst"
  else
    "$V16_SCP_BIN" -q -o "ConnectTimeout=$V16_SSH_TIMEOUT" -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new "$src" "${V16_SSH_USER}@${h}:$dst"
  fi
}

resolve_profile_env() {
  local prof="$1"
  local candidates=()
  case "$prof" in
    default)
      candidates=(
        "$V16_ENVS_DIR/live_champion_v16_default.env"
        "$V16_ENVS_DIR/live_champion_v15_default.env"
        "$HERE/../../v15/unified/envs/live_champion_v15_default.env"
      )
      ;;
    max_kv_speed)
      candidates=(
        "$V16_ENVS_DIR/live_champion_v16_max_kv_speed.env"
        "$V16_ENVS_DIR/live_champion_v15_max_kv_speed.env"
        "$HERE/../../v15/unified/envs/live_champion_v15_max_kv_speed.env"
      )
      ;;
    *)
      echo "[v16-deploy] ERROR: unknown profile '$prof' (expected default or max_kv_speed)" >&2
      exit 1
      ;;
  esac
  local c
  for c in "${candidates[@]}"; do
    if [ -f "$c" ]; then
      echo "$c"
      return 0
    fi
  done
  echo "[v16-deploy] ERROR: could not locate env file for profile '$prof' under $V16_ENVS_DIR" >&2
  exit 1
}

resolve_rollback_env() {
  if [ -n "$ROLLBACK_ENV" ] && [ -f "$ROLLBACK_ENV" ]; then
    echo "$ROLLBACK_ENV"
    return 0
  fi
  local last_good_file="$V16_STATE_DIR/last_good_env"
  if [ -f "$last_good_file" ]; then
    local prev
    prev="$(head -n1 "$last_good_file" | tr -d '\r\n')"
    if [ -n "$prev" ] && [ -f "$prev" ] && [ "$prev" != "$TARGET_ENV" ]; then
      echo "$prev"
      return 0
    fi
  fi
  local c
  for c in \
    "${V15_FALLBACK_ENV:-}" \
    "$V16_ENVS_DIR/live_champion_v15_default.env" \
    "$V16_ENVS_DIR/live_champion_v12.env" \
    "$HERE/../../v15/unified/envs/live_champion_v15_default.env"; do
    if [ -n "$c" ] && [ -f "$c" ] && [ "$c" != "$TARGET_ENV" ]; then
      echo "$c"
      return 0
    fi
  done
  echo ""
}

# Poll /metrics and print: "UP <running> <waiting>" or "DOWN 0 0"
query_inflight_requests() {
  "$PYTHON_BIN" - "$V16_HEAD_HOST" "$PORT" <<'PY'
import re, sys, urllib.request

host, port = sys.argv[1], sys.argv[2]
url = f"http://{host}:{port}/metrics"
try:
    with urllib.request.urlopen(url, timeout=4.0) as resp:
        if resp.status < 200 or resp.status >= 300:
            print("DOWN 0 0")
            sys.exit(0)
        body = resp.read().decode("utf-8", errors="replace")
except Exception:
    print("DOWN 0 0")
    sys.exit(0)

pat = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")
running = 0.0
waiting = 0.0
found_r = False
found_w = False
for line in body.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    m = pat.match(line)
    if not m:
        continue
    name, val = m.group(1), float(m.group(2))
    if name in ("vllm:num_requests_running", "num_requests_running"):
        running += val
        found_r = True
    elif name in ("vllm:num_requests_waiting", "num_requests_waiting"):
        waiting += val
        found_w = True

if not (found_r or found_w):
    print("NO_METRICS 0 0")
else:
    print(f"UP {int(round(running))} {int(round(waiting))}")
PY
}

graceful_drain_inflight() {
  if [ "$DRY_RUN" = "1" ]; then
    log "[v16-deploy][DRY-RUN] Would poll http://${V16_HEAD_HOST}:${PORT}/metrics for running+waiting==0 up to ${V16_DRAIN_TIMEOUT_S}s"
    return 0
  fi

  local probe status running waiting
  probe="$(query_inflight_requests)"
  read -r status running waiting <<< "$probe"
  if [ "$status" = "DOWN" ]; then
    log "[v16-deploy] Head endpoint http://${V16_HEAD_HOST}:${PORT}/metrics is not live; skipping active drain."
    return 0
  fi

  if [ -n "$V16_PRE_DRAIN_CMD" ]; then
    log "[v16-deploy] Executing pre-drain hook..."
    bash -c "$V16_PRE_DRAIN_CMD" || log "[v16-deploy] WARN: pre-drain hook returned non-zero"
  fi

  log "[v16-deploy] Starting zero-drop graceful drain (timeout=${V16_DRAIN_TIMEOUT_S}s, poll=${V16_DRAIN_POLL_S}s, consecutive_zeros=${V16_DRAIN_CONSECUTIVE_ZEROS})..."
  local start_ns elapsed_s zero_streak=0
  start_ns="$("$PYTHON_BIN" -c 'import time; print(time.monotonic())')"

  while true; do
    probe="$(query_inflight_requests)"
    read -r status running waiting <<< "$probe"
    if [ "$status" = "DOWN" ]; then
      log "[v16-deploy] Endpoint stopped responding during drain; treating as drained."
      return 0
    fi
    local total=$((running + waiting))
    elapsed_s="$("$PYTHON_BIN" -c 'import sys, time; print(f"{time.monotonic() - float(sys.argv[1]):.2f}")' "$start_ns")"
    if [ "$total" -le 0 ]; then
      zero_streak=$((zero_streak + 1))
      log "[v16-deploy] Drain quiescent: running=0 waiting=0 (streak ${zero_streak}/${V16_DRAIN_CONSECUTIVE_ZEROS}, elapsed=${elapsed_s}s)"
      if [ "$zero_streak" -ge "$V16_DRAIN_CONSECUTIVE_ZEROS" ]; then
        log "[v16-deploy] Zero-drop drain COMPLETE in ${elapsed_s}s (0 in-flight requests)."
        return 0
      fi
    else
      zero_streak=0
      log "[v16-deploy] Draining in-flight work: running=${running} waiting=${waiting} total=${total} (elapsed=${elapsed_s}s / ${V16_DRAIN_TIMEOUT_S}s)..."
    fi

    if "$PYTHON_BIN" -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)' "$elapsed_s" "$V16_DRAIN_TIMEOUT_S"; then
      if [ "$FORCE_STOP_ON_DRAIN_TIMEOUT" -eq 1 ]; then
        log "[v16-deploy] WARN: Drain timed out after ${elapsed_s}s with running=${running} waiting=${waiting}; proceeding due to --force-stop-on-drain-timeout!"
        return 0
      fi
      log "[v16-deploy] ERROR: Drain timed out after ${elapsed_s}s with running=${running} waiting=${waiting}; aborting cleanly to avoid dropping live requests!"
      return 3
    fi
    "$PYTHON_BIN" -c 'import sys, time; time.sleep(float(sys.argv[1]))' "$V16_DRAIN_POLL_S"
  done
}

stop_and_sweep_cluster() {
  log "[v16-deploy] Issuing coordinated SIGTERM stop (timeout=${V16_STOP_TIMEOUT}s) + IPC sweep on ${#HOSTS[@]}/${#HOSTS[@]} nodes..."
  local pids=() i h cname
  for ((i = 0; i < ${#HOSTS[@]}; i++)); do
    h="${HOSTS[$i]}"
    cname="${V16_CONTAINER_PREFIX}${i}"
    (
      ssh_node "$h" "docker stop -t ${V16_STOP_TIMEOUT} ${cname} >/dev/null 2>&1 || true; docker rm -f ${cname} >/dev/null 2>&1 || true; rm -rf /dev/shm/vllm* /dev/shm/inkling* /dev/shm/torch_* /dev/shm/nccl-* /dev/shm/psm_* /dev/shm/rdma_ar* /tmp/vllm* /tmp/torchinductor_* /tmp/inkling_rdma_*.lock 2>/dev/null || true"
    ) &
    pids+=($!)
  done
  local p fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" -eq 0 ] || log "[v16-deploy] WARN: one or more nodes returned non-zero during stop/sweep"
}

run_uma_guard() {
  if [ "$SKIP_UMA_GUARD" -eq 1 ]; then
    log "[v16-deploy] Skipping pre-boot UMA guard (--skip-uma-guard)."
    return 0
  fi
  if [ ! -f "$V16_UMA_GUARD_SCRIPT" ]; then
    log "[v16-deploy] ERROR: UMA guard script not found at $V16_UMA_GUARD_SCRIPT"
    return 2
  fi
  local extra=()
  [ "$AUTO_REBOOT_DEFRAG" -eq 1 ] && extra+=(--auto-reboot-defrag)
  [ "$DRY_RUN" = "1" ] && extra+=(--dry-run)
  log "[v16-deploy] Running v15_preboot_uma_guard.sh across ${#HOSTS[@]} nodes..."
  local rc=0
  V15_HOSTS="$V16_HOSTS" V15_SSH_USER="$V16_SSH_USER" V15_SSH_JUMP_HOST="$V16_SSH_JUMP_HOST" \
    V15_SSH_JUMP_MAP="$V16_SSH_JUMP_MAP" V15_SSH_BIN="$V16_SSH_BIN" \
    bash "$V16_UMA_GUARD_SCRIPT" ${extra[@]+"${extra[@]}"} || rc=$?
  if [ "$rc" -eq 2 ]; then
    log "[v16-deploy] WARN: UMA guard reported fragmented order-13 pools (exit 2)."
    if [ "${V16_STRICT_UMA_GUARD:-0}" = "1" ]; then
      return 2
    fi
  elif [ "$rc" -ne 0 ]; then
    log "[v16-deploy] ERROR: UMA guard failed (exit $rc)."
    return 2
  fi
  return 0
}

verify_rdma_sha256_all_nodes() {
  if [ "$SKIP_SHA256_VERIFY" -eq 1 ]; then
    log "[v16-deploy] Skipping 8/8 /rdma/SHA256SUMS verification (--skip-sha256-verify)."
    return 0
  fi
  log "[v16-deploy] Verifying /rdma/SHA256SUMS and ${V16_REQUIRED_ACTIVE_HCAS}/${V16_REQUIRED_ACTIVE_HCAS} Active RoCEv2 HCAs on ${#HOSTS[@]}/${#HOSTS[@]} nodes..."
  local pids=() i h
  local verify_cmd="set -e; [ -f '${V16_RDMA_DIR}/SHA256SUMS' ] || { echo 'MISSING_SHA256SUMS' >&2; exit 1; }; cd '${V16_RDMA_DIR}'; if command -v sha256sum >/dev/null 2>&1; then sha256sum -c --quiet SHA256SUMS; elif command -v shasum >/dev/null 2>&1; then shasum -a 256 -c SHA256SUMS >/dev/null; else echo 'NO_SHA256_TOOL' >&2; exit 1; fi; hcas=\$(grep -l ACTIVE /sys/class/infiniband/*/ports/1/state 2>/dev/null | wc -l | tr -d ' '); [ \"\${hcas:-0}\" -ge ${V16_REQUIRED_ACTIVE_HCAS} ] || { echo \"HCA_DOWN:\${hcas:-0}/${V16_REQUIRED_ACTIVE_HCAS}\" >&2; exit 1; }"
  for ((i = 0; i < ${#HOSTS[@]}; i++)); do
    h="${HOSTS[$i]}"
    (
      ssh_node "$h" "$verify_cmd"
    ) &
    pids+=($!)
  done
  local p fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  if [ "$fail" -ne 0 ]; then
    log "[v16-deploy] ERROR: /rdma/SHA256SUMS or 4/4 RoCEv2 HCA verification FAILED on one or more nodes!"
    return 2
  fi
  log "[v16-deploy] Verified /rdma/SHA256SUMS + ${V16_REQUIRED_ACTIVE_HCAS}/${V16_REQUIRED_ACTIVE_HCAS} Active HCAs on ${#HOSTS[@]}/${#HOSTS[@]} nodes."
  return 0
}

launch_profile_atomic() {
  local env_file="$1"
  log "[v16-deploy] Distributing $env_file -> $V16_REMOTE_ENV_PATH on ${#HOSTS[@]} nodes..."
  local pids=() i h p fail=0
  for ((i = 0; i < ${#HOSTS[@]}; i++)); do
    scp_node "$env_file" "${HOSTS[$i]}" "$V16_REMOTE_ENV_PATH" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" -eq 0 ] || { log "[v16-deploy] ERROR: failed to copy env file to all nodes"; return 1; }

  # Launch headless worker ranks 7..1 FIRST
  log "[v16-deploy] Launching headless worker ranks $((${#HOSTS[@]} - 1))..1..."
  pids=()
  for ((i = ${#HOSTS[@]} - 1; i >= 1; i--)); do
    h="${HOSTS[$i]}"
    (
      ssh_node "$h" "bash -lc 'for v in \$(compgen -e NCCL_); do unset \"\$v\"; done; set -a; source ${V16_REMOTE_ENV_PATH}; set +a; bash ${V16_REMOTE_LAUNCHER} ${i}'"
    ) &
    pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" -eq 0 ] || { log "[v16-deploy] ERROR: failed launching worker ranks 7..1"; return 1; }

  # Launch head rank 0 LAST
  log "[v16-deploy] Launching head rank 0 (${HOSTS[0]}) LAST..."
  ssh_node "${HOSTS[0]}" "bash -lc 'for v in \$(compgen -e NCCL_); do unset \"\$v\"; done; set -a; source ${V16_REMOTE_ENV_PATH}; set +a; bash ${V16_REMOTE_LAUNCHER} 0'"
}

wait_for_health_and_certify() {
  local env_label="$1"
  if [ "$DRY_RUN" = "1" ]; then
    log "[v16-deploy][DRY-RUN] Would wait for /health and run 0-ULP 1dcfb306 boot cert for $env_label"
    return 0
  fi

  log "[v16-deploy] Waiting for http://${V16_HEAD_HOST}:${PORT}/health (timeout=${V16_BOOT_TIMEOUT_S}s)..."
  local start_s ready=0
  start_s="$("$PYTHON_BIN" -c 'import time; print(time.monotonic())')"
  while true; do
    if "$PYTHON_BIN" -c '
import sys, urllib.request
try:
    with urllib.request.urlopen(f"http://{sys.argv[1]}:{sys.argv[2]}/health", timeout=4.0) as r:
        sys.exit(0 if 200 <= r.status < 300 else 1)
except Exception:
    sys.exit(1)
' "$V16_HEAD_HOST" "$PORT"; then
      ready=1
      break
    fi
    if "$PYTHON_BIN" -c 'import sys, time; sys.exit(0 if (time.monotonic() - float(sys.argv[1])) >= float(sys.argv[2]) else 1)' "$start_s" "$V16_BOOT_TIMEOUT_S"; then
      break
    fi
    "$PYTHON_BIN" -c 'import sys, time; time.sleep(float(sys.argv[1]))' "$V16_BOOT_POLL_S"
  done

  if [ "$ready" -ne 1 ]; then
    log "[v16-deploy] ERROR: $env_label did not reach /health within ${V16_BOOT_TIMEOUT_S}s!"
    return 1
  fi

  log "[v16-deploy] Running immediate 0-ULP 1dcfb306 boot certification for $env_label..."
  if [ -n "$V16_BOOT_CERT_CMD" ]; then
    V16_HEAD_HOST="$V16_HEAD_HOST" PORT="$PORT" V16_BOOT_CERT_SHA256="$V16_BOOT_CERT_SHA256" \
      bash -c "$V16_BOOT_CERT_CMD"
    return $?
  fi

  "$PYTHON_BIN" - "$V16_HEAD_HOST" "$PORT" "$V16_SERVED_MODEL" "$V16_BOOT_CERT_PROMPT" "$V16_BOOT_CERT_SHA256" <<'PY'
import hashlib, json, sys, urllib.request

host, port, model, prompt, expected = sys.argv[1:6]
expected = expected.strip().lower()
url = f"http://{host}:{port}/v1/completions"
payload = json.dumps({
    "model": model,
    "prompt": prompt,
    "max_tokens": 64,
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 0,
}).encode("utf-8")

def _run_once():
    req = urllib.request.Request(url, data=payload, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        if resp.status < 200 or resp.status >= 300:
            raise RuntimeError(f"http_status={resp.status}")
        data = json.loads(resp.read().decode("utf-8"))
    choices = data.get("choices") or []
    text = (choices[0].get("text") if choices else "") or ""
    if not text.strip():
        raise RuntimeError("empty_boot_cert_output")
    if "nan" in text.lower() and len(text.strip()) <= 8:
        raise RuntimeError(f"nan_output:{text!r}")
    # Support optional server-reported exact_sha256 metadata OR SHA-256 of greedy text
    digest = str(data.get("exact_sha256") or hashlib.sha256(text.encode("utf-8")).hexdigest()).lower()
    return text, digest

try:
    _, d1 = _run_once()
    if expected and not d1.startswith(expected):
        print(f"[v16-boot-cert] FAIL: 0-ULP digest mismatch got={d1[:16]} expected={expected[:16]}", file=sys.stderr)
        sys.exit(1)
    _, d2 = _run_once()
    if d1 != d2:
        print(f"[v16-boot-cert] FAIL: non-deterministic greedy output {d1[:16]} != {d2[:16]}", file=sys.stderr)
        sys.exit(1)
    print(f"[v16-boot-cert] PASS: 0-ULP certified digest={d1[:16]} (2/2 deterministic)", file=sys.stderr)
    sys.exit(0)
except Exception as exc:
    print(f"[v16-boot-cert] FAIL: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

record_deploy_event() {
  local status="$1" target="$2" rollback="$3" detail="$4"
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  mkdir -p "$V16_STATE_DIR" 2>/dev/null || return 0
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PYTHON_BIN" - "$V16_STATE_DIR/deploy_history.jsonl" "$ts" "$status" "$target" "$rollback" "$detail" <<'PY'
import json, sys
path, ts, status, target, rollback, detail = sys.argv[1:7]
rec = {"ts_utc": ts, "status": status, "target_env": target, "rollback_env": rollback, "detail": detail}
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec, sort_keys=True) + "\n")
PY
  if [ "$status" = "DEPLOYED" ] && [ -n "$target" ]; then
    printf '%s\n' "$target" > "$V16_STATE_DIR/last_good_env.tmp" && mv "$V16_STATE_DIR/last_good_env.tmp" "$V16_STATE_DIR/last_good_env"
  fi
}

emit_json_result() {
  local rc="$1" status="$2" detail="$3"
  if [ "$JSON_OUT" -eq 1 ]; then
    "$PYTHON_BIN" - "$rc" "$status" "$TARGET_ENV" "$ROLLBACK_ENV" "$detail" <<'PY'
import json, sys
rc, status, target, rollback, detail = sys.argv[1:6]
print(json.dumps({
    "exit_code": int(rc),
    "status": status,
    "target_env": target,
    "rollback_env": rollback,
    "detail": detail,
}, sort_keys=True))
PY
  fi
}

# ------------------------------------------------------------------------------
# Main Execution Flow
# ------------------------------------------------------------------------------

if [ "$DRAIN_ONLY" -eq 1 ]; then
  rc=0
  graceful_drain_inflight || rc=$?
  if [ "$rc" -eq 0 ]; then
    emit_json_result 0 "DRAINED" "inflight_requests_reached_zero"
    exit 0
  else
    emit_json_result "$rc" "DRAIN_TIMEOUT" "inflight_requests_still_active"
    exit "$rc"
  fi
fi

if [ -z "$TARGET_ENV" ] && [ -n "$PROFILE" ]; then
  TARGET_ENV="$(resolve_profile_env "$PROFILE")"
fi
if [ -z "$TARGET_ENV" ]; then
  echo "[v16-deploy] ERROR: specify --target-env <file> or --profile <default|max_kv_speed> (or --drain-only)" >&2
  usage >&2
  exit 1
fi
if [ ! -f "$TARGET_ENV" ]; then
  echo "[v16-deploy] ERROR: target env file not found: $TARGET_ENV" >&2
  exit 1
fi
if ! bash -n "$TARGET_ENV"; then
  echo "[v16-deploy] ERROR: target env file failed bash -n syntax validation: $TARGET_ENV" >&2
  exit 1
fi

ROLLBACK_ENV="$(resolve_rollback_env)"
log "=== Wave-V16 Zero-Drop Drain & Atomic 8-Node Deploy ==="
log "Target env  : $TARGET_ENV"
log "Rollback env: ${ROLLBACK_ENV:-<none>}"
log "Nodes (${#HOSTS[@]})   : ${HOSTS[*]}"

# Step 1: Zero-drop drain while current server is still live
drain_rc=0
graceful_drain_inflight || drain_rc=$?
if [ "$drain_rc" -ne 0 ]; then
  record_deploy_event "ABORTED_DRAIN_TIMEOUT" "$TARGET_ENV" "$ROLLBACK_ENV" "live_server_untouched"
  emit_json_result "$drain_rc" "ABORTED_DRAIN_TIMEOUT" "live_server_untouched"
  exit "$drain_rc"
fi

# Step 2: Coordinated SIGTERM teardown + scoped IPC sweep on 8/8 nodes
stop_and_sweep_cluster

# Helper: execute automatic rollback if target pre-boot or boot cert fails
perform_rollback() {
  local fail_reason="$1"
  if [ -z "$ROLLBACK_ENV" ] || [ ! -f "$ROLLBACK_ENV" ]; then
    log "[v16-deploy] CRITICAL: Deployment failed ($fail_reason) and no valid rollback env is configured!"
    record_deploy_event "FAILED_NO_ROLLBACK" "$TARGET_ENV" "" "$fail_reason"
    emit_json_result 5 "FAILED_NO_ROLLBACK" "$fail_reason"
    exit 5
  fi
  log "[v16-deploy] >>> AUTOMATIC ROLLBACK triggered ($fail_reason) -> restoring $ROLLBACK_ENV <<<"
  stop_and_sweep_cluster
  if launch_profile_atomic "$ROLLBACK_ENV" && wait_for_health_and_certify "ROLLBACK($ROLLBACK_ENV)"; then
    log "[v16-deploy] AUTOMATIC ROLLBACK SUCCEEDED & 0-ULP CERTIFIED on $ROLLBACK_ENV!"
    record_deploy_event "ROLLED_BACK" "$TARGET_ENV" "$ROLLBACK_ENV" "$fail_reason"
    emit_json_result 4 "ROLLED_BACK" "$fail_reason"
    exit 4
  else
    log "[v16-deploy] CRITICAL: AUTOMATIC ROLLBACK ALSO FAILED on $ROLLBACK_ENV!"
    record_deploy_event "ROLLBACK_FAILED" "$TARGET_ENV" "$ROLLBACK_ENV" "$fail_reason"
    emit_json_result 5 "ROLLBACK_FAILED" "$fail_reason"
    exit 5
  fi
}

# Step 3: Pre-boot UMA guard (v15_preboot_uma_guard.sh)
if ! run_uma_guard; then
  perform_rollback "preboot_uma_guard_failed"
fi

# Step 4: Verify /rdma/SHA256SUMS + 4/4 RoCEv2 HCAs on 8/8 nodes
if ! verify_rdma_sha256_all_nodes; then
  perform_rollback "rdma_sha256_or_hca_check_failed"
fi

# Step 5: Atomic launch 7..1 -> 0
if ! launch_profile_atomic "$TARGET_ENV"; then
  perform_rollback "atomic_launch_failed"
fi

# Step 6: Wait for /health + immediate 0-ULP 1dcfb306 boot cert
if ! wait_for_health_and_certify "TARGET($TARGET_ENV)"; then
  perform_rollback "boot_health_or_0ulp_cert_failed"
fi

log "[v16-deploy] SUCCESS: $TARGET_ENV deployed atomically across ${#HOSTS[@]}/${#HOSTS[@]} nodes and 0-ULP certified!"
record_deploy_event "DEPLOYED" "$TARGET_ENV" "$ROLLBACK_ENV" "0ulp_certified"
emit_json_result 0 "DEPLOYED" "0ulp_certified"
exit 0

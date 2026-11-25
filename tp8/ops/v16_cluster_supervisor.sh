#!/usr/bin/env bash
# Wave-V16 Permanent 24/7 8-Node Cluster Supervisor & Operations CLI
# Upgraded from v15_cluster_supervisor.sh with:
#   - Zero-drop graceful drain (`drain`) waiting for running+waiting == 0 before stop
#   - Atomic zero-drop profile deploy & auto-rollback (`deploy <env|--profile default|max_kv_speed>`)
#   - Integrated Prometheus telemetry exporter (:9108) health & lifecycle (`exporter`)
#   - Enhanced `status` showing 8-node UMA/RoCEv2/GPU + live vLLM /metrics + exporter /healthz
#   - Full backward compatibility with all V15 commands (preflight, start, stop, restart, logs, bundle, supervise, reset-breaker)
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

V15_HOSTS_DEFAULT="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
V15_HOSTS="${V16_HOSTS:-${V15_HOSTS:-$V15_HOSTS_DEFAULT}}"
# shellcheck disable=SC2206
HOSTS=($V15_HOSTS)
V15_SSH_USER="${V16_SSH_USER:-${V15_SSH_USER:-${USER:-jeweled}}}"
V15_SSH_JUMP_HOST="${V16_SSH_JUMP_HOST:-${V15_SSH_JUMP_HOST:-spark-one}}"
V15_SSH_JUMP_MAP="${V16_SSH_JUMP_MAP:-${V15_SSH_JUMP_MAP:-spark-five=10.77.0.51 spark-six=10.77.0.61}}"
V15_SSH_BIN="${V16_SSH_BIN:-${V15_SSH_BIN:-ssh}}"
V15_SCP_BIN="${V16_SCP_BIN:-${V15_SCP_BIN:-scp}}"
V15_SSH_TIMEOUT="${V16_SSH_TIMEOUT:-${V15_SSH_TIMEOUT:-10}}"
V15_CONTAINER_PREFIX="${V16_CONTAINER_PREFIX:-${V15_CONTAINER_PREFIX:-inkling-tp8-r}}"
V15_REMOTE_LAUNCHER="${V16_REMOTE_LAUNCHER:-${V15_REMOTE_LAUNCHER:-/home/${V15_SSH_USER}/inkling-tp8/inkling-tp8-v15.sh}}"
V15_REMOTE_ENV_PATH="${V16_REMOTE_ENV_PATH:-${V15_REMOTE_ENV_PATH:-/tmp/inkling_v16_active.env}}"
V15_PRIMARY_ENV="${V16_PRIMARY_ENV:-${V15_PRIMARY_ENV:-$HERE/live_champion_v15_default.env}}"
V15_FALLBACK_ENV="${V16_FALLBACK_ENV:-${V15_FALLBACK_ENV:-$HERE/live_champion_v12.env}}"
V15_HEAD_HOST="${V16_HEAD_HOST:-${V15_HEAD_HOST:-127.0.0.1}}"
PORT="${PORT:-8010}"
V16_EXPORTER_HOST="${V16_EXPORTER_HOST:-127.0.0.1}"
V16_EXPORTER_PORT="${V16_EXPORTER_PORT:-9108}"
V15_STOP_TIMEOUT="${V16_STOP_TIMEOUT:-${V15_STOP_TIMEOUT:-30}}"
V16_DRAIN_ON_STOP="${V16_DRAIN_ON_STOP:-1}"
V16_DRAIN_TIMEOUT_S="${V16_DRAIN_TIMEOUT_S:-120}"
V15_BOOT_TIMEOUT_S="${V16_BOOT_TIMEOUT_S:-${V15_BOOT_TIMEOUT_S:-420}}"
V15_LOG_DIR="${V16_LOG_DIR:-${V15_LOG_DIR:-/var/log/inkling-v15}}"
V15_STATE_DIR="${V16_STATE_DIR:-${V15_STATE_DIR:-/var/lib/inkling-v15}}"
V15_STOP_ON_EXIT="${V15_STOP_ON_EXIT:-0}"
V15_DRY_RUN="${V16_DRY_RUN:-${V15_DRY_RUN:-0}}"
V16_MANAGE_EXPORTER="${V16_MANAGE_EXPORTER:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

UMA_GUARD="${V16_UMA_GUARD_SCRIPT:-$HERE/v15_preboot_uma_guard.sh}"
[ -f "$UMA_GUARD" ] || UMA_GUARD="$HERE/../../v15/unified/ops/v15_preboot_uma_guard.sh"
WATCHDOG_PY="${V16_WATCHDOG_PY:-$HERE/v15_health_watchdog.py}"
[ -f "$WATCHDOG_PY" ] || WATCHDOG_PY="$HERE/../../v15/unified/ops/v15_health_watchdog.py"
DRAIN_DEPLOY_SH="$HERE/v16_drain_and_deploy.sh"
EXPORTER_PY="$HERE/v16_metrics_exporter.py"

jump_target_for() {
  local h="$1" pair key val
  for pair in $V15_SSH_JUMP_MAP; do
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
  if [ "$V15_DRY_RUN" = "1" ]; then
    echo "[v16-supervisor][DRY-RUN] ssh $h: $*"
    return 0
  fi
  local target=""
  if target="$(jump_target_for "$h")"; then
    "$V15_SSH_BIN" -o "ConnectTimeout=$V15_SSH_TIMEOUT" -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new -J "${V15_SSH_USER}@${V15_SSH_JUMP_HOST}" \
      "${V15_SSH_USER}@${target}" "$@"
  else
    "$V15_SSH_BIN" -o "ConnectTimeout=$V15_SSH_TIMEOUT" -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new "${V15_SSH_USER}@${h}" "$@"
  fi
}

scp_node() {
  local src="$1" h="$2" dst="$3"
  if [ "$V15_DRY_RUN" = "1" ]; then
    echo "[v16-supervisor][DRY-RUN] scp $src -> $h:$dst"
    return 0
  fi
  local target=""
  if target="$(jump_target_for "$h")"; then
    "$V15_SCP_BIN" -q -o "ConnectTimeout=$V15_SSH_TIMEOUT" -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new -o "ProxyJump=${V15_SSH_USER}@${V15_SSH_JUMP_HOST}" \
      "$src" "${V15_SSH_USER}@${target}:$dst"
  else
    "$V15_SCP_BIN" -q -o "ConnectTimeout=$V15_SSH_TIMEOUT" -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new "$src" "${V15_SSH_USER}@${h}:$dst"
  fi
}

notify_systemd() {
  if [ -n "${NOTIFY_SOCKET:-}" ] && command -v systemd-notify >/dev/null 2>&1; then
    systemd-notify "$@" >/dev/null 2>&1 || true
  fi
}

cmd_drain() {
  echo "[v16-supervisor] Draining active requests on http://${V15_HEAD_HOST}:${PORT}/metrics..."
  V16_DRY_RUN="$V15_DRY_RUN" V16_HEAD_HOST="$V15_HEAD_HOST" PORT="$PORT" \
    bash "$DRAIN_DEPLOY_SH" --drain-only --drain-timeout "$V16_DRAIN_TIMEOUT_S" "$@"
}

cmd_deploy() {
  if [ "$#" -lt 1 ]; then
    echo "[v16-supervisor] Usage: v16_cluster_supervisor.sh deploy <env_file|--profile default|max_kv_speed> [extra v16_drain_and_deploy opts]" >&2
    exit 1
  fi
  local first="$1"
  local args=()
  if [ "${first:0:1}" = "-" ]; then
    args=("$@")
  else
    shift
    args=(--target-env "$first" "$@")
  fi
  V16_HOSTS="$V15_HOSTS" V16_SSH_USER="$V15_SSH_USER" V16_SSH_JUMP_HOST="$V15_SSH_JUMP_HOST" \
    V16_SSH_JUMP_MAP="$V15_SSH_JUMP_MAP" V16_SSH_BIN="$V15_SSH_BIN" V16_SCP_BIN="$V15_SCP_BIN" \
    V16_HEAD_HOST="$V15_HEAD_HOST" PORT="$PORT" V16_STATE_DIR="$V15_STATE_DIR" \
    V16_DRY_RUN="$V15_DRY_RUN" V16_UMA_GUARD_SCRIPT="$UMA_GUARD" \
    bash "$DRAIN_DEPLOY_SH" "${args[@]}"
}

cmd_stop() {
  local force_stop=0
  local skip_drain=0
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --force|--no-drain) skip_drain=1; force_stop=1; shift ;;
      --drain-timeout) V16_DRAIN_TIMEOUT_S="${2:?missing --drain-timeout}"; shift 2 ;;
      *) shift ;;
    esac
  done

  if [ "$V16_DRAIN_ON_STOP" = "1" ] && [ "$skip_drain" -eq 0 ] && [ "$V15_DRY_RUN" != "1" ]; then
    local d_rc=0
    V16_HEAD_HOST="$V15_HEAD_HOST" PORT="$PORT" \
      bash "$DRAIN_DEPLOY_SH" --drain-only --drain-timeout "$V16_DRAIN_TIMEOUT_S" || d_rc=$?
    if [ "$d_rc" -eq 3 ] && [ "$force_stop" -eq 0 ]; then
      echo "[v16-supervisor] ERROR: Drain timed out with active requests; refusing abrupt stop (re-run with 'stop --force' to override)." >&2
      return 3
    fi
  fi

  echo "[v16-supervisor] Gracefully stopping ${#HOSTS[@]} ranks (stop_timeout=${V15_STOP_TIMEOUT}s) + sweeping IPC..."
  local pids=() i h cname
  for ((i = 0; i < ${#HOSTS[@]}; i++)); do
    h="${HOSTS[$i]}"
    cname="${V15_CONTAINER_PREFIX}${i}"
    (
      ssh_node "$h" "docker stop -t ${V15_STOP_TIMEOUT} ${cname} >/dev/null 2>&1 || true; docker rm -f ${cname} >/dev/null 2>&1 || true; rm -rf /dev/shm/vllm* /dev/shm/inkling* /dev/shm/torch_* /dev/shm/nccl-* /dev/shm/psm_* /dev/shm/rdma_ar* /tmp/vllm* /tmp/torchinductor_* /tmp/inkling_rdma_*.lock 2>/dev/null || true"
    ) &
    pids+=($!)
  done
  local p fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" -eq 0 ] || echo "[v16-supervisor] WARN: one or more nodes returned non-zero during stop" >&2
  echo "[v16-supervisor] Stopped ${#HOSTS[@]}/${#HOSTS[@]} ranks & swept stale IPC."
}

cmd_preflight() {
  local extra_args=()
  [ "$#" -gt 0 ] && extra_args=("$@")
  echo "[v16-supervisor] Running 8-node UMA buddy guard..."
  if [ "$V15_DRY_RUN" = "1" ]; then
    echo "[v16-supervisor][DRY-RUN] bash $UMA_GUARD --hosts \"$V15_HOSTS\" ${extra_args[*]:-}"
  else
    local rc=0
    V15_HOSTS="$V15_HOSTS" V15_SSH_USER="$V15_SSH_USER" V15_SSH_JUMP_HOST="$V15_SSH_JUMP_HOST" \
      V15_SSH_JUMP_MAP="$V15_SSH_JUMP_MAP" V15_SSH_BIN="$V15_SSH_BIN" \
      bash "$UMA_GUARD" ${extra_args[@]+"${extra_args[@]}"} || rc=$?
    if [ "$rc" -eq 2 ]; then
      echo "[v16-supervisor] WARN: UMA guard flagged fragmented order-13 pools (exit 2). Consider --auto-reboot-defrag!" >&2
      if [ "${V15_STRICT_UMA_GUARD:-0}" = "1" ]; then
        return 2
      fi
    elif [ "$rc" -ne 0 ]; then
      echo "[v16-supervisor] ERROR: UMA guard failed (exit $rc)" >&2
      return "$rc"
    fi
  fi

  echo "[v16-supervisor] Checking 4/4 RoCEv2 HCAs across ${#HOSTS[@]} nodes..."
  local pids=() i h
  for ((i = 0; i < ${#HOSTS[@]}; i++)); do
    h="${HOSTS[$i]}"
    (
      if [ "$V15_DRY_RUN" = "1" ]; then
        echo "[v16-supervisor][DRY-RUN] RoCE check on $h"
      else
        cnt="$(ssh_node "$h" "grep -l ACTIVE /sys/class/infiniband/*/ports/1/state 2>/dev/null | wc -l | tr -d ' '" || echo 0)"
        if [ "${cnt:-0}" -lt "${V15_REQUIRED_ACTIVE_HCAS:-4}" ]; then
          echo "[v16-supervisor] ERROR: $h has only ${cnt:-0}/${V15_REQUIRED_ACTIVE_HCAS:-4} Active RoCEv2 HCAs!" >&2
          exit 1
        fi
        echo "[v16-supervisor] RoCE OK $h: ${cnt}/${V15_REQUIRED_ACTIVE_HCAS:-4} Active"
      fi
    ) &
    pids+=($!)
  done
  local p fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" -eq 0 ] || return 1
  echo "[v16-supervisor] Preflight PASSED on ${#HOSTS[@]}/${#HOSTS[@]} nodes."
}

cmd_start() {
  local env_file="${1:-$V15_PRIMARY_ENV}"
  [ -f "$env_file" ] || { echo "[v16-supervisor] ERROR: env file not found: $env_file" >&2; exit 1; }
  shift || true
  cmd_deploy "$env_file" "$@"
}

cmd_exporter() {
  local sub="${1:-status}"
  shift || true
  case "$sub" in
    run|start)
      if [ "$V15_DRY_RUN" = "1" ]; then
        echo "[v16-supervisor][DRY-RUN] $PYTHON_BIN $EXPORTER_PY --port $V16_EXPORTER_PORT --vllm-host $V15_HEAD_HOST --vllm-port $PORT $*"
        return 0
      fi
      exec "$PYTHON_BIN" "$EXPORTER_PY" \
        --port "$V16_EXPORTER_PORT" \
        --hosts "$V15_HOSTS" \
        --vllm-host "$V15_HEAD_HOST" \
        --vllm-port "$PORT" \
        --watchdog-state "$V15_STATE_DIR/watchdog_state.json" \
        "$@"
      ;;
    status)
      if [ "$V15_DRY_RUN" = "1" ]; then
        echo "[v16-supervisor][DRY-RUN] Would query http://${V16_EXPORTER_HOST}:${V16_EXPORTER_PORT}/healthz"
        return 0
      fi
      if curl -fsS --max-time 3 "http://${V16_EXPORTER_HOST}:${V16_EXPORTER_PORT}/healthz"; then
        return 0
      else
        echo "[v16-supervisor] Exporter DOWN on http://${V16_EXPORTER_HOST}:${V16_EXPORTER_PORT}/healthz" >&2
        return 1
      fi
      ;;
    *)
      echo "[v16-supervisor] Usage: v16_cluster_supervisor.sh exporter <run|status> [args...]" >&2
      exit 1
      ;;
  esac
}

cmd_status() {
  echo "=== Wave-V16 8-Node Cluster & Telemetry Status (API :${PORT}, Exporter :${V16_EXPORTER_PORT}) ==="
  if [ "$V15_DRY_RUN" = "1" ]; then
    echo "[v16-supervisor][DRY-RUN] Would query ${#HOSTS[@]} nodes + :${PORT}/health + :${V16_EXPORTER_PORT}/healthz"
    return 0
  fi
  if curl -fsS --max-time 3 "http://${V15_HEAD_HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "API /health      : OK (http://${V15_HEAD_HOST}:${PORT}/health)"
  else
    echo "API /health      : DOWN (http://${V15_HEAD_HOST}:${PORT}/health)"
  fi
  if exp_h="$(curl -fsS --max-time 3 "http://${V16_EXPORTER_HOST}:${V16_EXPORTER_PORT}/healthz" 2>/dev/null)"; then
    echo "Exporter /healthz: OK ($exp_h)"
  else
    echo "Exporter /healthz: DOWN (http://${V16_EXPORTER_HOST}:${V16_EXPORTER_PORT}/healthz)"
  fi
  local pids=() i h cname
  for ((i = 0; i < ${#HOSTS[@]}; i++)); do
    h="${HOSTS[$i]}"
    cname="${V15_CONTAINER_PREFIX}${i}"
    (
      info="$(ssh_node "$h" "
        st=\$(docker ps -a --filter name=^${cname}\$ --format '{{.Status}}' 2>/dev/null | head -n1); [ -z \"\$st\" ] && st='missing'
        hcas=\$(grep -l ACTIVE /sys/class/infiniband/*/ports/1/state 2>/dev/null | wc -l | tr -d ' ')
        gpu=\$(nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')
        sw=\$(awk '/SwapTotal/{t=\$2}/SwapFree/{f=\$2} END{printf \"%.0fMiB\", (t>f?t-f:0)/1024}' /proc/meminfo 2>/dev/null)
        o13=\$(awk '/zone/{for(k=1;k<=NF;k++)if(\$k==\"zone\"){z=k;break}; if(NF>=z+15) s+=\$(z+15)} END{print s+0}' /proc/buddyinfo 2>/dev/null)
        echo \"r${i} (${h}): container=[\${st}] roce=\${hcas}/4 gpu=[\${gpu:-na}] swap=\${sw:-na} order13=\${o13:-na}\"
      " 2>/dev/null || echo "r${i} (${h}): UNREACHABLE")"
      echo "$info"
    ) &
    pids+=($!)
  done
  local p
  for p in "${pids[@]}"; do wait "$p"; done
}

cmd_logs() {
  local target="${1:-0}"
  local lines="${2:-100}"
  if [ "$target" = "all" ]; then
    local i
    for ((i = 0; i < ${#HOSTS[@]}; i++)); do
      echo "===== rank $i (${HOSTS[$i]}) ====="
      ssh_node "${HOSTS[$i]}" "docker logs --tail ${lines} ${V15_CONTAINER_PREFIX}${i} 2>&1" || true
    done
  else
    local h="${HOSTS[$target]}"
    ssh_node "$h" "docker logs --tail ${lines} ${V15_CONTAINER_PREFIX}${target} 2>&1"
  fi
}

cmd_bundle() {
  local reason="${1:-manual_capture}"
  "$PYTHON_BIN" "$WATCHDOG_PY" --capture-bundle "$reason" \
    --hosts "$V15_HOSTS" --log-dir "$V15_LOG_DIR" --state-dir "$V15_STATE_DIR"
}

cmd_supervise() {
  local wd_pid=""
  local exp_pid=""
  _on_term() {
    echo "[v16-supervisor] Caught shutdown signal; stopping watchdog & exporter..."
    [ -n "$wd_pid" ] && kill -TERM "$wd_pid" 2>/dev/null || true
    [ -n "$exp_pid" ] && kill -TERM "$exp_pid" 2>/dev/null || true
    [ -n "$wd_pid" ] && wait "$wd_pid" 2>/dev/null || true
    [ -n "$exp_pid" ] && wait "$exp_pid" 2>/dev/null || true
    if [ "$V15_STOP_ON_EXIT" = "1" ]; then
      cmd_stop || true
    fi
    exit 0
  }
  trap _on_term TERM INT

  notify_systemd --status="Checking cluster health before starting V16 watchdog..."
  if [ "$V15_DRY_RUN" != "1" ]; then
    if ! curl -fsS --max-time 5 "http://${V15_HEAD_HOST}:${PORT}/health" >/dev/null 2>&1; then
      echo "[v16-supervisor] Cluster not currently serving on port ${PORT}; initiating cold start..."
      if [ -f "$V15_PRIMARY_ENV" ]; then
        cmd_deploy "$V15_PRIMARY_ENV" || echo "[v16-supervisor] WARN: initial start failed; watchdog daemon will manage recovery/fallback." >&2
      else
        echo "[v16-supervisor] WARN: V15_PRIMARY_ENV=$V15_PRIMARY_ENV not found; delegating to watchdog daemon." >&2
      fi
    else
      echo "[v16-supervisor] Attaching to already-healthy cluster on port ${PORT}."
    fi
  fi

  if [ "$V16_MANAGE_EXPORTER" = "1" ]; then
    if [ "$V15_DRY_RUN" = "1" ]; then
      echo "[v16-supervisor][DRY-RUN] Would spawn background exporter on :${V16_EXPORTER_PORT}"
    else
      "$PYTHON_BIN" "$EXPORTER_PY" \
        --port "$V16_EXPORTER_PORT" \
        --hosts "$V15_HOSTS" \
        --vllm-host "$V15_HEAD_HOST" \
        --vllm-port "$PORT" \
        --watchdog-state "$V15_STATE_DIR/watchdog_state.json" &
      exp_pid=$!
    fi
  fi

  notify_systemd --ready --status="Supervising 8-node Inkling-2-Small TP8 V16 cluster"
  local wd_args=(
    "$WATCHDOG_PY" --daemon
    --hosts "$V15_HOSTS"
    --head-host "$V15_HEAD_HOST"
    --port "$PORT"
    --log-dir "$V15_LOG_DIR"
    --state-dir "$V15_STATE_DIR"
  )
  [ -n "$V15_PRIMARY_ENV" ] && wd_args+=(--primary-env "$V15_PRIMARY_ENV")
  [ -n "$V15_FALLBACK_ENV" ] && wd_args+=(--fallback-env "$V15_FALLBACK_ENV")

  if [ "$V15_DRY_RUN" = "1" ]; then
    echo "[v16-supervisor][DRY-RUN] Would exec: $PYTHON_BIN ${wd_args[*]}"
    return 0
  fi

  "$PYTHON_BIN" "${wd_args[@]}" &
  wd_pid=$!
  wait "$wd_pid"
}

cmd_reset_breaker() {
  "$PYTHON_BIN" "$WATCHDOG_PY" --reset-breaker --state-dir "$V15_STATE_DIR"
}

usage() {
  cat <<'EOF'
Usage: v16_cluster_supervisor.sh <command> [args...]
Commands:
  preflight [--auto-reboot-defrag]  Check 8-node UMA buddyinfo/swap & 4/4 RoCEv2 links
  drain [--drain-timeout 120]       Wait cleanly for running+waiting requests -> 0 (zero-drop)
  deploy <env|--profile ...> [opts] Zero-drop drain -> UMA guard -> 8/8 SHA256 -> atomic 7..0 -> 0-ULP cert + auto-rollback
  start [env_file]                  Zero-drop deploy & 0-ULP certified start
  stop [--force]                    Zero-drop drain -> graceful SIGTERM (30s) + scoped IPC sweep on 8/8
  restart [env_file]                Zero-drop drain + atomic deploy
  exporter <run|status>             Run or check Prometheus telemetry exporter (:9108)
  status                            8-node health + API + Prometheus exporter dashboard
  logs [rank|all] [lines]           Inspect container logs
  bundle [reason]                   Capture timestamped 8-node diagnostic crash bundle
  supervise                         Run foreground systemd supervisor daemon + active watchdog
  reset-breaker                     Clear circuit breaker & fallback lock after hardware repair
EOF
}

CMD="${1:-}"
[ -n "$CMD" ] || { usage >&2; exit 1; }
shift || true

case "$CMD" in
  preflight)     cmd_preflight "$@" ;;
  drain)         cmd_drain "$@" ;;
  deploy)        cmd_deploy "$@" ;;
  start)         cmd_start "$@" ;;
  stop)          cmd_stop "$@" ;;
  restart)       cmd_start "$@" ;;
  exporter)      cmd_exporter "$@" ;;
  status)        cmd_status "$@" ;;
  logs)          cmd_logs "$@" ;;
  bundle)        cmd_bundle "$@" ;;
  supervise)     cmd_supervise "$@" ;;
  reset-breaker) cmd_reset_breaker "$@" ;;
  -h|--help)     usage ;;
  *)             echo "[v16-supervisor] Unknown command: $CMD" >&2; usage >&2; exit 1 ;;
esac

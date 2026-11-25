#!/usr/bin/env bash
# Wave-V15 Permanent 24/7 8-Node Cluster Supervisor & Operations CLI
# Upgraded from v8_cluster.sh for unattended DGX Spark GB10 TP8 production serving.
#
# Subcommands:
#   preflight [--auto-reboot-defrag]  Run 8-node UMA buddy guard + 4/4 RoCEv2 + RDMA manifest check
#   start [env_file]                  Preflight -> clean teardown -> atomic 7..0 launch -> /health -> 0-ULP boot cert
#   stop                              Graceful SIGTERM drain (30s) + rm -f + scoped /dev/shm & /tmp/vllm* sweep
#   restart [env_file]                Clean stop + start
#   status                            8-node container, GPU, RoCEv2, swap, order-13 & /health dashboard
#   logs [rank|all] [lines]           Fetch container logs from rank 0..7 or all ranks
#   bundle [reason]                   Capture 8-node forensic crash bundle to /var/log/inkling-v15/crash-<ts>/
#   supervise                         Foreground systemd daemon: ensures healthy boot & runs v15_health_watchdog.py --daemon
#   reset-breaker                     Reset watchdog circuit breaker & V12 fallback state after hardware repair
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# Optional external cluster config (zero hardcoded secrets)
if [ -n "${V15_CLUSTER_CONF:-}" ] && [ -f "$V15_CLUSTER_CONF" ]; then
  # shellcheck disable=SC1090
  set -a; source "$V15_CLUSTER_CONF"; set +a
elif [ -f "/etc/inkling-v15/supervisor.env" ]; then
  # shellcheck disable=SC1091
  set -a; source "/etc/inkling-v15/supervisor.env"; set +a
fi

V15_HOSTS_DEFAULT="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
V15_HOSTS="${V15_HOSTS:-$V15_HOSTS_DEFAULT}"
# shellcheck disable=SC2206
HOSTS=($V15_HOSTS)
V15_SSH_USER="${V15_SSH_USER:-${USER:-jeweled}}"
V15_SSH_JUMP_HOST="${V15_SSH_JUMP_HOST:-spark-one}"
V15_SSH_JUMP_MAP="${V15_SSH_JUMP_MAP:-spark-five=10.77.0.51 spark-six=10.77.0.61}"
V15_SSH_BIN="${V15_SSH_BIN:-ssh}"
V15_SCP_BIN="${V15_SCP_BIN:-scp}"
V15_SSH_TIMEOUT="${V15_SSH_TIMEOUT:-10}"
V15_CONTAINER_PREFIX="${V15_CONTAINER_PREFIX:-inkling-tp8-r}"
V15_REMOTE_LAUNCHER="${V15_REMOTE_LAUNCHER:-/home/${V15_SSH_USER}/inkling-tp8/inkling-tp8-v15.sh}"
V15_REMOTE_ENV_PATH="${V15_REMOTE_ENV_PATH:-/tmp/inkling_v15_active.env}"
V15_PRIMARY_ENV="${V15_PRIMARY_ENV:-$HERE/v15_production.env}"
V15_FALLBACK_ENV="${V15_FALLBACK_ENV:-$HERE/live_champion_v12.env}"
V15_HEAD_HOST="${V15_HEAD_HOST:-127.0.0.1}"
PORT="${PORT:-8010}"
V15_STOP_TIMEOUT="${V15_STOP_TIMEOUT:-30}"
V15_BOOT_TIMEOUT_S="${V15_BOOT_TIMEOUT_S:-420}"
V15_LOG_DIR="${V15_LOG_DIR:-/var/log/inkling-v15}"
V15_STATE_DIR="${V15_STATE_DIR:-/var/lib/inkling-v15}"
V15_STOP_ON_EXIT="${V15_STOP_ON_EXIT:-0}"
V15_DRY_RUN="${V15_DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

UMA_GUARD="$HERE/v15_preboot_uma_guard.sh"
WATCHDOG_PY="$HERE/v15_health_watchdog.py"

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
    echo "[supervisor][DRY-RUN] ssh $h: $*"
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
    echo "[supervisor][DRY-RUN] scp $src -> $h:$dst"
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

cmd_stop() {
  echo "[supervisor] Gracefully stopping ${#HOSTS[@]} ranks (stop_timeout=${V15_STOP_TIMEOUT}s) + sweeping IPC..."
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
  [ "$fail" -eq 0 ] || echo "[supervisor] WARN: one or more nodes returned non-zero during stop" >&2
  echo "[supervisor] Stopped ${#HOSTS[@]}/${#HOSTS[@]} ranks & swept stale IPC."
}

cmd_preflight() {
  local extra_args=()
  [ "$#" -gt 0 ] && extra_args=("$@")
  echo "[supervisor] Running 8-node UMA buddy guard..."
  if [ "$V15_DRY_RUN" = "1" ]; then
    echo "[supervisor][DRY-RUN] bash $UMA_GUARD --hosts \"$V15_HOSTS\" ${extra_args[*]:-}"
  else
    local rc=0
    V15_HOSTS="$V15_HOSTS" V15_SSH_USER="$V15_SSH_USER" V15_SSH_JUMP_HOST="$V15_SSH_JUMP_HOST" \
      V15_SSH_JUMP_MAP="$V15_SSH_JUMP_MAP" V15_SSH_BIN="$V15_SSH_BIN" \
      bash "$UMA_GUARD" ${extra_args[@]+"${extra_args[@]}"} || rc=$?
    if [ "$rc" -eq 2 ]; then
      echo "[supervisor] WARN: UMA guard flagged fragmented order-13 pools (exit 2). Consider --auto-reboot-defrag!" >&2
      if [ "${V15_STRICT_UMA_GUARD:-0}" = "1" ]; then
        return 2
      fi
    elif [ "$rc" -ne 0 ]; then
      echo "[supervisor] ERROR: UMA guard failed (exit $rc)" >&2
      return "$rc"
    fi
  fi

  echo "[supervisor] Checking 4/4 RoCEv2 HCAs across ${#HOSTS[@]} nodes..."
  local pids=() i h
  for ((i = 0; i < ${#HOSTS[@]}; i++)); do
    h="${HOSTS[$i]}"
    (
      if [ "$V15_DRY_RUN" = "1" ]; then
        echo "[supervisor][DRY-RUN] RoCE check on $h"
      else
        cnt="$(ssh_node "$h" "grep -l ACTIVE /sys/class/infiniband/*/ports/1/state 2>/dev/null | wc -l | tr -d ' '" || echo 0)"
        if [ "${cnt:-0}" -lt "${V15_REQUIRED_ACTIVE_HCAS:-4}" ]; then
          echo "[supervisor] ERROR: $h has only ${cnt:-0}/${V15_REQUIRED_ACTIVE_HCAS:-4} Active RoCEv2 HCAs!" >&2
          exit 1
        fi
        echo "[supervisor] RoCE OK $h: ${cnt}/${V15_REQUIRED_ACTIVE_HCAS:-4} Active"
      fi
    ) &
    pids+=($!)
  done
  local p fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" -eq 0 ] || return 1
  echo "[supervisor] Preflight PASSED on ${#HOSTS[@]}/${#HOSTS[@]} nodes."
}

cmd_start() {
  local env_file="${1:-$V15_PRIMARY_ENV}"
  [ -f "$env_file" ] || { echo "[supervisor] ERROR: env file not found: $env_file" >&2; exit 1; }

  # 1. Clean stop + IPC sweep + preflight
  cmd_stop
  if [ "${V15_SKIP_PREFLIGHT:-0}" != "1" ]; then
    cmd_preflight
  fi

  # 2. Sync env file to all 8 nodes in parallel
  echo "[supervisor] Syncing $env_file -> $V15_REMOTE_ENV_PATH on ${#HOSTS[@]} nodes..."
  local pids=() i h p fail=0
  for ((i = 0; i < ${#HOSTS[@]}; i++)); do
    scp_node "$env_file" "${HOSTS[$i]}" "$V15_REMOTE_ENV_PATH" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" -eq 0 ] || { echo "[supervisor] ERROR syncing env file!" >&2; exit 1; }

  # 3. Atomic launch: headless worker ranks 7..1 FIRST, then head rank 0 LAST
  echo "[supervisor] Launching headless ranks $((${#HOSTS[@]} - 1))..1..."
  pids=()
  for ((i = ${#HOSTS[@]} - 1; i >= 1; i--)); do
    h="${HOSTS[$i]}"
    (
      ssh_node "$h" "bash -lc 'for v in \$(compgen -e NCCL_); do unset \"\$v\"; done; set -a; source ${V15_REMOTE_ENV_PATH}; set +a; bash ${V15_REMOTE_LAUNCHER} ${i}'"
    ) &
    pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" -eq 0 ] || { echo "[supervisor] ERROR launching worker ranks 7..1!" >&2; exit 1; }

  echo "[supervisor] Launching head rank 0 (${HOSTS[0]})..."
  ssh_node "${HOSTS[0]}" "bash -lc 'for v in \$(compgen -e NCCL_); do unset \"\$v\"; done; set -a; source ${V15_REMOTE_ENV_PATH}; set +a; bash ${V15_REMOTE_LAUNCHER} 0'"

  if [ "$V15_DRY_RUN" = "1" ]; then
    echo "[supervisor][DRY-RUN] Launched 8/8; skipping live /health wait and boot cert."
    return 0
  fi

  # 4. Wait for /health + 1-prompt 0-ULP boot certification
  echo "[supervisor] Waiting for http://${V15_HEAD_HOST}:${PORT}/health (timeout=${V15_BOOT_TIMEOUT_S}s)..."
  local deadline=$(( $(date +%s) + V15_BOOT_TIMEOUT_S ))
  local ready=0
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -fsS --max-time 5 "http://${V15_HEAD_HOST}:${PORT}/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 5
  done
  if [ "$ready" -ne 1 ]; then
    echo "[supervisor] ERROR: Cluster did not reach /health within ${V15_BOOT_TIMEOUT_S}s!" >&2
    cmd_bundle "boot_health_timeout" || true
    exit 1
  fi

  echo "[supervisor] Running 1-prompt 0-ULP boot certification..."
  "$PYTHON_BIN" "$WATCHDOG_PY" --certify-only --head-host "$V15_HEAD_HOST" --port "$PORT"
  echo "[supervisor] Cluster READY & 0-ULP CERTIFIED on 8/8 nodes!"
}

cmd_status() {
  echo "=== Wave-V15 8-Node Cluster Status (port ${PORT}) ==="
  if [ "$V15_DRY_RUN" = "1" ]; then
    echo "[supervisor][DRY-RUN] Would query ${#HOSTS[@]} nodes + http://${V15_HEAD_HOST}:${PORT}/health"
    return 0
  fi
  if curl -fsS --max-time 3 "http://${V15_HEAD_HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "API /health: OK (http://${V15_HEAD_HOST}:${PORT}/health)"
  else
    echo "API /health: DOWN (http://${V15_HEAD_HOST}:${PORT}/health)"
  fi
  local pids=() i h cname
  for ((i = 0; i < ${#HOSTS[@]}; i++)); do
    h="${HOSTS[$i]}"
    cname="${V15_CONTAINER_PREFIX}${i}"
    (
      info="$(ssh_node "$h" "
        st=\$(docker ps -a --filter name=^${cname}\$ --format '{{.Status}}' 2>/dev/null | head -n1); [ -z \"\$st\" ] && st='missing'
        hcas=\$(grep -l ACTIVE /sys/class/infiniband/*/ports/1/state 2>/dev/null | wc -l | tr -d ' ')
        gpu=\$(nvidia-smi --query-gpu=temperature.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')
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
  _on_term() {
    echo "[supervisor] Caught shutdown signal; stopping watchdog..."
    [ -n "$wd_pid" ] && kill -TERM "$wd_pid" 2>/dev/null || true
    [ -n "$wd_pid" ] && wait "$wd_pid" 2>/dev/null || true
    if [ "$V15_STOP_ON_EXIT" = "1" ]; then
      cmd_stop || true
    fi
    exit 0
  }
  trap _on_term TERM INT

  notify_systemd --status="Checking cluster health before starting watchdog..."
  if [ "$V15_DRY_RUN" != "1" ]; then
    if ! curl -fsS --max-time 5 "http://${V15_HEAD_HOST}:${PORT}/health" >/dev/null 2>&1; then
      echo "[supervisor] Cluster not currently serving on port ${PORT}; initiating cold start..."
      if [ -f "$V15_PRIMARY_ENV" ]; then
        cmd_start "$V15_PRIMARY_ENV" || echo "[supervisor] WARN: initial start failed; watchdog daemon will manage recovery/fallback." >&2
      else
        echo "[supervisor] WARN: V15_PRIMARY_ENV=$V15_PRIMARY_ENV not found; delegating to watchdog daemon." >&2
      fi
    else
      echo "[supervisor] Attaching to already-healthy cluster on port ${PORT}."
    fi
  fi

  notify_systemd --ready --status="Supervising 8-node Inkling-2-Small TP8 V15 cluster"
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
    echo "[supervisor][DRY-RUN] Would exec: $PYTHON_BIN ${wd_args[*]}"
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
Usage: v15_cluster_supervisor.sh <command> [args...]
Commands:
  preflight [--auto-reboot-defrag]  Check 8-node UMA buddyinfo/swap & 4/4 RoCEv2 links
  start [env_file]                  Clean start: preflight -> teardown -> launch 7..0 -> /health -> 0-ULP cert
  stop                              Graceful SIGTERM drain (30s) + rm -f + scoped IPC sweep on 8/8
  restart [env_file]                stop + start
  status                            8-node health dashboard
  logs [rank|all] [lines]           Inspect container logs
  bundle [reason]                   Capture timestamped 8-node diagnostic crash bundle
  supervise                         Run foreground systemd supervisor daemon + active watchdog
  reset-breaker                     Clear circuit breaker & V12 fallback lock after repair
EOF
}

CMD="${1:-}"
[ -n "$CMD" ] || { usage >&2; exit 1; }
shift || true

case "$CMD" in
  preflight)     cmd_preflight "$@" ;;
  start)         cmd_start "$@" ;;
  stop)          cmd_stop "$@" ;;
  restart)       cmd_start "$@" ;;
  status)        cmd_status "$@" ;;
  logs)          cmd_logs "$@" ;;
  bundle)        cmd_bundle "$@" ;;
  supervise)     cmd_supervise "$@" ;;
  reset-breaker) cmd_reset_breaker "$@" ;;
  -h|--help)     usage ;;
  *)             echo "[supervisor] Unknown command: $CMD" >&2; usage >&2; exit 1 ;;
esac

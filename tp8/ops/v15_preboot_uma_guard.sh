#!/usr/bin/env bash
# Wave-V15 Pre-Boot GB10 UMA Buddy Fragmentation & Swap Guard
# Inspects /proc/buddyinfo (orders >= 9/10 & order-13 32 MiB blocks) + /proc/meminfo across all 8 DGX Spark nodes.
# Remediates non-destructively (drop_caches=3 + compact_memory=1 + swapoff/swapon when containers are stopped)
# and supports explicit --auto-reboot-defrag when order-13 contiguous blocks remain below threshold (default 2500).
#
# Exit codes:
#   0 = All nodes clean & ready for 8-node TP8 launch
#   1 = Hard error (SSH failure, container still running when reboot requested, bad args)
#   2 = Fragmented / reboot-defrag recommended (--auto-reboot-defrag not enabled)
set -euo pipefail

# Parameterized cluster configuration (ZERO hardcoded secrets)
V15_HOSTS_DEFAULT="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
V15_HOSTS="${V15_HOSTS:-$V15_HOSTS_DEFAULT}"
V15_SSH_USER="${V15_SSH_USER:-${USER:-jeweled}}"
V15_SSH_JUMP_HOST="${V15_SSH_JUMP_HOST:-spark-one}"
V15_SSH_JUMP_MAP="${V15_SSH_JUMP_MAP:-spark-five=10.77.0.51 spark-six=10.77.0.61}"
V15_SSH_BIN="${V15_SSH_BIN:-ssh}"
V15_SSH_TIMEOUT="${V15_SSH_TIMEOUT:-10}"
V15_CONTAINER_PREFIX="${V15_CONTAINER_PREFIX:-inkling-tp8-r}"

# GB10 121.63 GiB UMA preboot thresholds (clean reboot = ~3720 order-13 32 MiB blocks, ~116.3 GiB free, 0 MiB swap)
UMA_MIN_ORDER13_BLOCKS="${UMA_MIN_ORDER13_BLOCKS:-2500}"
UMA_MIN_ORDER_GE9_BLOCKS="${UMA_MIN_ORDER_GE9_BLOCKS:-2048}"
UMA_MIN_ORDER_GE10_BLOCKS="${UMA_MIN_ORDER_GE10_BLOCKS:-1500}"
UMA_MIN_ORDER_GE9_GIB="${UMA_MIN_ORDER_GE9_GIB:-80}"
UMA_MAX_SWAP_USED_MIB="${UMA_MAX_SWAP_USED_MIB:-256}"
UMA_MIN_MEM_AVAIL_GIB="${UMA_MIN_MEM_AVAIL_GIB:-100}"
UMA_REBOOT_WAIT_SEC="${UMA_REBOOT_WAIT_SEC:-300}"
UMA_REBOOT_POLL_SEC="${UMA_REBOOT_POLL_SEC:-5}"
UMA_PAGE_SIZE_OVERRIDE="${UMA_PAGE_SIZE_OVERRIDE:-}"

LOCAL_MODE=0
PROC_ROOT=""
DRY_RUN=0
NO_REMEDIATE=0
AUTO_REBOOT_DEFRAG=0
FORCE_REMEDIATE=0
JSON_OUT=0

usage() {
  cat <<'EOF'
Usage: v15_preboot_uma_guard.sh [OPTIONS]
  --local                 Inspect local host only (no SSH)
  --proc-root <dir>       Inspect mock procfs root (implies --local; reads <dir>/buddyinfo & <dir>/meminfo)
  --hosts "<h0 h1 ...>"   Override space-separated host list
  --no-remediate          Audit-only mode (do not run drop_caches / compact_memory / swapoff)
  --auto-reboot-defrag    Reboot ONLY nodes whose order-13 blocks stay below UMA_MIN_ORDER13_BLOCKS after defrag
                          (requires inkling containers stopped; reboots non-jump-host nodes before jump host)
  --force                 Allow non-destructive sysctl remediation even if containers check fails (NEVER forces reboot)
  --dry-run               Print remediation/reboot actions without mutating kernel state
  --json                  Emit machine-readable JSON summary on stdout (human logs go to stderr)
  -h, --help              Show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --local) LOCAL_MODE=1; shift ;;
    --proc-root) PROC_ROOT="${2:?missing --proc-root arg}"; LOCAL_MODE=1; shift 2 ;;
    --hosts) V15_HOSTS="${2:?missing --hosts arg}"; shift 2 ;;
    --no-remediate) NO_REMEDIATE=1; shift ;;
    --auto-reboot-defrag) AUTO_REBOOT_DEFRAG=1; shift ;;
    --force) FORCE_REMEDIATE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --json) JSON_OUT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[uma_guard] Unknown option: $1" >&2; usage >&2; exit 1 ;;
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

run_on_node() {
  local h="$1"; shift
  local cmd="$*"
  if [ "$LOCAL_MODE" -eq 1 ]; then
    bash -c "$cmd"
    return $?
  fi
  local target=""
  if target="$(jump_target_for "$h")"; then
    "$V15_SSH_BIN" -o "ConnectTimeout=$V15_SSH_TIMEOUT" -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new -J "${V15_SSH_USER}@${V15_SSH_JUMP_HOST}" \
      "${V15_SSH_USER}@${target}" "$cmd"
  else
    "$V15_SSH_BIN" -o "ConnectTimeout=$V15_SSH_TIMEOUT" -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new "${V15_SSH_USER}@${h}" "$cmd"
  fi
}

# Emit a single parseable KV line for a node:
# OK|ERR|<mem_total_gib>|<mem_avail_gib>|<swap_used_mib>|<max_order>|<order_ge9_blks>|<order_ge10_blks>|<order_ge9_gib>|<order13_blks>|<running_containers>
probe_node_metrics() {
  local h="$1"
  local p_root="${PROC_ROOT:-/proc}"
  local page_ov="${UMA_PAGE_SIZE_OVERRIDE:-}"
  local prefix="$V15_CONTAINER_PREFIX"

  # Remote/local probe script (pure awk + standard coreutils, zero external deps)
  local probe_cmd
  probe_cmd=$(cat <<EOF
set -e
PROOT="${p_root}"
PAGE_SZ="${page_ov}"
if [ -z "\$PAGE_SZ" ]; then
  PAGE_SZ="\$(getconf PAGESIZE 2>/dev/null || echo 4096)"
fi
if [ ! -r "\$PROOT/buddyinfo" ] || [ ! -r "\$PROOT/meminfo" ]; then
  echo "ERR|procfs_missing"
  exit 0
fi
RUNNING=0
if [ -n "${V15_MOCK_CONTAINERS_FILE:-}" ] && [ -r "${V15_MOCK_CONTAINERS_FILE:-}" ]; then
  RUNNING="\$(grep -c "^${prefix}" "${V15_MOCK_CONTAINERS_FILE:-}" 2>/dev/null || true)"
elif command -v docker >/dev/null 2>&1; then
  RUNNING="\$(docker ps --filter "name=^${prefix}" -q 2>/dev/null | wc -l | tr -d ' ')"
fi
MEM_LINE="\$(awk '
  /^MemTotal:/     { mt=\$2 }
  /^MemAvailable:/ { ma=\$2; has_ma=1 }
  /^MemFree:/      { mf=\$2 }
  /^SwapTotal:/    { st=\$2 }
  /^SwapFree:/     { sf=\$2 }
  END {
    if (!has_ma) ma=mf;
    su = (st > sf) ? (st - sf) : 0;
    printf "%.2f|%.2f|%.1f", mt/1048576.0, ma/1048576.0, su/1024.0;
  }' "\$PROOT/meminfo")"
BUDDY_LINE="\$(awk -v psz="\$PAGE_SZ" '
  /zone/ {
    z=0;
    for (i=1; i<=NF; i++) { if (\$i == "zone") { z=i; break; } }
    if (z == 0) next;
    ord=0;
    for (i=z+2; i<=NF; i++) {
      if (\$i ~ /^[0-9]+\$/) {
        tot[ord] += \$i + 0;
        if (ord > maxo) maxo = ord;
        ord++;
      }
    }
  }
  END {
    ge9_b=0; ge10_b=0; ge9_bytes=0;
    for (o=9; o<=maxo; o++) {
      c = tot[o] + 0;
      ge9_b += c;
      if (o >= 10) ge10_b += c;
      ge9_bytes += c * psz * (2 ^ o);
    }
    o13 = (maxo >= 13) ? (tot[13] + 0) : 0;
    printf "%d|%d|%d|%.2f|%d", maxo, ge9_b, ge10_b, ge9_bytes/1073741824.0, o13;
  }' "\$PROOT/buddyinfo")"
echo "OK|\${MEM_LINE}|\${BUDDY_LINE}|\${RUNNING}"
EOF
)
  local out=""
  if ! out="$(run_on_node "$h" "$probe_cmd" 2>/dev/null | tail -n 1)"; then
    echo "ERR|ssh_failed"
    return 0
  fi
  if [ -z "$out" ]; then
    echo "ERR|empty_probe"
    return 0
  fi
  echo "$out"
}

remediate_node_nondestructive() {
  local h="$1"
  local swap_mib="$2"
  local running="$3"

  if [ "$running" -gt 0 ] && [ "$FORCE_REMEDIATE" -ne 1 ]; then
    log "[uma_guard] SKIP non-destructive defrag on $h: $running running ${V15_CONTAINER_PREFIX}* container(s) detected (stop containers first or pass --force)"
    return 1
  fi

  local do_swap=0
  if awk -v s="$swap_mib" 'BEGIN { exit !(s > 0.5) }'; then
    do_swap=1
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    log "[uma_guard] DRY-RUN defrag on $h: sync; sysctl -w vm.drop_caches=3; sysctl -w vm.compact_memory=1$( [ "$do_swap" -eq 1 ] && echo '; swapoff -a && swapon -a' )"
    return 0
  fi

  # Optional mock hook for offline testing
  if [ -n "${V15_MOCK_DEFRAG_CMD:-}" ]; then
    log "[uma_guard] Executing mock defrag hook on $h..."
    V15_TARGET_HOST="$h" V15_SWAP_MIB="$swap_mib" V15_DO_SWAP="$do_swap" bash -c "$V15_MOCK_DEFRAG_CMD"
    return $?
  fi

  local rem_cmd="sync && sudo -n sysctl -w vm.drop_caches=3 >/dev/null && sudo -n sysctl -w vm.compact_memory=1 >/dev/null"
  if [ "$do_swap" -eq 1 ]; then
    rem_cmd="${rem_cmd} && sudo -n swapoff -a && sudo -n swapon -a"
  fi
  log "[uma_guard] Running non-destructive UMA defrag on $h (swap_drain=$do_swap)..."
  if ! run_on_node "$h" "$rem_cmd"; then
    log "[uma_guard] WARN: non-destructive defrag command returned non-zero on $h (check passwordless sudo for sysctl/swapoff)"
    return 1
  fi
  return 0
}

# Evaluates metric tuple and prints reason list (empty = PASS)
evaluate_reasons() {
  local mem_avail="$1" swap_mib="$2" max_ord="$3" ge9_b="$4" ge10_b="$5" ge9_gib="$6" o13="$7"
  local reasons=""
  if [ "$max_ord" -ge 13 ] && [ "$o13" -lt "$UMA_MIN_ORDER13_BLOCKS" ]; then
    reasons="${reasons} order13=${o13}<${UMA_MIN_ORDER13_BLOCKS}"
  fi
  if [ "$ge9_b" -lt "$UMA_MIN_ORDER_GE9_BLOCKS" ]; then
    reasons="${reasons} order>=9_blks=${ge9_b}<${UMA_MIN_ORDER_GE9_BLOCKS}"
  fi
  if [ "$ge10_b" -lt "$UMA_MIN_ORDER_GE10_BLOCKS" ]; then
    reasons="${reasons} order>=10_blks=${ge10_b}<${UMA_MIN_ORDER_GE10_BLOCKS}"
  fi
  if awk -v g="$ge9_gib" -v m="$UMA_MIN_ORDER_GE9_GIB" 'BEGIN { exit !(g < m) }'; then
    reasons="${reasons} order>=9_gib=${ge9_gib}<${UMA_MIN_ORDER_GE9_GIB}"
  fi
  if awk -v s="$swap_mib" -v m="$UMA_MAX_SWAP_USED_MIB" 'BEGIN { exit !(s > m) }'; then
    reasons="${reasons} swap_mib=${swap_mib}>${UMA_MAX_SWAP_USED_MIB}"
  fi
  if awk -v a="$mem_avail" -v m="$UMA_MIN_MEM_AVAIL_GIB" 'BEGIN { exit !(a < m) }'; then
    reasons="${reasons} mem_avail_gib=${mem_avail}<${UMA_MIN_MEM_AVAIL_GIB}"
  fi
  echo "${reasons# }"
}

reboot_and_wait() {
  local nodes_to_reboot="$1"
  local non_jump="" jump_included=0 h

  for h in $nodes_to_reboot; do
    if [ "$h" = "$V15_SSH_JUMP_HOST" ]; then
      jump_included=1
    else
      non_jump="${non_jump} ${h}"
    fi
  done
  non_jump="${non_jump# }"

  # Order: reboot non-jump-host nodes FIRST so spark-five/six receive reboot before spark-one drops!
  local ordered=""
  if [ -n "$non_jump" ]; then
    ordered="$non_jump"
  fi
  if [ "$jump_included" -eq 1 ]; then
    ordered="${ordered:+${ordered} }${V15_SSH_JUMP_HOST}"
  fi

  log "[uma_guard] Initiating ordered --auto-reboot-defrag sequence: ${ordered}"
  for h in $ordered; do
    if [ "$DRY_RUN" -eq 1 ]; then
      log "[uma_guard] DRY-RUN reboot on $h: sudo -n reboot"
      continue
    fi
    if [ -n "${V15_MOCK_REBOOT_CMD:-}" ]; then
      log "[uma_guard] Executing mock reboot hook on $h..."
      V15_TARGET_HOST="$h" bash -c "$V15_MOCK_REBOOT_CMD"
      continue
    fi
    if [ "$h" = "$V15_SSH_JUMP_HOST" ] && [ -n "$non_jump" ]; then
      sleep 2
    fi
    log "[uma_guard] Rebooting $h for UMA buddy defragmentation..."
    run_on_node "$h" "nohup sudo -n sh -c 'sleep 1 && reboot' >/dev/null 2>&1 &" || true
  done

  if [ "$DRY_RUN" -eq 1 ] || [ -n "${V15_MOCK_REBOOT_CMD:-}" ]; then
    return 0
  fi

  # Wait for rebooted nodes to come back online
  sleep 10
  local deadline=$(( $(date +%s) + UMA_REBOOT_WAIT_SEC ))
  local remaining="$ordered"
  while [ -n "$remaining" ] && [ "$(date +%s)" -lt "$deadline" ]; do
    local next_rem=""
    for h in $remaining; do
      if run_on_node "$h" "true" >/dev/null 2>&1; then
        log "[uma_guard] $h is back online after reboot-defrag."
      else
        next_rem="${next_rem} ${h}"
      fi
    done
    remaining="${next_rem# }"
    [ -n "$remaining" ] && sleep "$UMA_REBOOT_POLL_SEC"
  done

  if [ -n "$remaining" ]; then
    log "[uma_guard] ERROR: Timed out waiting for rebooted node(s): ${remaining}"
    return 1
  fi
  return 0
}

# Build target node list
if [ "$LOCAL_MODE" -eq 1 ]; then
  TARGET_HOSTS="$(hostname -s 2>/dev/null || echo localhost)"
else
  TARGET_HOSTS="$V15_HOSTS"
fi

log "=== Wave-V15 Pre-Boot UMA Buddy & Swap Guard ==="
log "Thresholds: order13>=${UMA_MIN_ORDER13_BLOCKS} (32MiB), order>=9>=${UMA_MIN_ORDER_GE9_BLOCKS} blks / ${UMA_MIN_ORDER_GE9_GIB} GiB, order>=10>=${UMA_MIN_ORDER_GE10_BLOCKS}, swap<=${UMA_MAX_SWAP_USED_MIB} MiB, avail>=${UMA_MIN_MEM_AVAIL_GIB} GiB"

HARD_ERR=0
DEFRAG_NEEDED_NODES=""
REBOOT_NEEDED_NODES=""
JSON_RECORDS=""

append_json_record() {
  local rec="$1"
  if [ -z "$JSON_RECORDS" ]; then
    JSON_RECORDS="$rec"
  else
    JSON_RECORDS="${JSON_RECORDS},${rec}"
  fi
}

for h in $TARGET_HOSTS; do
  raw="$(probe_node_metrics "$h")"
  IFS='|' read -r status f1 f2 f3 f4 f5 f6 f7 f8 f9 <<< "$raw"
  if [ "$status" != "OK" ]; then
    log "[uma_guard] ERROR $h: probe failed ($f1)"
    HARD_ERR=1
    append_json_record "{\"host\":\"${h}\",\"status\":\"ERROR\",\"error\":\"${f1}\"}"
    continue
  fi
  mem_total="$f1"; mem_avail="$f2"; swap_mib="$f3"; max_ord="$f4"
  ge9_b="$f5"; ge10_b="$f6"; ge9_gib="$f7"; o13="$f8"; running="$f9"

  reasons="$(evaluate_reasons "$mem_avail" "$swap_mib" "$max_ord" "$ge9_b" "$ge10_b" "$ge9_gib" "$o13")"
  remediated=0

  if [ -z "$reasons" ]; then
    log "[uma_guard] PASS  $h: avail=${mem_avail}/${mem_total}GiB swap=${swap_mib}MiB order13=${o13} order>=9=${ge9_b}blks(${ge9_gib}GiB) order>=10=${ge10_b} running=${running}"
  else
    log "[uma_guard] WARN  $h: BEFORE avail=${mem_avail}GiB swap=${swap_mib}MiB order13=${o13} order>=9=${ge9_b}blks(${ge9_gib}GiB) reasons=[${reasons}]"
    if [ "$NO_REMEDIATE" -eq 0 ]; then
      if remediate_node_nondestructive "$h" "$swap_mib" "$running"; then
        remediated=1
        if [ "$DRY_RUN" -eq 0 ]; then
          raw2="$(probe_node_metrics "$h")"
          IFS='|' read -r status2 f1 f2 f3 f4 f5 f6 f7 f8 f9 <<< "$raw2"
          if [ "$status2" = "OK" ]; then
            mem_total="$f1"; mem_avail="$f2"; swap_mib="$f3"; max_ord="$f4"
            ge9_b="$f5"; ge10_b="$f6"; ge9_gib="$f7"; o13="$f8"; running="$f9"
            reasons="$(evaluate_reasons "$mem_avail" "$swap_mib" "$max_ord" "$ge9_b" "$ge10_b" "$ge9_gib" "$o13")"
          fi
        fi
      fi
    fi
    if [ -z "$reasons" ]; then
      log "[uma_guard] FIXED $h: AFTER  avail=${mem_avail}GiB swap=${swap_mib}MiB order13=${o13} order>=9=${ge9_b}blks(${ge9_gib}GiB) (non-destructive defrag succeeded)"
    else
      log "[uma_guard] [DEFRAG_REQUIRED] $h: AFTER  avail=${mem_avail}GiB swap=${swap_mib}MiB order13=${o13} order>=9=${ge9_b}blks(${ge9_gib}GiB) reasons=[${reasons}]"
      DEFRAG_NEEDED_NODES="${DEFRAG_NEEDED_NODES} ${h}"
      if [ "$running" -gt 0 ] && [ "$AUTO_REBOOT_DEFRAG" -eq 1 ]; then
        log "[uma_guard] ERROR $h: refusing --auto-reboot-defrag while $running ${V15_CONTAINER_PREFIX}* container(s) are running!"
        HARD_ERR=1
      else
        REBOOT_NEEDED_NODES="${REBOOT_NEEDED_NODES} ${h}"
      fi
    fi
  fi

  node_state="OK"
  [ -n "$reasons" ] && node_state="DEFRAG_REQUIRED"
  append_json_record "{\"host\":\"${h}\",\"status\":\"${node_state}\",\"mem_total_gib\":${mem_total},\"mem_avail_gib\":${mem_avail},\"swap_used_mib\":${swap_mib},\"max_order\":${max_ord},\"order_ge9_blocks\":${ge9_b},\"order_ge10_blocks\":${ge10_b},\"order_ge9_gib\":${ge9_gib},\"order_13_blocks\":${o13},\"running_containers\":${running},\"remediated\":${remediated},\"reasons\":\"${reasons}\"}"
done

DEFRAG_NEEDED_NODES="${DEFRAG_NEEDED_NODES# }"
REBOOT_NEEDED_NODES="${REBOOT_NEEDED_NODES# }"
REBOOT_PERFORMED=0

if [ "$HARD_ERR" -eq 0 ] && [ -n "$REBOOT_NEEDED_NODES" ] && [ "$AUTO_REBOOT_DEFRAG" -eq 1 ]; then
  if reboot_and_wait "$REBOOT_NEEDED_NODES"; then
    REBOOT_PERFORMED=1
    if [ "$DRY_RUN" -eq 0 ]; then
      # Re-verify all rebooted nodes cleanly!
      STILL_BAD=""
      JSON_RECORDS=""
      for h in $TARGET_HOSTS; do
        raw="$(probe_node_metrics "$h")"
        IFS='|' read -r status f1 f2 f3 f4 f5 f6 f7 f8 f9 <<< "$raw"
        if [ "$status" != "OK" ]; then
          HARD_ERR=1
          append_json_record "{\"host\":\"${h}\",\"status\":\"ERROR\",\"error\":\"${f1}\"}"
          continue
        fi
        mem_total="$f1"; mem_avail="$f2"; swap_mib="$f3"; max_ord="$f4"
        ge9_b="$f5"; ge10_b="$f6"; ge9_gib="$f7"; o13="$f8"; running="$f9"
        reasons="$(evaluate_reasons "$mem_avail" "$swap_mib" "$max_ord" "$ge9_b" "$ge10_b" "$ge9_gib" "$o13")"
        node_state="OK"
        if [ -n "$reasons" ]; then
          node_state="DEFRAG_REQUIRED"
          STILL_BAD="${STILL_BAD} ${h}"
          log "[uma_guard] FAIL  $h after reboot-defrag: order13=${o13} swap=${swap_mib}MiB reasons=[${reasons}]"
        else
          log "[uma_guard] PASS  $h after reboot-defrag: avail=${mem_avail}GiB swap=${swap_mib}MiB order13=${o13} order>=9=${ge9_gib}GiB"
        fi
        append_json_record "{\"host\":\"${h}\",\"status\":\"${node_state}\",\"mem_total_gib\":${mem_total},\"mem_avail_gib\":${mem_avail},\"swap_used_mib\":${swap_mib},\"max_order\":${max_ord},\"order_ge9_blocks\":${ge9_b},\"order_ge10_blocks\":${ge10_b},\"order_ge9_gib\":${ge9_gib},\"order_13_blocks\":${o13},\"running_containers\":${running},\"remediated\":1,\"rebooted\":1,\"reasons\":\"${reasons}\"}"
      done
      DEFRAG_NEEDED_NODES="${STILL_BAD# }"
    else
      DEFRAG_NEEDED_NODES=""
    fi
  else
    HARD_ERR=1
  fi
fi

FINAL_RC=0
FINAL_STATUS="OK"
if [ "$HARD_ERR" -ne 0 ]; then
  FINAL_RC=1
  FINAL_STATUS="ERROR"
elif [ -n "$DEFRAG_NEEDED_NODES" ]; then
  FINAL_RC=2
  FINAL_STATUS="DEFRAG_REQUIRED"
  log "[uma_guard] ACTION NEEDED: Fragmented node(s) [${DEFRAG_NEEDED_NODES}] below order-13>=${UMA_MIN_ORDER13_BLOCKS}. Re-run with --auto-reboot-defrag (when containers stopped) to restore 3,720x32MiB contiguous UMA."
else
  log "[uma_guard] ALL NODES PASS: contiguous UMA buddy pools and swap are clean."
fi

if [ "$JSON_OUT" -eq 1 ]; then
  printf '{"status":"%s","exit_code":%d,"reboot_performed":%d,"defrag_needed_nodes":"%s","nodes":[%s]}\n' \
    "$FINAL_STATUS" "$FINAL_RC" "$REBOOT_PERFORMED" "$DEFRAG_NEEDED_NODES" "$JSON_RECORDS"
fi
exit "$FINAL_RC"

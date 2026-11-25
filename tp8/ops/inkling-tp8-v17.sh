#!/usr/bin/env bash
# Inkling-2-Small TP8 V17 Permanent Production Launcher
# Upgraded from inkling-tp8-v15.sh with:
#   - Multi-alias served model names (--served-model-name inkling-2-small Inkling-2-Small)
#   - Priority scheduling (--scheduling-policy priority) + standalone InklingScheduler
#     (--scheduler-cls inkling_sched.InklingScheduler) for short-skip + pure-step separation
#   - Persistent CUDA PTX cache mount (-v "$HOME/.nv:/root/.nv") + CUDA_*/PYTHON* env forwarding
#   - Production container guards: --oom-score-adj 500 --ulimit nofile=524288:524288
#   - Clean BIND_HOST override (0.0.0.0 prod, 127.0.0.1 isolated QA)
#   - Preserves 4/4 CX-7 HCA discovery, SHA256SUMS verification, MANIFEST verification,
#     bounded Docker JSON logs (200m x 5), and graceful 30s SIGTERM drain.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

if [ "${1:-}" = "--generate-rdma-manifest" ]; then
  RDMA_DIR="${2:-$HERE/rdma}"
  [ -d "$RDMA_DIR" ] || { echo "[v17-launch] ERROR: RDMA dir not found: $RDMA_DIR" >&2; exit 1; }
  HASH_CMD=""
  if command -v sha256sum >/dev/null 2>&1; then
    HASH_CMD="sha256sum"
  elif command -v shasum >/dev/null 2>&1; then
    HASH_CMD="shasum -a 256"
  else
    echo "[v17-launch] ERROR: neither sha256sum nor shasum found" >&2; exit 1
  fi
  (
    cd "$RDMA_DIR"
    shopt -s nullglob
    sos=( *.so )
    [ "${#sos[@]}" -gt 0 ] || { echo "[v17-launch] ERROR: no .so files in $RDMA_DIR" >&2; exit 1; }
    $HASH_CMD "${sos[@]}" > SHA256SUMS
    echo "[v17-launch] Wrote $RDMA_DIR/SHA256SUMS (${#sos[@]} libraries)"
  )
  exit 0
fi

DRY_RUN="${V17_DRY_RUN:-${V15_DRY_RUN:-0}}"
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
  shift
fi

RANK="${1:?usage: inkling-tp8-v17.sh [--dry-run] <rank 0..7> | --generate-rdma-manifest [dir]}"
case "$RANK" in
  0|1|2|3|4|5|6|7) ;;
  *) echo "[v17-launch] ERROR: rank must be 0..7, got '$RANK'" >&2; exit 1 ;;
esac

IMAGE="${INKLING_IMAGE:-vllm/vllm-openai:inkling-2-small}"
RDMA_HOST_DIR="${V17_RDMA_HOST_DIR:-${V15_RDMA_HOST_DIR:-$HERE/rdma}}"
V17_SYSFS_IB_ROOT="${V17_SYSFS_IB_ROOT:-${V15_SYSFS_IB_ROOT:-/sys/class/infiniband}}"
V17_SYSFS_NET_ROOT="${V17_SYSFS_NET_ROOT:-${V15_SYSFS_NET_ROOT:-/sys/class/net}}"
V17_IP_BIN="${V17_IP_BIN:-${V15_IP_BIN:-ip}}"
V17_REQUIRED_ACTIVE_HCAS="${V17_REQUIRED_ACTIVE_HCAS:-${V15_REQUIRED_ACTIVE_HCAS:-4}}"
V17_RDMA_AUTO_DISCOVER_HCAS="${V17_RDMA_AUTO_DISCOVER_HCAS:-${V15_RDMA_AUTO_DISCOVER_HCAS:-1}}"
V17_SKIP_RDMA_PREFLIGHT="${V17_SKIP_RDMA_PREFLIGHT:-${V15_SKIP_RDMA_PREFLIGHT:-0}}"
V17_VERIFY_RDMA_SHA256="${V17_VERIFY_RDMA_SHA256:-${V15_VERIFY_RDMA_SHA256:-1}}"
V17_RDMA_MANIFEST="${V17_RDMA_MANIFEST:-${V15_RDMA_MANIFEST:-$RDMA_HOST_DIR/SHA256SUMS}}"
V17_STOP_TIMEOUT="${V17_STOP_TIMEOUT:-${V15_STOP_TIMEOUT:-30}}"
V17_DOCKER_LOG_MAX_SIZE="${V17_DOCKER_LOG_MAX_SIZE:-${V15_DOCKER_LOG_MAX_SIZE:-200m}}"
V17_DOCKER_LOG_MAX_FILES="${V17_DOCKER_LOG_MAX_FILES:-${V15_DOCKER_LOG_MAX_FILES:-5}}"
V17_DETACH="${V17_DETACH:-${V15_DETACH:-1}}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"

if [ "${INKLING_MOE_TP:-0}" = "1" ]; then
  MODEL_DIR="${MODEL_DIR:-$HOME/models/inkling2-small-tp8r$RANK}"
  [ -d "$MODEL_DIR" ] || { echo "[v17-launch] ERROR: no TP slice $MODEL_DIR" >&2; exit 1; }
fi
MODEL_DIR="${MODEL_DIR:-$HOME/models/inkling2-small-ep8r$RANK}"
[ -d "$MODEL_DIR" ] || MODEL_DIR="$HOME/models/inkling2-small"
[ -d "$MODEL_DIR" ] || { echo "[v17-launch] ERROR: MODEL_DIR missing: $MODEL_DIR" >&2; exit 1; }

IP="${V17_NODE_IP:-${V15_NODE_IP:-10.77.0.$((RANK + 1))1}}"
MASTER_ADDR="${V17_MASTER_ADDR:-${V15_MASTER_ADDR:-10.77.0.11}}"

# Preflight 1: Validate ulimit -l (memlock unlimited required for RDMA MRs & CUDA pin)
check_memlock() {
  local cur_lim
  cur_lim="$(ulimit -l 2>/dev/null || echo unknown)"
  if [ "$cur_lim" != "unlimited" ]; then
    ulimit -l unlimited 2>/dev/null || true
    cur_lim="$(ulimit -l 2>/dev/null || echo unknown)"
  fi
  if [ "$cur_lim" != "unlimited" ]; then
    if [ "${V17_STRICT_MEMLOCK:-0}" = "1" ]; then
      echo "[v17-launch] ERROR: ulimit -l is '$cur_lim' (must be unlimited)" >&2
      exit 1
    fi
    echo "[v17-launch] WARN: host shell ulimit -l='$cur_lim'; enforcing --ulimit memlock=-1 on Docker container" >&2
  fi
}
check_memlock

# Preflight 2: Dynamic RoCEv2 netdev interface resolution
resolve_roce_ifname() {
  local desired="${NCCL_SOCKET_IFNAME:-}"
  local found=""
  if command -v "$V17_IP_BIN" >/dev/null 2>&1; then
    found="$("$V17_IP_BIN" -o -4 addr show 2>/dev/null | awk -v ip="$IP/" 'index($4, ip) == 1 { print $2; exit }')"
  fi
  if [ -z "$found" ] && [ -n "$desired" ]; then
    if [ -r "$V17_SYSFS_NET_ROOT/$desired/operstate" ]; then
      local st
      st="$(tr -d '[:space:]' < "$V17_SYSFS_NET_ROOT/$desired/operstate")"
      if [ "$st" = "up" ] || [ "$st" = "UP" ]; then
        found="$desired"
      fi
    fi
  fi
  if [ -z "$found" ] && [ -d "$V17_SYSFS_NET_ROOT" ]; then
    local cand name st
    for cand in "$V17_SYSFS_NET_ROOT"/enp1s0*; do
      [ -d "$cand" ] || continue
      name="$(basename "$cand")"
      st="$(tr -d '[:space:]' < "$cand/operstate" 2>/dev/null || echo down)"
      if [ "$st" = "up" ] || [ "$st" = "UP" ]; then
        found="$name"
        break
      fi
    done
  fi
  if [ -z "$found" ]; then
    if [ "$V17_SKIP_RDMA_PREFLIGHT" = "1" ]; then
      found="${desired:-enp1s0f0np0}"
    else
      echo "[v17-launch] ERROR: could not find UP RoCEv2 netdev for IP $IP" >&2
      exit 1
    fi
  fi
  echo "$found"
}

# Preflight 3: Dynamic 4/4 ConnectX-7 HCA Active & GID validation
verify_and_resolve_hcas() {
  local default_hcas="rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1"
  local configured="${NCCL_IB_HCA:-$default_hcas}"
  local gid_idx="${NCCL_IB_GID_INDEX:-5}"

  if [ "$V17_SKIP_RDMA_PREFLIGHT" = "1" ]; then
    echo "$configured"
    return 0
  fi
  if [ ! -d "$V17_SYSFS_IB_ROOT" ]; then
    echo "[v17-launch] ERROR: IB sysfs missing at $V17_SYSFS_IB_ROOT" >&2
    exit 1
  fi

  is_hca_active() {
    local dev="$1"
    local st_file="$V17_SYSFS_IB_ROOT/$dev/ports/1/state"
    local ll_file="$V17_SYSFS_IB_ROOT/$dev/ports/1/link_layer"
    local gid_file="$V17_SYSFS_IB_ROOT/$dev/ports/1/gids/$gid_idx"
    [ -r "$st_file" ] || return 1
    grep -q "ACTIVE" "$st_file" 2>/dev/null || return 1
    if [ -r "$ll_file" ] && ! grep -qi "Ethernet" "$ll_file" 2>/dev/null; then
      return 1
    fi
    if [ -r "$gid_file" ]; then
      local gval
      gval="$(tr -d '[:space:]' < "$gid_file")"
      if [ "$gval" = "0000:0000:0000:0000:0000:0000:0000:0000" ]; then
        echo "[v17-launch] WARN: HCA $dev GID[$gid_idx] is zeroed" >&2
        return 1
      fi
    fi
    return 0
  }

  local active_configured="" dev
  local IFS_SAVED="$IFS"
  IFS=','
  for dev in $configured; do
    IFS="$IFS_SAVED"
    dev="${dev#"${dev%%[![:space:]]*}"}"
    dev="${dev%"${dev##*[![:space:]]}"}"
    [ -z "$dev" ] && continue
    if is_hca_active "$dev"; then
      active_configured="${active_configured:+${active_configured},}${dev}"
    else
      local st_txt="missing"
      [ -r "$V17_SYSFS_IB_ROOT/$dev/ports/1/state" ] && st_txt="$(cat "$V17_SYSFS_IB_ROOT/$dev/ports/1/state" 2>/dev/null || echo err)"
      echo "[v17-launch] WARN: Configured CX-7 HCA '$dev' not ACTIVE (state: $st_txt)" >&2
    fi
    IFS=','
  done
  IFS="$IFS_SAVED"

  local resolved="$active_configured"
  local active_count=0
  if [ -n "$resolved" ]; then
    active_count="$(printf '%s\n' "$resolved" | tr ',' '\n' | grep -c .)"
  fi

  if [ "$active_count" -lt "$V17_REQUIRED_ACTIVE_HCAS" ] && [ "$V17_RDMA_AUTO_DISCOVER_HCAS" = "1" ]; then
    local discovered="" dpath dname p0_list="" p2_list="" other_list=""
    for dpath in "$V17_SYSFS_IB_ROOT"/*; do
      [ -d "$dpath" ] || continue
      dname="$(basename "$dpath")"
      if is_hca_active "$dname"; then
        case "$dname" in
          roceP2*) p2_list="${p2_list} ${dname}" ;;
          rocep*)  p0_list="${p0_list} ${dname}" ;;
          *)       other_list="${other_list} ${dname}" ;;
        esac
      fi
    done
    # shellcheck disable=SC2206
    local a0=($p0_list) a2=($p2_list) ao=($other_list)
    local max_n="${#a0[@]}"
    [ "${#a2[@]}" -gt "$max_n" ] && max_n="${#a2[@]}"
    local idx=0
    while [ "$idx" -lt "$max_n" ]; do
      [ "$idx" -lt "${#a0[@]}" ] && discovered="${discovered:+${discovered},}${a0[$idx]}"
      [ "$idx" -lt "${#a2[@]}" ] && discovered="${discovered:+${discovered},}${a2[$idx]}"
      idx=$((idx + 1))
    done
    if [ "${#ao[@]}" -gt 0 ]; then
      for dname in "${ao[@]}"; do
        discovered="${discovered:+${discovered},}${dname}"
      done
    fi
    if [ -n "$discovered" ]; then
      local disc_count
      disc_count="$(printf '%s\n' "$discovered" | tr ',' '\n' | grep -c .)"
      if [ "$disc_count" -ge "$V17_REQUIRED_ACTIVE_HCAS" ]; then
        echo "[v17-launch] INFO: Auto-discovered ${disc_count} ACTIVE RoCEv2 HCAs: ${discovered}" >&2
        resolved="$discovered"
        active_count="$disc_count"
      fi
    fi
  fi

  if [ "$active_count" -lt "$V17_REQUIRED_ACTIVE_HCAS" ]; then
    echo "[v17-launch] ERROR: Only ${active_count}/${V17_REQUIRED_ACTIVE_HCAS} CX-7 RoCEv2 HCAs ACTIVE on rank $RANK" >&2
    exit 1
  fi
  echo "$resolved"
}

# Preflight 4: Verify /rdma/*.so existence & SHA-256 manifest
verify_rdma_libs() {
  [ "${RDMA:-0}" = "1" ] || return 0
  [ -d "$RDMA_HOST_DIR" ] || { echo "[v17-launch] ERROR: RDMA dir missing: $RDMA_HOST_DIR" >&2; exit 1; }

  local kv k v rel host_so
  while IFS= read -r kv; do
    k="${kv%%=*}"
    v="${kv#*=}"
    case "$v" in
      /rdma/*.so)
        rel="${v#/rdma/}"
        host_so="$RDMA_HOST_DIR/$rel"
        if [ ! -s "$host_so" ]; then
          echo "[v17-launch] ERROR: $k=$v missing or empty at $host_so" >&2
          exit 1
        fi
        ;;
    esac
  done < <(env | grep -E '^(RDMA_AR_EXT|RDMA_RS_EXT|INKLING_[A-Z0-9_]+_LIB)=' || true)

  if [ "$V17_VERIFY_RDMA_SHA256" = "0" ]; then
    return 0
  fi
  if [ ! -f "$V17_RDMA_MANIFEST" ]; then
    if [ "$V17_VERIFY_RDMA_SHA256" = "strict" ]; then
      echo "[v17-launch] ERROR: V17_VERIFY_RDMA_SHA256=strict but manifest missing: $V17_RDMA_MANIFEST" >&2
      exit 1
    fi
    echo "[v17-launch] WARN: No RDMA SHA256SUMS at $V17_RDMA_MANIFEST" >&2
    return 0
  fi

  local hash_mode=""
  if command -v sha256sum >/dev/null 2>&1; then
    hash_mode="sha256sum"
  elif command -v shasum >/dev/null 2>&1; then
    hash_mode="shasum"
  else
    echo "[v17-launch] ERROR: missing sha256sum/shasum" >&2
    exit 1
  fi

  local verified=0 expected fname actual
  while read -r expected fname; do
    [ -z "$expected" ] || [ "${expected:0:1}" = "#" ] && continue
    fname="${fname#\*}"
    local full="$RDMA_HOST_DIR/$fname"
    if [ ! -f "$full" ]; then
      echo "[v17-launch] ERROR: RDMA manifest file missing: $full" >&2
      exit 1
    fi
    if [ "$hash_mode" = "sha256sum" ]; then
      actual="$(sha256sum "$full" | awk '{print $1}')"
    else
      actual="$(shasum -a 256 "$full" | awk '{print $1}')"
    fi
    if [ "$actual" != "$expected" ]; then
      echo "[v17-launch] ERROR: SHA-256 MISMATCH for $full! expected=$expected actual=$actual" >&2
      exit 1
    fi
    verified=$((verified + 1))
  done < "$V17_RDMA_MANIFEST"
  echo "[v17-launch] Verified ${verified} /rdma/*.so SHA-256 digests from $V17_RDMA_MANIFEST"
}

# Build vLLM arguments
if [ "${INKLING_FASTSTART:-1}" = "1" ]; then
  if [ "${SERVING:-0}" = "1" ]; then MAX_BATCHED="${MAX_BATCHED:-4096}"; else MAX_BATCHED="${MAX_BATCHED:-2048}"; fi
  [ -z "${VLLM_EXTRA:-}" ] && VLLM_EXTRA="--cudagraph-capture-sizes 1 2 4 8 16 24 32 48"
fi

if [ "${PARITY:-0}" = "1" ]; then
  GMU="${GMU:-0.85}" MAXLEN="${MAXLEN:-1048576}" SEQS="${SEQS:-32}" MAX_BATCHED="${MAX_BATCHED:-24576}"
else
  GMU="${GMU:-0.80}" MAXLEN="${MAXLEN:-32768}" SEQS="${SEQS:-4}" MAX_BATCHED="${MAX_BATCHED:-8192}"
fi

# V17: serve both canonical lowercase and Title-Case model aliases to prevent client 404s
SERVED_MODEL_NAMES="${SERVED_MODEL_NAMES:-inkling-2-small Inkling-2-Small}"
# shellcheck disable=SC2206
SERVED_ARR=($SERVED_MODEL_NAMES)

ARGS=(/model --served-model-name "${SERVED_ARR[@]}" --trust-remote-code
  --quantization "${QUANT:-inkling2}"
  --tensor-parallel-size 8 --nnodes 8 --node-rank "$RANK"
  --master-addr "$MASTER_ADDR" --master-port "${MASTER_PORT:-29511}"
  --distributed-executor-backend mp
  --distributed-timeout-seconds 7200 --cpu-distributed-timeout-seconds 7200
  --gpu-memory-utilization "$GMU" --max-model-len "$MAXLEN"
  --max-num-seqs "$SEQS" --max-num-batched-tokens "$MAX_BATCHED"
  --kv-cache-dtype "${KV_DTYPE:-auto}" --enable-prefix-caching --enable-chunked-prefill
  --disable-uvicorn-access-log
  --host "$BIND_HOST" --port "${PORT:-8010}")

[ -n "${SCHEDULING_POLICY:-}" ] && ARGS+=(--scheduling-policy "$SCHEDULING_POLICY")
if [ "${INKLING_SCHED:-0}" = "1" ]; then
  ARGS+=(--scheduler-cls "${SCHEDULER_CLS:-inkling_sched.InklingScheduler}")
fi
[ "${REASONING:-1}" = "1" ] && ARGS+=(--reasoning-parser inkling)
[ "$RANK" != "0" ] && ARGS+=(--headless)
if [ "${PARITY:-0}" = "1" ]; then
  ARGS+=(--enable-auto-tool-choice --tool-call-parser inkling --enable-prompt-tokens-details)
  if [ "${IMAGES:-16}" -gt 0 ]; then
    ARGS+=(--limit-mm-per-prompt "{\"image\":${IMAGES:-16}}" --mm-processor-cache-gb 0.5 --allowed-media-domains media.invalid)
  else
    ARGS+=(--limit-mm-per-prompt '{"image":0}' --mm-processor-cache-gb 0)
  fi
else
  ARGS+=(--limit-mm-per-prompt '{"image":0}' --mm-processor-cache-gb 0)
fi
if [ -n "${LONG_PREFILL_THRESHOLD:-}" ] && [ "${LONG_PREFILL_THRESHOLD}" -gt 0 ] 2>/dev/null; then
  ARGS+=(--long-prefill-token-threshold "$LONG_PREFILL_THRESHOLD" --long-prefill-token-threshold-adaptive)
fi
if [ -n "${STREAM_INTERVAL:-}" ] && [ "${STREAM_INTERVAL}" -gt 1 ] 2>/dev/null; then
  ARGS+=(--stream-interval "$STREAM_INTERVAL")
fi
if [ -n "${PREFILL_SCHEDULE_INTERVAL:-}" ] && [ "${PREFILL_SCHEDULE_INTERVAL}" -gt 1 ] 2>/dev/null; then
  ARGS+=(--prefill-schedule-interval "$PREFILL_SCHEDULE_INTERVAL")
fi
if [ -n "${CUDAGRAPH_REQ_SIZES:-}" ]; then
  case "${VLLM_EXTRA:-}" in
    *compilation-config*) ;;
    *) ARGS+=("--compilation-config={\"decode_cudagraph_capture_request_sizes\":[${CUDAGRAPH_REQ_SIZES}]}") ;;
  esac
fi
if [ "${DSPARK:-0}" = "1" ]; then
  SPEC_JSON="{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_K:-7},\"draft_sample_method\":\"${DRAFT_SAMPLE:-greedy}\"${ADAPTIVE:+,\"enable_adaptive_verification\":true}${SPEC_BATCH_SCHEDULE:+,\"num_speculative_tokens_per_batch_size\":${SPEC_BATCH_SCHEDULE}}}"
  ARGS+=(--speculative-config "$SPEC_JSON")
fi
# shellcheck disable=SC2206
[ -n "${VLLM_EXTRA:-}" ] && ARGS+=($VLLM_EXTRA)

# Dynamic RoCEv2 interface + 4-HCA + IPv4 RoCEv2 GID auto-discovery
RESOLVED_IFNAME="$(resolve_roce_ifname)"
export NCCL_SOCKET_IFNAME="$RESOLVED_IFNAME"
# Auto-discover IPv4-mapped RoCEv2 GID index (immune to stale NCCL_IB_GID_INDEX=3 in ~/.bashrc!)
resolve_rocev2_ipv4_gid() {
  local probe_hca="${1%%,*}"
  local gdir="$V17_SYSFS_IB_ROOT/$probe_hca/ports/1"
  if [ -d "$gdir/gids" ]; then
    local idx gval gtype
    for idx in 5 3 1 7 2 4 6 0; do
      [ -r "$gdir/gids/$idx" ] || continue
      gval="$(tr -d '[:space:]' < "$gdir/gids/$idx" 2>/dev/null || true)"
      gtype="$(cat "$gdir/gid_attrs/types/$idx" 2>/dev/null || true)"
      case "$gval" in
        0000:0000:0000:0000:0000:ffff:*)
          if echo "$gtype" | grep -qi "RoCE v2"; then
            echo "$idx"
            return 0
          fi
          ;;
      esac
    done
  fi
  echo "5"
}
export NCCL_IB_GID_INDEX="${V17_FORCE_GID_INDEX:-$(resolve_rocev2_ipv4_gid "${NCCL_IB_HCA:-rocep1s0f0}")}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
if [ "${SERVING:-0}" = "1" ]; then
  export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-8}"
  export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
else
  export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-4}"
  export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-4}"
fi

RESOLVED_HCAS="$(verify_and_resolve_hcas)"
export NCCL_IB_HCA="$RESOLVED_HCAS"

ENVS=(-e "GLOO_SOCKET_IFNAME=$RESOLVED_IFNAME"
  -e "INKLING_NODE_RANK=$RANK"
  -e "VLLM_HOST_IP=$IP" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1)
for k in INKLING_LOAD_STAGED INKLING_LOAD_CHUNK_MB INKLING_LOAD_DROP_CACHE PYTORCH_CUDA_ALLOC_CONF VLLM_LOGGING_LEVEL VLLM_TORCH_PROFILER_DIR VLLM_PROFILER_MAX_ITERS INKLING_MOE_TP INKLING_L2PF INKLING_L2PF_MB INKLING_L2PF_GATE INKLING_L2PF_CTAS INKLING_L2PF_WARPS INKLING_SKINNY INKLING_EXTRA_ENV; do
  [ -n "${!k:-}" ] && ENVS+=(-e "$k=${!k}")
done
export INKLING_CERT_HEAD_CACHE="${INKLING_CERT_HEAD_CACHE:-/root/.cache/certhead}"

if [ "${RDMA:-0}" = "1" ]; then
  MOUNTS_RDMA=(-v "$RDMA_HOST_DIR:/rdma:ro")
  export DSV41_RDMA_AR="${DSV41_RDMA_AR:-1}"
  export RDMA_AR_EXT="${RDMA_AR_EXT:-/rdma/rdma_ar_ext_c32_v16.so}"
  export RDMA_AR_MAX_BYTES="${RDMA_AR_MAX_BYTES:-4718592}"
  export RDMA_AR_MAX_REDUCE_BYTES="${RDMA_AR_MAX_REDUCE_BYTES:-4718592}"
  export RDMA_AR_MAX_GATHER_BYTES="${RDMA_AR_MAX_GATHER_BYTES:-1048576}"
  export DSV41_RDMA_PARLOAD="${DSV41_RDMA_PARLOAD:-1}"
  export RDMA_AR_XSTREAM="${RDMA_AR_XSTREAM:-wait}"
  export RDMA_AR_FINGERPRINT="${RDMA_AR_FINGERPRINT:-1}"
  export RDMA_AR_SPLIT_POST="${RDMA_AR_SPLIT_POST:-1}"
  export RDMA_AR_PUB_V1="${RDMA_AR_PUB_V1:-1}"
  export RDMA_AR_PUB_V2="${RDMA_AR_PUB_V2:-1}"
  export RDMA_AR_HYG_PROXY="${RDMA_AR_HYG_PROXY:-1}"
  export RDMA_AR_HYG_POLL="${RDMA_AR_HYG_POLL:-1}"
  export DSV41_RDMA_TRIM="${DSV41_RDMA_TRIM:-${RDMA_TRIM:-1}}"
  export RDMA_AR_HCAS="$RESOLVED_HCAS"
  verify_rdma_libs
else
  MOUNTS_RDMA=()
fi

# Forward every NCCL_*, INKLING_*, VLLM_*, DSV41_*, RDMA_*, CUDA_*, PYTHON*, OMP/BLAS/TORCH/MALLOC env var
while IFS= read -r kv; do
  ENVS+=(-e "$kv")
done < <(env | grep -E '^(NCCL_[A-Z0-9_]+|INKLING_[A-Z0-9_]+|VLLM_[A-Z0-9_]+|DSV41_[A-Z0-9_]+|RDMA_AR_[A-Z0-9_]+|RDMA_[A-Z0-9_]+|CUDA_[A-Z0-9_]+|PYTHON[A-Z0-9_]+|OMP_NUM_THREADS|OPENBLAS_NUM_THREADS|MKL_NUM_THREADS|TORCH_NUM_THREADS|MALLOC_[A-Z0-9_]+)=' | grep -v '^VLLM_EXTRA=' || true)

MOUNTS=(-v "$MODEL_DIR:/model:ro"
  -v "$HOME/.cache/inkling-jit:/root/.cache"
  -v "$HOME/.triton:/root/.triton"
  -v "$HOME/.nv:/root/.nv"
  -v "$HOME/.cache/certhead:/root/.cache/certhead"
  -v "$HOME/inkling-prof:/prof"
  ${MOUNTS_RDMA[@]+"${MOUNTS_RDMA[@]}"})
[ -n "${INKLING_MX_NVFP4:-}" ] && MOUNTS+=(-v "${INKLING_MX_NVFP4_HOST:-$HOME/ink-work/lossy/sidecar}:${INKLING_MX_NVFP4}:ro")

# Preflight 5: Validate PATCH_DIR/MANIFEST & every mapped source file
PATCH_DIR="${PATCH_DIR:-$HERE/patches-s6-v17}"
[ -f "$PATCH_DIR/MANIFEST" ] || { echo "[v17-launch] ERROR: MANIFEST missing at $PATCH_DIR/MANIFEST" >&2; exit 1; }
PATCH_COUNT=0
while read -r f dest; do
  [ -z "$f" ] || [ "${f:0:1}" = "#" ] && continue
  if [ ! -f "$PATCH_DIR/$f" ]; then
    echo "[v17-launch] ERROR: MANIFEST source file missing: $PATCH_DIR/$f" >&2
    exit 1
  fi
  MOUNTS+=(-v "$PATCH_DIR/$f:$dest:ro")
  PATCH_COUNT=$((PATCH_COUNT + 1))
done < "$PATCH_DIR/MANIFEST"

CONTAINER_NAME="inkling-tp8-r$RANK"
CPUSET_ARGS=()
[ -n "${INKLING_CPUSET:-}" ] && CPUSET_ARGS=(--cpuset-cpus "$INKLING_CPUSET")

DOCKER_CMD=(docker run)
[ "$V17_DETACH" = "1" ] && DOCKER_CMD+=(-d)
DOCKER_CMD+=(
  --name "$CONTAINER_NAME"
  --gpus all --privileged --ipc=host --network host
  --oom-score-adj "${V17_OOM_SCORE_ADJ:-500}"
  --ulimit memlock=-1 --ulimit nofile=524288:524288 --shm-size 16g
  --stop-signal SIGTERM --stop-timeout "$V17_STOP_TIMEOUT"
  --log-opt "max-size=$V17_DOCKER_LOG_MAX_SIZE"
  --log-opt "max-file=$V17_DOCKER_LOG_MAX_FILES"
  --label "app=inkling-tp8" --label "version=v17" --label "rank=$RANK"
  ${CPUSET_ARGS[@]+"${CPUSET_ARGS[@]}"} "${ENVS[@]}" "${MOUNTS[@]}" "$IMAGE" "${ARGS[@]}"
)

echo "[v17-launch][rank $RANK] model=$MODEL_DIR host=$BIND_HOST:${PORT:-8010} ip=$IP ifname=$RESOLVED_IFNAME hcas=$RESOLVED_HCAS gmu=$GMU maxlen=$MAXLEN dspark=${DSPARK:-0} sched=${INKLING_SCHED:-0}/${SCHEDULING_POLICY:-fcfs} cpuset=${INKLING_CPUSET:-all} patches=$PATCH_COUNT"

if [ "$DRY_RUN" = "1" ]; then
  echo "[v17-launch][DRY-RUN] Preflight PASSED. Command:"
  printf ' %q' "${DOCKER_CMD[@]}"
  echo
  exit 0
fi

mkdir -p "$HOME/.cache/inkling-jit" "$HOME/.triton" "$HOME/.nv" "$HOME/.cache/certhead" "$HOME/inkling-prof"

if docker ps -q --filter "name=^${CONTAINER_NAME}$" 2>/dev/null | grep -q .; then
  echo "[v17-launch][rank $RANK] Gracefully draining existing $CONTAINER_NAME (timeout=${V17_STOP_TIMEOUT}s)..."
  docker stop -t "$V17_STOP_TIMEOUT" "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

if [ "$V17_DETACH" = "1" ]; then
  exec "${DOCKER_CMD[@]}"
else
  _drain_container() {
    echo "[v17-launch][rank $RANK] Caught shutdown signal; draining $CONTAINER_NAME..." >&2
    docker stop -t "$V17_STOP_TIMEOUT" "$CONTAINER_NAME" >/dev/null 2>&1 || true
  }
  trap _drain_container TERM INT
  "${DOCKER_CMD[@]}" &
  child_pid=$!
  wait "$child_pid"
fi

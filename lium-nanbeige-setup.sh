#!/usr/bin/env bash
# Nanbeige4.3-33B daily driver on a rented GPU: llama.cpp's llama-server (official CUDA image) in Docker,
# OpenAI-compatible API with native tool calls, served on 0.0.0.0:PORT.
#   ./lium-nanbeige-setup.sh [PORT]              install / update / restart
#   CTX=131072 QUANT=Q4_K_M ./lium-nanbeige-setup.sh [PORT]   override the automatic choice
#   VERIFY_SHA256=1 ./lium-nanbeige-setup.sh [PORT]   also hash the whole model (slow; off by default)
#   ./lium-nanbeige-setup.sh [PORT] --verify       same thing, as a flag
#   QUANT=Q4_K_M ./lium-nanbeige-setup.sh [PORT]  better weights (192K ctx on 48 GB); 256K needs Q3_K_M
#   THINK_BUDGET=4096 TEMPERATURE=0.6 ./lium-nanbeige-setup.sh [PORT]  cap thinking / chat sampling
#   SPEC=off ./lium-nanbeige-setup.sh [PORT]     disable n-gram speculative decoding (on by default)
#   ./lium-nanbeige-setup.sh --uninstall [-y]    remove the server, image and the model files
#   The port is locked to the IP you SSH in from. ALLOW_IP=1.2.3.4,5.6.7.0/24 to choose, ALLOW_IP=any to open it.
#   ./lium-nanbeige-setup.sh --lock [PORT]       re-lock to your current IP (new home IP / after a reboot)
#
# Verified against the model card and the GGUF's own Jinja template (2026-11-17): the template enables
# thinking by default (enable_thinking=false injects an empty <think> block) and uses XML tool calls by
# default. The card recommends preserve_thinking=True for multi-turn tool use / code agents and False for
# general chat; sampling is temp 1.0 / top_p 0.95 for agentic work, 0.6 / 0.95 for reasoning and chat.
# This script is an agent daily driver, so it uses the agentic row; TEMPERATURE/TOP_P override the server
# default, and API clients can still set temperature/top_p per request. Native 512K context, 52 layers,
# 8 KV heads x 128 dims -> the K/V numbers below.
#
# Why Nanbeige4.3-33B: text-only, built for agentic coding and research, 512K native context. Independent AA numbers
# back its own claims: Intelligence Index 20.4 (top of the <=40B class), lowest hallucination rate in it (18%),
# Terminal-Bench 4 0.30 (Qwen3.8 27B: 0.06), Automation Bench 0.53.
# Why llama-server: it reads the model's own chat template (--jinja) and parses its XML tool calls (Qwen3-Coder
# format) and <think> blocks. Ollama maps a GGUF's Jinja template to one of its own built-in Go templates
# (HF/Ollama docs), so this model's custom XML tools / thinking scaffolding aren't preserved there.
#
# Memory: Nanbeige uses full attention in all 52 layers, so its K/V cache is large: ~110 MB per 1K tokens at q8_0
# (27.6 GiB at 256K). Recommended card: RTX A6000 48 GB -> Q3_K_M weights + q8_0 K/V at 256K (~45 GiB).
# 24 GB cards (3090/4090) top out around 64K. The script picks the longest context first, then the best weights,
# then f16 K/V over q8_0 when there is room. It never uses a 4-bit K/V cache (it hurts long-context recall).
# Trade-off on 48 GB: 256K forces Q3_K_M, which bartowski labels "low quality". QUANT=Q4_K_M (his recommended
# quant) gives 192K on this card; CTX=131072 gets Q6_K (best weights that fit at 128K).
set -e
export DEBIAN_FRONTEND=noninteractive

IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda-b12429      # llama.cpp b12429 (2026-11-16), CUDA 12.8
NAME=nanbeige
ALIAS=nanbeige-33b
REPO=https://huggingface.co/bartowski/Nanbeige_Nanbeige4.3-33B-GGUF/resolve/main
# bartowski's GGUFs, best first: "quant|bytes|sha256" (from the Hugging Face file list)
QUANTS=(
  "Q8_0|35195128960|8523452a975fa67d84a926c9a1bbdbb32b78b2a516dbf14d9360271fc53cd7f1"
  "Q6_K|29016857728|b11c9f21e69d896a1c20b12f88174f069e0e15623140733a49ff8e47c5541fbc"
  "Q5_K_M|25073368192|2b10206e3ba0a6a8f88b8c23fdbc49e9efccb3331564e711e4fc2901b18288cc"
  "Q4_K_M|20788825216|2cef79855ca47ee6571e166db32a07089e34f3cf78e0397b262d124ddb814c37"
  "IQ4_XS|18635041920|510cb2e07497ee3e9b7cb20aae3f50998adb29d95a71e363ba342544a47647a3"
  "IQ3_M|17802005632|7e4f92185c3b7917e68752364fbee2549b9a00b78fc3bbf31afa30952dd02d5e"
  "Q3_K_M|15919746176|671a48692387750eaa94f113f2635386d9d1c309b2d16c77b2890876d0e04e51"
  "IQ3_XS|15212665984|fdb7e52deb939c3f08e520c7dd968802c4cdb84c7d308637a989839d2916e1af"
  "IQ3_XXS|14489959552|5d959f2812f07632fc6e8de21373f90d12bda81d0178a2a8868fc3e6812a25da"
)
CTX_TIERS=(262144 196608 131072 98304 65536 49152 32768)
# K/V bytes per token: 52 layers x 8 KV heads x (128 K + 128 V); q8_0 = 1 byte + a 2-byte scale per 32
KV_F16_BYTES=$(( 52 * 8 * 256 * 2 ))           # 212992
KV_Q8_BYTES=$(( 52 * 8 * 256 * 34 / 32 ))      # 113152
OVERHEAD_MB=3584                               # CUDA context + compute buffers at long context + ~1.5 GB margin

# The data folder is bind-mounted into the container. On Lium, /root is an encrypted FUSE mount that the pod's
# Sysbox runtime can't bind into nested containers, so the first folder that mounts wins.
DIR_CANDIDATES=(/root/.nanbeige /var/lib/nanbeige /var/lib/docker/nanbeige-data)
mount_ok() {   # $1 = folder: can a container see a file in it?
  local new=0; [ -d "$1" ] || new=1
  mkdir -p "$1" && echo ok > "$1/.mount-probe"
  docker run --rm -v "$1:/p" busybox cat /p/.mount-probe 2>/dev/null | grep -qx ok
  local r=$?; rm -f "$1/.mount-probe"; [ "$new" = 1 ] && [ "$r" != 0 ] && rmdir "$1" 2>/dev/null; return $r
}

# --- Port lock: only your IP may reach the server (firewall rule, no VPN) ---
# Default: the IP of your current SSH session. ALLOW_IP=1.2.3.4,5.6.7.0/24 overrides, ALLOW_IP=any leaves it open.
# Docker-published ports skip the INPUT chain, so the rule lives in DOCKER-USER and matches the original
# (pre-NAT) port. iptables rules don't survive a reboot: rerun the script, or `--lock`, after one.
unlock() {   # $1 = INPUT|DOCKER-USER: remove every LIUM-LOCK jump there, and its chain
  command -v iptables >/dev/null || return 0
  local r
  iptables -S "$1" 2>/dev/null | grep -E -- '-j LIUM-LOCK-[0-9]+$' | sed 's/^-A /-D /' | while read -r r; do
    iptables $r; iptables -F "${r##* }" 2>/dev/null; iptables -X "${r##* }" 2>/dev/null || true
  done
}
lock_port() {   # $1 = external port, $2 = docker|host
  local port=$1 chain=LIUM-LOCK-$1 allow=${ALLOW_IP:-} ip parent=INPUT
  [ "$2" = docker ] && parent=DOCKER-USER
  if [ -z "$allow" ]; then
    allow=${SSH_CLIENT%% *}; [ -n "$allow" ] || allow=${SSH_CONNECTION%% *}
    if [ -z "$allow" ]; then read -rp "Public IP allowed to use port $port (blank = open to everyone): " allow; fi
    # A private source address means the provider NATs inbound traffic: every client would look the same.
    if [[ $allow =~ ^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.|100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.) ]]; then
      echo "WARNING: your SSH session comes from a private address ($allow), so this host can't see real client IPs."
      echo "Port lock left unchanged (none on a first install). Set ALLOW_IP=<your public IP> only if you know"
      echo "inbound traffic keeps its source IP here."
      return 0
    fi
  fi
  [ -n "$allow" ] && [ "$allow" != any ] || { unlock "$parent"; echo "Port $port is open to everyone (no ALLOW_IP)."; return 0; }
  for ip in ${allow//,/ }; do
    [[ $ip =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}(/[0-9]{1,2})?$ ]] || { echo "Invalid IP in ALLOW_IP: '$ip'"; exit 1; }
  done
  command -v iptables >/dev/null || apt-get install -y -qq iptables >/dev/null 2>&1 || true
  command -v iptables >/dev/null || { echo "iptables is missing; can't lock port $port."; exit 1; }
  unlock "$parent"
  iptables -N "$chain" 2>/dev/null || iptables -F "$chain"
  for ip in ${allow//,/ }; do iptables -A "$chain" -s "$ip" -j RETURN; done
  iptables -A "$chain" -s 127.0.0.0/8 -j RETURN
  iptables -A "$chain" -j DROP
  if [ "$2" = docker ]; then
    iptables -N DOCKER-USER 2>/dev/null || true
    iptables -C FORWARD -j DOCKER-USER 2>/dev/null || iptables -I FORWARD 1 -j DOCKER-USER
    iptables -I DOCKER-USER 1 -p tcp -m conntrack --ctorigdstport "$port" --ctdir ORIGINAL -j "$chain"
  else
    iptables -I INPUT 1 -p tcp --dport "$port" -j "$chain"
  fi
  echo "Port $port locked to: ${allow//,/, }   (check hits: iptables -L $chain -v -n)"
}

# --- Uninstall ---
if [ "$1" = "--uninstall" ]; then
  DIRS=(); for d in "${DIR_CANDIDATES[@]}"; do [ -d "$d" ] && DIRS+=("$d"); done
  USED=$(du -shc "${DIRS[@]}" 2>/dev/null | tail -1 | cut -f1)
  if [ "$2" != "-y" ]; then
    read -rp "Remove the Nanbeige server, its image and DELETE the model files in ${DIRS[*]:-(none)} (~${USED:-0})? [y/N] " a
    [[ $a == [yY]* ]] || { echo "Cancelled."; exit 0; }
  fi
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker rmi "$IMAGE" >/dev/null 2>&1 || true
  unlock DOCKER-USER
  rm -rf "${DIRS[@]}"
  echo "Nanbeige server removed, model files deleted (freed ~${USED:-0}). Docker itself is left installed."
  exit 0
fi

# --- Re-lock only: ./lium-nanbeige-setup.sh --lock [PORT]  (new home IP, or after a reboot; no restart) ---
if [ "$1" = "--lock" ]; then
  PORT=${2:-$(docker port "$NAME" 8080/tcp 2>/dev/null | awk -F: 'NR==1 {print $NF}')}
  [[ "$PORT" =~ ^[0-9]+$ ]] || { echo "No running server found; pass the port: $0 --lock PORT"; exit 1; }
  lock_port "$PORT" docker
  exit 0
fi

# --- GPU autodetect ---
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found - no NVIDIA GPU/driver?"; exit 1; }
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits)
GPU_COUNT=$(echo "$GPU_INFO" | wc -l)
GPU_NAME=$(echo "$GPU_INFO" | head -1 | cut -d, -f1)
FREE_MB=$(echo "$GPU_INFO" | awk -F', ' '{s+=$3} END{print s}')   # llama.cpp splits layers over all cards
BUDGET_MB=$(( FREE_MB - OVERHEAD_MB - (GPU_COUNT - 1) * 1024 ))
echo "Detected ${GPU_COUNT}x $GPU_NAME, $((FREE_MB/1024)) GB free VRAM"

# Validate overrides here, so a typo doesn't turn into an arithmetic error or a misleading "doesn't fit" message.
if [ -n "${CTX:-}" ]; then
  [[ "$CTX" =~ ^[0-9]+$ ]] && (( CTX > 0 )) || { echo "CTX must be a positive number of tokens"; exit 1; }
fi
if [ -n "${QUANT:-}" ]; then
  qok=0; for e in "${QUANTS[@]}"; do [ "${e%%|*}" = "$QUANT" ] && qok=1; done
  (( qok )) || { echo "Unknown QUANT '$QUANT' (see the QUANTS table in $0)"; exit 1; }
fi

# --- Pick context, weights and K/V type: longest context first, then best weights, then f16 over q8_0 ---
need_mb() { echo $(( $1 / 1048576 + $2 * $3 / 1048576 )); }   # bytes, ctx, kv bytes/token -> MB
PICK=""
for ctx in ${CTX:-${CTX_TIERS[*]}}; do
  for e in "${QUANTS[@]}"; do
    IFS='|' read -r q bytes sha <<< "$e"
    [ -n "$QUANT" ] && [ "$q" != "$QUANT" ] && continue
    for kv in f16 q8_0; do
      kvb=$KV_Q8_BYTES; [ "$kv" = f16 ] && kvb=$KV_F16_BYTES
      if (( $(need_mb "$bytes" "$ctx" "$kvb") <= BUDGET_MB )); then PICK="$q|$bytes|$sha|$ctx|$kv"; break 3; fi
    done
  done
done
if [ -z "$PICK" ]; then
  echo "Nanbeige doesn't fit in $((FREE_MB/1024)) GB of VRAM${CTX:+ at CTX=$CTX}${QUANT:+ with QUANT=$QUANT}."
  echo "Smallest option is IQ3_XXS with a q8_0 K/V cache. Drop the CTX/QUANT override, or use a 24 GB+ card"
  echo "for 32-64K / an RTX A6000 (48 GB) for 256K."
  exit 1
fi
IFS='|' read -r QUANT_SEL BYTES SHA CTX_SEL KV_SEL <<< "$PICK"
KVB=$KV_Q8_BYTES; [ "$KV_SEL" = f16 ] && KVB=$KV_F16_BYTES
echo "Plan: $QUANT_SEL weights ($((BYTES/1073741824)) GB) + $KV_SEL K/V at $((CTX_SEL/1024))K context = ~$(( $(need_mb "$BYTES" "$CTX_SEL" "$KVB") / 1024 + OVERHEAD_MB / 1024 )) GB of $((FREE_MB/1024)) GB"
if [ -z "$CTX" ] && (( CTX_SEL < 262144 )); then
  if [ -n "$QUANT" ]; then echo "Note: $QUANT at 256K doesn't fit this card; longest context for it is $((CTX_SEL/1024))K."
  else echo "Note: 256K needs ~45 GB with Q3_K_M (RTX A6000). This card gets $((CTX_SEL/1024))K."; fi
fi

# Port: pass as an argument (plus optional --verify), or get prompted
case "${VERIFY_SHA256:-0}" in 1|true|yes|on) VERIFY_SHA256=1 ;; *) VERIFY_SHA256=0 ;; esac
PORT=""
for arg in "$@"; do
  case "$arg" in
    --verify) VERIFY_SHA256=1 ;;
    -*) echo "Unknown option: $arg"; exit 1 ;;
    *) [ -z "$PORT" ] || { echo "Only one port argument is accepted: '$arg'"; exit 1; }; PORT=$arg ;;
  esac
done
if [ -z "$PORT" ]; then
  read -rp "External port for this machine [20001]: " PORT
  PORT=${PORT:-20001}
fi
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1 && PORT <= 65535 )) || { echo "Invalid port: '$PORT'"; exit 1; }

# --- Optional tuning (defaults = the model card's agentic row) ---
# TEMPERATURE/TOP_P are server defaults only; an API request can override them per call. THINK_BUDGET maps to
# llama.cpp's --reasoning-budget (N>0 caps thinking tokens, 0 ends it immediately, -1/unset = unlimited).
# It is optional on purpose: capping thinking trades answer quality on hard tasks for lower latency.
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
for v in "$TEMPERATURE" "$TOP_P"; do
  [[ "$v" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "TEMPERATURE/TOP_P must be numbers (got '$v')"; exit 1; }
done
if [ -n "${THINK_BUDGET:-}" ]; then
  [[ "$THINK_BUDGET" =~ ^[0-9]+$ ]] || { echo "THINK_BUDGET must be a non-negative integer of tokens"; exit 1; }
fi

# --- Prerequisites: only touch apt when something is actually missing ---
# A restart on a warm box shouldn't wait for apt-get update + a package install every time. btop/bmon are
# conveniences only: some minimal cloud images don't carry them (the same goes for whiptail elsewhere).
APT_PKGS=()
command -v curl   >/dev/null || APT_PKGS+=(curl ca-certificates)
command -v gpg    >/dev/null || APT_PKGS+=(gnupg)
command -v docker >/dev/null || APT_PKGS+=(docker.io)
if (( ${#APT_PKGS[@]} )) || ! command -v nvidia-ctk >/dev/null; then
  apt-get update -qq
  if (( ${#APT_PKGS[@]} )); then apt-get install -y -qq "${APT_PKGS[@]}" >/dev/null; fi
  apt-get install -y -qq btop bmon >/dev/null 2>&1 || echo "Note: optional monitors (btop/bmon) unavailable, continuing."
fi

# --- Docker + NVIDIA container toolkit (rented pods often have no systemd: start dockerd directly) ---
if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit >/dev/null
  nvidia-ctk runtime configure --runtime=docker >/dev/null
  RESTART_DOCKER=1
fi
if [ -d /run/systemd/system ]; then
  [ -n "$RESTART_DOCKER" ] && systemctl restart docker || systemctl start docker
else
  if [ -n "$RESTART_DOCKER" ] && [ -f /var/run/docker.pid ]; then kill "$(cat /var/run/docker.pid)" 2>/dev/null || true; sleep 3; fi
  if ! docker info >/dev/null 2>&1; then
    nohup dockerd >/var/log/dockerd.log 2>&1 &
    for i in $(seq 30); do docker info >/dev/null 2>&1 && break; sleep 1; done
  fi
fi
docker info >/dev/null 2>&1 || { echo "Docker cannot run on this machine. Rent a VM or a Docker-in-Docker pod."; tail -5 /var/log/dockerd.log 2>/dev/null; exit 1; }
DDIR=""
for d in "${DIR_CANDIDATES[@]}"; do if mount_ok "$d"; then DDIR=$d; break; fi; done
[ -n "$DDIR" ] || { echo "Docker can't mount any folder into containers here. Use a VM or a privileged pod."; exit 1; }
echo "Docker OK: $(docker --version). Data folder: $DDIR"

# --- Model download (resumable) ---
# A full sha256 pass re-reads the entire 14-35 GB file: minutes of extra disk I/O after a long download.
# Off by default. Instead we do O(1) checks that catch the realistic failures - a truncated/interrupted
# download (exact byte count) or an HTML error page instead of a model (GGUF magic). VERIFY_SHA256=1 or
# --verify adds the original full-file hash for people who want it.
FILE="Nanbeige_Nanbeige4.3-33B-$QUANT_SEL.gguf"
PART="$DDIR/models/$FILE.part"
MODEL="$DDIR/models/$FILE"
part_size() { stat -c %s "$PART" 2>/dev/null || echo 0; }
mkdir -p "$DDIR/models"
# Pull the ~3 GB server image while the model downloads instead of after it.
docker pull -q "$IMAGE" >/dev/null 2>&1 & PULL_PID=$!

if [ -f "$MODEL" ]; then
  echo "Model already present: $MODEL (no download, no hash)"
else
  # One HEAD request: HF publishes the blob's exact size and sha256 (x-linked-size / x-linked-etag), so we
  # can confirm the QUANTS entry is still current without reading gigabytes. Best-effort: if HEAD fails or
  # the headers change, fall back to the hardcoded table.
  EXPECT_SHA=$SHA
  META=$(curl -fsSIL --connect-timeout 10 --max-time 30 "$REPO/$FILE" 2>/dev/null | tr -d '\r' || true)
  if [ -n "$META" ]; then
    REMOTE_SIZE=$(echo "$META" | awk -F': ' 'tolower($1)=="x-linked-size" {print $2; exit}')
    REMOTE_SHA=$(echo "$META" | awk -F': ' 'tolower($1)=="x-linked-etag" {gsub(/"/,"",$2); print $2; exit}')
    if [ -n "$REMOTE_SIZE" ] && [ "$REMOTE_SIZE" != "$BYTES" ]; then
      echo "HF reports $REMOTE_SIZE bytes for $FILE but this script expects $BYTES (repacked file?)."
      echo "Update the QUANTS table in $0 before retrying."; exit 1
    fi
    if [ -n "$REMOTE_SHA" ] && [ "$REMOTE_SHA" != "$SHA" ]; then
      echo "Note: HF's sha256 for $FILE differs from the QUANTS table; using HF's value."
      EXPECT_SHA=$REMOTE_SHA
    fi
  fi

  # -C - resumes a partial file, but curl errors (33/416) if the .part is already complete: reuse it instead.
  if [ -f "$PART" ] && (( $(part_size) > BYTES )); then
    echo "Removing an oversized partial download ($(part_size) bytes > $BYTES)."; rm -f "$PART"
  fi
  if [ -f "$PART" ] && (( $(part_size) == BYTES )); then
    echo "Reusing the completed download $PART"
  else
    # Fail now, not 20 minutes into the download.
    NEED_MB=$(( (BYTES - $(part_size)) / 1048576 + 1024 ))
    FREE_DISK_MB=$(df -Pm "$DDIR/models" | awk 'NR==2 {print $4}')
    if [ -n "$FREE_DISK_MB" ] && (( FREE_DISK_MB < NEED_MB )); then
      echo "Not enough disk: $FREE_DISK_MB MB free in $DDIR/models, the download needs ~$NEED_MB MB."
      echo "Delete other quants there (ls $DDIR/models) or rent a box with more disk."; exit 1
    fi
    [ -f "$PART" ] && echo "Resuming $PART from $(part_size) bytes..."
    echo "Downloading $FILE ($((BYTES/1073741824)) GB)..."
    # --retry-all-errors also retries mid-transfer drops (curl >= 7.71); -C - makes each retry a resume.
    RETRY=(--retry 5 --retry-delay 5)
    if curl --help all 2>/dev/null | grep -q -- '--retry-all-errors'; then RETRY+=(--retry-all-errors); fi
    curl -fL "${RETRY[@]}" --connect-timeout 20 -C - --progress-bar -o "$PART" "$REPO/$FILE"
  fi

  SIZE=$(part_size)
  if (( SIZE != BYTES )); then
    echo "Incomplete download: $SIZE of $BYTES bytes in $PART. Rerun to resume."
    echo "If it never grows, the QUANTS size is stale (check the Hugging Face file list)."; exit 1
  fi
  if [ "$(head -c 4 "$PART")" != GGUF ]; then
    echo "$PART doesn't start with the GGUF magic (bad download?); delete it and retry."; exit 1
  fi
  if [ "$VERIFY_SHA256" = 1 ]; then
    echo "Checking sha256 (VERIFY_SHA256=1; this reads the whole file)..."
    echo "$EXPECT_SHA  $PART" | sha256sum -c --status || { echo "sha256 mismatch: delete $PART and retry"; exit 1; }
  else
    echo "Verified $SIZE bytes + GGUF header; sha256 skipped (use VERIFY_SHA256=1 / --verify to check)."
  fi
  mv "$PART" "$MODEL"
fi
# other quants from earlier runs: keep (switching back is instant), but say where they are
OTHERS=$(ls "$DDIR/models" | grep -vxF "$FILE" || true); [ -n "$OTHERS" ] && echo "Other model files in $DDIR/models: $OTHERS"

# --- Start (or replace) the server ---
wait "$PULL_PID" || docker pull -q "$IMAGE" >/dev/null   # retry in the foreground so a pull error is shown
docker rm -f "$NAME" >/dev/null 2>&1 || true
lock_port "$PORT" docker   # before the port opens
# Prompt cache in host RAM: with one slot, any side request (chat titles, a subagent) overwrites the main
# conversation's K/V. llama-server parks the old state in RAM first, but its default cap is 8 GiB, and at q8_0
# that holds only ~75K tokens of this model. A longer conversation would be re-read from scratch after every
# side request (minutes at 200K). Give the cache half of the available host RAM instead.
AVAIL_MB=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
CACHE_RAM_MB=$(( AVAIL_MB / 2 )); (( CACHE_RAM_MB < 8192 )) && CACHE_RAM_MB=8192
SERVER_ARGS=(--cache-ram "$CACHE_RAM_MB")
# -ngl 999 + --fit off: every layer on the GPU, or fail loudly (never a silent CPU offload or a smaller context).
# (--fit defaults to on in b12429 and would silently shrink an unset -c down to --fit-ctx, min 4096; --fit off
# plus an explicit -c keeps the context we planned. -ctv q8_0 also requires Flash Attention, hence -fa on.)
# -np 1: one slot, so the whole context is one conversation's; requests queue. --cache-reuse: reuse cached K/V
# chunks when an agent edits earlier messages. --jinja is on by default in b12429; kept explicit because native
# XML tool calls and the <think> template depend on it. preserve_thinking=true + temp 1.0/top_p 0.95: the model
# card's agentic settings, as verified above.
if [ -n "${THINK_BUDGET:-}" ]; then SERVER_ARGS+=(--reasoning-budget "$THINK_BUDGET"); fi
# Self-speculative decoding from n-grams already in the context: no draft model, no extra VRAM, output is
# identical (drafts are verified by the model). It pays off when the model re-types text it has seen: editing
# files, rewriting code, repeating its thinking in the answer. Nanbeige has no MTP head, so this is the free
# speed-up available. Values: llama.cpp's docs/speculative.md example, with shorter drafts for a dense model.
# SPEC=off disables it.
if [ "${SPEC:-ngram-mod}" != off ]; then
  SERVER_ARGS+=(--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 16 --spec-ngram-mod-n-max 48)
fi
docker run -d --name "$NAME" --restart unless-stopped --gpus all \
  -v "$DDIR/models:/models:ro" -p "0.0.0.0:$PORT:8080" "$IMAGE" \
  -m "/models/$FILE" --alias "$ALIAS" --host 0.0.0.0 --port 8080 \
  --jinja --chat-template-kwargs '{"preserve_thinking": true}' \
  -c "$CTX_SEL" -np 1 -ngl 999 --fit off -fa on -ctk "$KV_SEL" -ctv "$KV_SEL" \
  --cache-reuse 256 --temp "$TEMPERATURE" --top-p "$TOP_P" "${SERVER_ARGS[@]}" >/dev/null

echo "Loading the model (1-3 min)..."
for i in $(seq 300); do
  # a crash at load (e.g. out of VRAM) shows as a stopped container or, with the restart policy, a restart count
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME" || [ "$(docker inspect -f '{{.RestartCount}}' "$NAME" 2>/dev/null)" != 0 ]; then
    docker update --restart=no "$NAME" >/dev/null 2>&1 || true
    echo "The server stopped. Last log lines:"; docker logs --tail 25 "$NAME" 2>&1
    echo "If it ran out of memory, rerun with a smaller context, e.g.: CTX=$(( CTX_SEL / 2 )) $0 $PORT"; exit 1
  fi
  curl -fs --max-time 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
  sleep 2
done
curl -fs --max-time 5 "http://127.0.0.1:$PORT/health" | grep -q '"ok"' || { echo "Server not healthy after 10 min:"; docker logs --tail 25 "$NAME" 2>&1; exit 1; }

# --- Smoke test: a tool call must come back as an OpenAI tool_calls entry ---
echo "Testing tool calling..."
R=$(curl -fs --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$ALIAS"'", "max_tokens": 4096,
  "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}' || true)
if echo "$R" | grep -q '"tool_calls"' && echo "$R" | grep -q 'get_weather'; then TOOLS="OK"; else TOOLS="NOT DETECTED - check: docker logs $NAME"; fi
VRAM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tr '\n' ' ')

echo
echo "Done. Nanbeige4.3-33B $QUANT_SEL is serving on 0.0.0.0:$PORT (OpenAI API)"
echo "  Context:    $CTX_SEL tokens, $KV_SEL K/V cache, all layers on the GPU (VRAM used: $VRAM)"
echo "  Sampling:   temp $TEMPERATURE / top_p $TOP_P${THINK_BUDGET:+, thinking capped at $THINK_BUDGET tokens}"
echo "  Speculation: ${SPEC:-ngram-mod} (n-gram drafts from the context; SPEC=off to disable)"
echo "  Tool calls: $TOOLS"
echo "  Base URL:   http://<this-host>:$PORT/v1     model: $ALIAS     api key: any string"
echo "  Clients:    set the context window to $CTX_SEL; one request at a time (others queue)"
echo "  Prompt cache: up to $((CACHE_RAM_MB/1024)) GB of host RAM keeps conversations across side requests"
echo "  Logs:       docker logs -f $NAME          Monitor: btop / nvidia-smi"

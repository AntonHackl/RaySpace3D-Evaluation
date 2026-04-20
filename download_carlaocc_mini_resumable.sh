#!/usr/bin/env bash
set -u

REPO_ID="fengyi233/CarlaOcc"
SUBSET="CarlaOccV1_mini"
LOCAL_DIR="./carlaocc_dataset"
MAX_WORKERS="2"
INITIAL_BACKOFF_SECONDS="330"
MAX_BACKOFF_SECONDS="900"
LOG_DIR="./tmp"
LOG_FILE="${LOG_DIR}/carlaocc_download_last.log"

usage() {
  cat <<'EOF'
Usage:
  ./download_carlaocc_mini_resumable.sh [options]

Options:
  --repo-id <id>            Hugging Face repo id (default: fengyi233/CarlaOcc)
  --subset <name>           Dataset subset prefix (default: CarlaOccV1_mini)
  --local-dir <path>        Local destination directory (default: ./carlaocc_dataset)
  --max-workers <n>         Parallel download workers (default: 2)
  --initial-backoff <sec>   First retry wait on 429 (default: 330)
  --max-backoff <sec>       Maximum retry wait on 429 (default: 900)
  --help                    Show this help

Auth:
  Set one of these environment variables before running:
  - HF_TOKEN
  - HF_READING_DATASETS_TOKEN

Example:
  export HF_TOKEN=hf_xxx
  ./download_carlaocc_mini_resumable.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-id)
      REPO_ID="$2"
      shift 2
      ;;
    --subset)
      SUBSET="$2"
      shift 2
      ;;
    --local-dir)
      LOCAL_DIR="$2"
      shift 2
      ;;
    --max-workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    --initial-backoff)
      INITIAL_BACKOFF_SECONDS="$2"
      shift 2
      ;;
    --max-backoff)
      MAX_BACKOFF_SECONDS="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if command -v hf >/dev/null 2>&1; then
  DOWNLOAD_CMD=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  DOWNLOAD_CMD=(huggingface-cli download)
else
  echo "Error: Neither 'hf' nor 'huggingface-cli' was found in PATH." >&2
  exit 127
fi

TOKEN="${HF_TOKEN:-${HF_READING_DATASETS_TOKEN:-}}"
if [[ -z "$TOKEN" ]]; then
  echo "Error: Set HF_TOKEN or HF_READING_DATASETS_TOKEN before running." >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
mkdir -p "$LOCAL_DIR"

attempt=1
backoff="$INITIAL_BACKOFF_SECONDS"

while true; do
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Attempt ${attempt}: downloading ${SUBSET} from ${REPO_ID}"

  "${DOWNLOAD_CMD[@]}" "$REPO_ID" \
    --repo-type dataset \
    --include "${SUBSET}/**" \
    --local-dir "$LOCAL_DIR" \
    --max-workers "$MAX_WORKERS" \
    --token "$TOKEN" 2>&1 | tee "$LOG_FILE"

  status=${PIPESTATUS[0]}
  if [[ "$status" -eq 0 ]]; then
    subset_path="${LOCAL_DIR}/${SUBSET}"
    if [[ -d "$subset_path" ]]; then
      subset_count=$(find "$subset_path" -type f | wc -l)
    else
      subset_count=0
    fi

    if [[ "$subset_count" -gt 0 ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Download completed successfully."
      break
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No files found under ${subset_path}." >&2
    echo "The subset pattern may be invalid or not present in the remote repo." >&2
    exit 3
  fi

  if grep -Eqi '429|Too Many Requests|rate limit' "$LOG_FILE"; then
    jitter=$((RANDOM % 30))
    wait_seconds=$((backoff + jitter))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Hit rate limit (429). Waiting ${wait_seconds}s, then resuming."
    sleep "$wait_seconds"

    if (( backoff < MAX_BACKOFF_SECONDS )); then
      backoff=$((backoff * 2))
      if (( backoff > MAX_BACKOFF_SECONDS )); then
        backoff="$MAX_BACKOFF_SECONDS"
      fi
    fi

    attempt=$((attempt + 1))
    continue
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Download failed with non-rate-limit error (exit ${status})." >&2
  echo "Inspect $LOG_FILE for details."
  exit "$status"
done

echo
if [[ -d "${LOCAL_DIR}/${SUBSET}" ]]; then
  count=$(find "${LOCAL_DIR}/${SUBSET}" -type f | wc -l)
  size=$(du -sh "${LOCAL_DIR}/${SUBSET}" | awk '{print $1}')
  echo "Files downloaded in ${LOCAL_DIR}/${SUBSET}: ${count}"
  echo "Disk usage: ${size}"
fi

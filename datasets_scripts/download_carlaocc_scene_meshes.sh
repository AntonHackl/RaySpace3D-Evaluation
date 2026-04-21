#!/usr/bin/env bash
set -u

TOWN="Town01_Opt"
MAX_WORKERS="1"
INITIAL_BACKOFF_SECONDS="330"
MAX_BACKOFF_SECONDS="900"
LOCAL_DIR="$(dirname "$0")/carlaocc_dataset"

usage() {
  cat <<'EOF'
Usage:
  ./download_carlaocc_scene_meshes.sh [options]

Options:
  --town <TownXX_Opt>       Town folder in SceneMeshes (default: Town01_Opt)
  --max-workers <n>         Passed to downloader script (default: 1)
  --initial-backoff <sec>   First retry wait on 429 (default: 330)
  --max-backoff <sec>       Maximum retry wait on 429 (default: 900)
  --local-dir <path>        Local destination directory (default: ./carlaocc_dataset)
  --help                    Show this help

Auth:
  Set one of these environment variables before running:
  - HF_TOKEN
  - HF_READING_DATASETS_TOKEN

What gets downloaded:
  - CarlaOccV1/SceneMeshes/<town>
  - CarlaOccV1/SceneMeshes/fg_actors
  - CarlaOccV1/SceneMeshes/fg_actor_occ

Example:
  set -a; source ./.env; set +a
  export HF_TOKEN="$HF_READING_DATASETS_TOKEN"
  ./download_carlaocc_scene_meshes.sh --town Town01_Opt
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --town)
      TOWN="$2"
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
    --local-dir)
      LOCAL_DIR="$2"
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

if [[ ! -f "$(dirname "$0")/download_carlaocc_mini_resumable.sh" ]]; then
  echo "Error: Missing required script $(dirname "$0")/download_carlaocc_mini_resumable.sh" >&2
  exit 2
fi

if [[ ! -x "$(dirname "$0")/download_carlaocc_mini_resumable.sh" ]]; then
  chmod +x "$(dirname "$0")/download_carlaocc_mini_resumable.sh"
fi

TOKEN="${HF_TOKEN:-${HF_READING_DATASETS_TOKEN:-}}"
if [[ -z "$TOKEN" ]]; then
  echo "Error: Set HF_TOKEN or HF_READING_DATASETS_TOKEN before running." >&2
  exit 2
fi

mkdir -p "$(dirname "$0")/../tmp"
TREE_FILE="$(dirname "$0")/../tmp/carlaocc_tree_preflight.json"
curl -s -H "Authorization: Bearer ${TOKEN}" \
  "https://huggingface.co/api/datasets/fengyi233/CarlaOcc/tree/main?recursive=true" \
  > "$TREE_FILE"

if ! grep -Eqi '"path":"[^"]*(SceneMeshes|mesh|meshes)[^"]*"' "$TREE_FILE"; then
  echo "Error: No SceneMeshes/mesh paths are currently published in the HF dataset tree." >&2
  echo "This dataset revision exposes modality archives/directories like all_depth, all_lidar, all_rgb, all_semantics, etc." >&2
  echo "If you need scene meshes, request the current mesh path layout from the dataset maintainers or another release." >&2
  exit 4
fi

subsets=(
  "CarlaOccV1/SceneMeshes/${TOWN}"
  "CarlaOccV1/SceneMeshes/fg_actors"
  "CarlaOccV1/SceneMeshes/fg_actor_occ"
)

for subset in "${subsets[@]}"; do
  echo
  echo "=== Downloading ${subset} ==="
  ./download_carlaocc_mini_resumable.sh \
    --subset "$subset" \
    --max-workers "$MAX_WORKERS" \
    --initial-backoff "$INITIAL_BACKOFF_SECONDS" \
    --max-backoff "$MAX_BACKOFF_SECONDS" \
    --local-dir "$LOCAL_DIR"
done

echo
base_path="${LOCAL_DIR}/CarlaOccV1/SceneMeshes"
if [[ -d "$base_path" ]]; then
  echo "SceneMeshes summary in ${base_path}:"
  find "$base_path" -mindepth 1 -maxdepth 1 -type d -print
fi

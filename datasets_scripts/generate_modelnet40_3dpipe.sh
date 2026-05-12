#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELNET_VERSION="${MODELNET_VERSION:-1}"
MODELNET_ROOT="${MODELNET_ROOT:-$ROOT_DIR/datasets_scripts/modelnet_data/${MODELNET_VERSION}/ModelNet40}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/datasets_scripts/modelnet_data/generated_3dpipe}"
TMP_LAYOUT_ROOT="${TMP_LAYOUT_ROOT:-$ROOT_DIR/datasets_scripts/modelnet_data/.tmp_modelnet_layout}"
SIM_BIN="${SIM_BIN:-$ROOT_DIR/3dpipe/src/build/simulator_modelnet}"
SANITIZE_OFF="${SANITIZE_OFF:-1}"

# simulator_modelnet currently replicates each selected base mesh 100x.
REPEAT_FACTOR=100

# We size by fraction-of-train (proxy), based on the expected full train output size.
TARGET_TRAIN_GB="${TARGET_TRAIN_GB:-5}"
ASSUMED_FULL_TRAIN_GB="${ASSUMED_FULL_TRAIN_GB:-6}"
DATASET_A_SEED="${DATASET_A_SEED:-101}"
DATASET_B_SEED="${DATASET_B_SEED:-202}"

if [[ ! -d "$MODELNET_ROOT" ]]; then
    echo "ModelNet root not found: $MODELNET_ROOT"
    echo "Expected a category structure like <category>/train and <category>/test."
    exit 1
fi

if [[ ! -x "$SIM_BIN" ]]; then
    echo "simulator_modelnet not found or not executable: $SIM_BIN"
    echo "Build it first via: ./build_all.sh --only 3dpipe"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TMP_LAYOUT_ROOT"

ALL_TRAIN_LIST="$TMP_LAYOUT_ROOT/all_train_files.txt"
SELECTED_A_LIST="$TMP_LAYOUT_ROOT/selected_train_a.txt"
SELECTED_B_LIST="$TMP_LAYOUT_ROOT/selected_train_b.txt"

echo "Scanning train OFF files under: $MODELNET_ROOT"
find "$MODELNET_ROOT" -path '*/train/*.off' -type f | sort > "$ALL_TRAIN_LIST"
train_count="$(wc -l < "$ALL_TRAIN_LIST")"

if [[ "$train_count" -eq 0 ]]; then
    echo "No train OFF files found under: $MODELNET_ROOT"
    exit 1
fi

target_fraction="$(awk -v t="$TARGET_TRAIN_GB" -v f="$ASSUMED_FULL_TRAIN_GB" 'BEGIN {x=t/f; if (x>1) x=1; if (x<0.01) x=0.01; printf "%.6f", x}')"
target_count="$(awk -v n="$train_count" -v p="$target_fraction" 'BEGIN {c=int(n*p+0.5); if (c<1) c=1; print c}')"

select_list_with_seed() {
    local source_list="$1"
    local out_list="$2"
    local seed="$3"
    local count="$4"
    awk -v s="$seed" 'BEGIN{srand(s)} {print rand() "\t" $0}' "$source_list" \
        | sort -n -k1,1 \
        | awk -v c="$count" 'NR<=c' \
        | cut -f2- > "$out_list"
}

echo "Selecting train subset A (seed=$DATASET_A_SEED, count=$target_count)..."
select_list_with_seed "$ALL_TRAIN_LIST" "$SELECTED_A_LIST" "$DATASET_A_SEED" "$target_count"
echo "Selecting train subset B (seed=$DATASET_B_SEED, count=$target_count)..."
select_list_with_seed "$ALL_TRAIN_LIST" "$SELECTED_B_LIST" "$DATASET_B_SEED" "$target_count"

build_stage_from_list() {
    local list_file="$1"
    local stage_dir="$2"
    rm -rf "$stage_dir"
    mkdir -p "$stage_dir"

    local src category dst_category dst_file
    while IFS= read -r src; do
        category="$(basename "$(dirname "$(dirname "$src")")")"
        dst_category="$stage_dir/$category/test"
        mkdir -p "$dst_category"
        dst_file="$dst_category/$(basename "$src")"
        ln -s "$src" "$dst_file"
    done < "$list_file"
}

sanitize_stage_meshes() {
    local in_stage="$1"
    local out_stage="$2"
    rm -rf "$out_stage"
    mkdir -p "$out_stage"

    python3 - "$in_stage" "$out_stage" <<'PY'
import os
import sys
from pathlib import Path

in_stage = Path(sys.argv[1])
out_stage = Path(sys.argv[2])

def parse_off(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines or lines[0] != "OFF":
        return None
    if len(lines) < 2:
        return None
    try:
        nv, nf, _ = map(int, lines[1].split()[:3])
    except Exception:
        return None
    if len(lines) < 2 + nv + nf:
        return None
    verts = []
    for i in range(nv):
        parts = lines[2 + i].split()
        if len(parts) < 3:
            return None
        try:
            verts.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except Exception:
            return None

    faces = []
    base = 2 + nv
    for i in range(nf):
        parts = lines[base + i].split()
        if not parts:
            continue
        try:
            k = int(parts[0])
        except Exception:
            continue
        if k < 3 or len(parts) < 1 + k:
            continue
        try:
            idxs = [int(x) for x in parts[1:1 + k]]
        except Exception:
            continue
        if any(j < 0 or j >= nv for j in idxs):
            continue
        # Triangulate polygons by fan to make downstream processing safer.
        for t in range(1, k - 1):
            a, b, c = idxs[0], idxs[t], idxs[t + 1]
            if a != b and b != c and a != c:
                faces.append((a, b, c))

    if not faces:
        return None

    used = sorted({j for tri in faces for j in tri})
    remap = {old: new for new, old in enumerate(used)}
    new_verts = [verts[j] for j in used]
    new_faces = [(remap[a], remap[b], remap[c]) for (a, b, c) in faces]
    return new_verts, new_faces

def write_off(path: Path, verts, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("OFF\n")
        f.write(f"{len(verts)} {len(faces)} 0\n")
        for x, y, z in verts:
            f.write(f"{x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in faces:
            f.write(f"3 {a} {b} {c}\n")

total = 0
ok = 0
skipped = 0
for src in sorted(in_stage.glob("*/*/*.off")):
    total += 1
    rel = src.relative_to(in_stage)
    dst = out_stage / rel
    parsed = parse_off(src)
    if parsed is None:
        skipped += 1
        continue
    verts, faces = parsed
    write_off(dst, verts, faces)
    ok += 1

print(f"[sanitize] total={total} ok={ok} skipped={skipped}")
if ok == 0:
    sys.exit(2)
PY
}

run_stage() {
    local stage_dir="$1"
    local name="$2"
    local out_dir="$OUTPUT_DIR/$name"
    local out_file="$out_dir/ModelNet40_${name}_rep100_target${TARGET_TRAIN_GB}gb.dt"

    mkdir -p "$out_dir"
    rm -f "$out_dir/ModelNet_test.dt" "$out_file"

    echo "Running simulator_modelnet for $name"
    (
        cd "$out_dir"
        "$SIM_BIN" -d "$stage_dir"
    )

    if [[ ! -f "$out_dir/ModelNet_test.dt" ]]; then
        echo "Expected simulator output not found: $out_dir/ModelNet_test.dt"
        exit 1
    fi
    mv "$out_dir/ModelNet_test.dt" "$out_file"
    echo "Wrote: $out_file"
}

echo "Base ModelNet40 train OFF meshes: $train_count"
echo "Target size: ~${TARGET_TRAIN_GB}GB (assumed full train ~= ${ASSUMED_FULL_TRAIN_GB}GB)"
echo "Using fraction: $target_fraction -> selected train meshes per dataset: $target_count"
echo "Expected objects per dataset after replication x${REPEAT_FACTOR}: $((target_count * REPEAT_FACTOR))"

STAGE_A="$TMP_LAYOUT_ROOT/train_a_stage"
STAGE_B="$TMP_LAYOUT_ROOT/train_b_stage"
STAGE_A_CLEAN="$TMP_LAYOUT_ROOT/train_a_stage_clean"
STAGE_B_CLEAN="$TMP_LAYOUT_ROOT/train_b_stage_clean"

build_stage_from_list "$SELECTED_A_LIST" "$STAGE_A"
echo "Prepared staged layout: $STAGE_A"
build_stage_from_list "$SELECTED_B_LIST" "$STAGE_B"
echo "Prepared staged layout: $STAGE_B"

if [[ "$SANITIZE_OFF" == "1" ]]; then
    echo "Sanitizing OFF meshes for train_a..."
    sanitize_stage_meshes "$STAGE_A" "$STAGE_A_CLEAN"
    echo "Sanitizing OFF meshes for train_b..."
    sanitize_stage_meshes "$STAGE_B" "$STAGE_B_CLEAN"
    run_stage "$STAGE_A_CLEAN" "train_a"
    run_stage "$STAGE_B_CLEAN" "train_b"
else
    run_stage "$STAGE_A" "train_a"
    run_stage "$STAGE_B" "train_b"
fi

rm -rf "$TMP_LAYOUT_ROOT"
echo "Done. Generated files under: $OUTPUT_DIR"

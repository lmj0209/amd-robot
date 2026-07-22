#!/usr/bin/env bash
# Run the pre-registered randomized Push candidate and its gated matrices.

set -euo pipefail

usage() {
  echo "usage: $0 --run-root ABSOLUTE_PATH --expected-commit FULL_SHA" >&2
}

run_root=""
expected_commit=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root)
      run_root="${2:-}"
      shift 2
      ;;
    --expected-commit)
      expected_commit="${2:-}"
      shift 2
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$run_root" || -z "$expected_commit" ]]; then
  usage
  exit 2
fi
if [[ "$run_root" != /* ]]; then
  echo "run root must be an absolute path: $run_root" >&2
  exit 2
fi
if [[ ! "$expected_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "expected commit must be a full lowercase Git SHA" >&2
  exit 2
fi
if [[ -e "$run_root" ]]; then
  echo "refusing to overwrite run root: $run_root" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${RGC_PYTHON:-/workspace/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "RGC Python is not executable: $python_bin" >&2
  exit 2
fi

actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "commit mismatch: expected=$expected_commit actual=$actual_commit" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "refusing candidate run from a dirty worktree" >&2
  exit 2
fi

plan_path="experiments/2026-07-22-push-2mm-randomized-training/manifest.json"
IFS=$'\t' read -r \
  training_config training_config_sha params_name qualification_config \
  qualification_config_sha \
  screening_seed_start screening_seed_count blind_seed_start blind_seed_count \
  < <(
    "$python_bin" - "$plan_path" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text())
candidate = plan["candidate"]
qualification = plan["qualification"]
screening = qualification["screening"]
blind = qualification["blind"]
fields = (
    candidate["config_path"],
    candidate["config_sha256"],
    candidate["params_output_name"],
    qualification["config_path"],
    qualification["config_sha256"],
    screening["seed_start"],
    screening["seed_count"],
    blind["seed_start"],
    blind["seed_count"],
)
print("\t".join(str(value) for value in fields))
PY
  )

actual_training_config_sha="$(sha256sum "$training_config" | cut -d' ' -f1)"
if [[ "$actual_training_config_sha" != "$training_config_sha" ]]; then
  echo "training config hash differs from pre-registration" >&2
  exit 2
fi
actual_qualification_config_sha="$(
  sha256sum "$qualification_config" | cut -d' ' -f1
)"
if [[ "$actual_qualification_config_sha" != "$qualification_config_sha" ]]; then
  echo "qualification config hash differs from pre-registration" >&2
  exit 2
fi

mkdir "$run_root"
params_path="$run_root/$params_name"
training_state_dir="$run_root/training-state"
screening_dir="$run_root/screening-20"
blind_dir="$run_root/blind-100"

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export LLVM_PATH="${LLVM_PATH:-/opt/rocm/llvm}"
export HIP_DEVICE_LIB_PATH="${HIP_DEVICE_LIB_PATH:-/opt/rocm-7.2.1/lib/llvm/lib/clang/22/lib/amdgcn/bitcode}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_COMPILATION_CACHE_DIR="$run_root/jax-compilation-cache"

"$python_bin" - "$run_root/launch.json" "$actual_commit" "$plan_path" <<'PY'
from datetime import UTC, datetime
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
commit = sys.argv[2]
plan_path = Path(sys.argv[3])
payload = {
    "schema_version": 1,
    "status": "started",
    "started_at_utc": datetime.now(UTC).isoformat(),
    "commit": commit,
    "plan_path": plan_path.as_posix(),
    "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
}
with output.open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

stage_rc=0
run_stage() {
  local name="$1"
  shift
  set +e
  "$@" >"$run_root/$name.log" 2>&1
  stage_rc=$?
  set -e
  printf '%s\n' "$stage_rc" >"$run_root/${name}_exit_code.txt"
}

write_result() {
  local status="$1"
  "$python_bin" - \
    "$run_root/result.json" "$run_root" "$status" "$params_path" <<'PY'
from datetime import UTC, datetime
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
root = Path(sys.argv[2])
status = sys.argv[3]
params = Path(sys.argv[4])

def read_exit(name):
    path = root / f"{name}_exit_code.txt"
    return int(path.read_text().strip()) if path.exists() else None

def read_summary(name):
    path = root / name / "matrix_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["summary"]

payload = {
    "schema_version": 1,
    "status": status,
    "completed_at_utc": datetime.now(UTC).isoformat(),
    "system_info_exit_code": read_exit("system-info"),
    "training_exit_code": read_exit("training"),
    "screening_exit_code": read_exit("screening"),
    "blind_exit_code": read_exit("blind"),
    "params_sha256": (
        hashlib.sha256(params.read_bytes()).hexdigest() if params.exists() else None
    ),
    "screening_summary": read_summary("screening-20"),
    "blind_summary": read_summary("blind-100"),
}
with output.open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
    stream.write("\n")
PY
}

run_stage system-info \
  "$python_bin" scripts/system_info.py \
  --image-name amd-oneclick-base:rocm7.2.1-py3.12-v20260416 \
  --config "$training_config" \
  --config "$qualification_config" \
  --output-dir "$run_root/system-info"
if (( stage_rc != 0 )); then
  write_result system_info_failed
  exit "$stage_rc"
fi

run_stage training \
  "$python_bin" scripts/locomotion_learn_smoke.py \
  --task push \
  --config "$training_config" \
  --params-out "$params_path" \
  --training-state-dir "$training_state_dir" \
  --skip-eval
if (( stage_rc != 0 )); then
  write_result training_failed
  exit "$stage_rc"
fi
if [[ ! -f "$params_path" ]]; then
  echo "training exited successfully without params output" >&2
  write_result training_output_missing
  exit 1
fi
sha256sum "$params_path" >"$run_root/params.sha256"

run_stage screening \
  "$python_bin" scripts/push_qualification_matrix.py \
  --config "$qualification_config" \
  --params-in "$params_path" \
  --output-dir "$screening_dir" \
  --seed-start "$screening_seed_start" \
  --seed-count "$screening_seed_count" \
  --num-steps 4608 \
  --chunk-steps 2 \
  --python "$python_bin"
if (( stage_rc != 0 )); then
  write_result screening_failed
  exit "$stage_rc"
fi

run_stage blind \
  "$python_bin" scripts/push_qualification_matrix.py \
  --config "$qualification_config" \
  --params-in "$params_path" \
  --output-dir "$blind_dir" \
  --seed-start "$blind_seed_start" \
  --seed-count "$blind_seed_count" \
  --num-steps 4608 \
  --chunk-steps 2 \
  --python "$python_bin"
if (( stage_rc != 0 )); then
  write_result blind_failed
  exit "$stage_rc"
fi

write_result pass
echo "PUSH_RANDOMIZED_CANDIDATE_DONE status=pass result=$run_root/result.json"

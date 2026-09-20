#!/usr/bin/env bash
# One-command setup and validation for a Linux NVIDIA server.
set -Eeuo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
사용법: bash setup.sh [옵션]

기본: 프로젝트 위치 확인 → uv/Python 3.11 설치 → CUDA 패키지 설치
      → GPU 확인 → 단위 테스트 → 합성 데이터 smoke test → VRAM 측정 → 실제 데이터 검사

옵션:
  --train                    모든 검사 통과 후 teacher + student 8조건 전체 학습
  --config PATH              실험 설정 파일 (기본: configs/pilot.json)
  --data-root PATH           Waterbirds metadata.csv가 있는 폴더
  --seg-root PATH            CUB segmentation 종별 폴더의 상위 폴더
  --download-data            공식 데이터 다운로드/압축 해제 (프로젝트 data/)
  --segmentation-archive PATH 기존 segmentation archive 사용 (데이터 준비 포함)
  --waterbirds-archive PATH   기존 Waterbirds archive 사용 (데이터 준비 포함)
  --dry-run                  수행할 명령만 출력; 설치/학습/파일 생성 안 함
  -h, --help                 도움말

예시:
  bash setup.sh
  bash setup.sh --data-root /datasets/waterbirds --seg-root /datasets/segmentations
  bash setup.sh --download-data --train

GPU는 CUDA_VISIBLE_DEVICES로 선택하며, 지정하지 않으면 0번을 사용합니다.
EOF
}

CONFIG="configs/pilot.json"
SETUP_ARGS=("$@")
TRAIN=0
DOWNLOAD=0
DRY_RUN=0
COMMON_ARGS=()
DOWNLOAD_ARGS=()
while (($#)); do
  case "$1" in
    --train) TRAIN=1; shift ;;
    --download-data) DOWNLOAD=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --config|--data-root|--seg-root|--segmentation-archive|--waterbirds-archive)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        printf '오류: %s 뒤에 경로가 필요합니다.\n' "$1" >&2
        exit 2
      fi
      case "$1" in
        --config) CONFIG="$2" ;;
        --data-root|--seg-root) COMMON_ARGS+=("$1" "$2") ;;
        *) DOWNLOAD=1; DOWNLOAD_ARGS+=("$1" "$2") ;;
      esac
      shift 2 ;;
    *) printf '알 수 없는 옵션: %s\n' "$1" >&2; usage; exit 2 ;;
  esac
done

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
LOG_FILE="$PROJECT_DIR/logs/setup_${RUN_ID}.log"
REPORT_DIR="$PROJECT_DIR/reports/setup/$RUN_ID"
SMOKE_OUTPUT="$PROJECT_DIR/outputs/setup_smoke/$RUN_ID"
PYTHON="$PROJECT_DIR/.venv/bin/python"
CURRENT_STEP="시작"
FINISHED=()
STEP_NAMES=()
STEP_STATUS=()
STEP_INDEX=0
STEP_LOG=""
CODE_REVISION="$(git rev-parse HEAD 2>/dev/null || printf 'not-a-git-checkout')"
if [[ "$DRY_RUN" == 0 ]]; then
  mkdir -p "$PROJECT_DIR/logs" "$REPORT_DIR"
  exec > >(tee -a "$LOG_FILE") 2>&1
  if [[ -f "$CONFIG" ]]; then cp "$CONFIG" "$REPORT_DIR/config.json"; fi
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git status --short --untracked-files=normal -- . ':!reports' ':!logs' > "$REPORT_DIR/git_status.txt"
  fi
fi

write_report() {
  [[ "$DRY_RUN" == 0 ]] || return 0
  local state="$1" code="$2" index filename
  {
    printf '# Server setup validation\n\n'
    printf -- '- Status: **%s**\n- Exit code: %s\n- Run ID: `%s`\n' "$state" "$code" "$RUN_ID"
    printf -- '- Recorded at (UTC): %s\n- Git commit: `%s`\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CODE_REVISION"
    printf -- '- Selected CUDA devices: `%s`\n' "${CUDA_VISIBLE_DEVICES:-0}"
    printf -- '- Setup arguments: `'; printf '%q ' "${SETUP_ARGS[@]}"; printf '`\n\n'
    printf '| Step | Result | Evidence |\n|---|---|---|\n'
    for ((index=0; index<STEP_INDEX; index++)); do
      printf -v filename '%02d.log' "$((index+1))"
      printf '| %s | %s | [%s](%s) |\n' "${STEP_NAMES[$index]}" "${STEP_STATUS[$index]}" "$filename" "$filename"
    done
    printf '\nEach log contains the executed command and its combined stdout/stderr.\n'
    printf 'config.json is the input config; command-line overrides are recorded in the logs.\n'
    printf 'git_status.txt records local changes when this is a Git checkout.\n'
    printf '\nThis report verifies setup and diagnostics. Synthetic smoke accuracy is not Waterbirds research evidence.\n'
    printf 'Dataset files, model checkpoints and environment variables are not bundled in this report.\n'
  } > "$REPORT_DIR/README.md.tmp"
  mv "$REPORT_DIR/README.md.tmp" "$REPORT_DIR/README.md"
}

failed() {
  local status=$?
  trap - ERR
  printf '\n[실패] %s (종료 코드 %s)\n' "$CURRENT_STEP" "$status"
  printf '이후 단계는 실행하지 않았습니다. 위 오류를 해결한 뒤 같은 명령으로 다시 실행하세요.\n'
  printf '로그: %s\n' "$LOG_FILE"
  if [[ "$CURRENT_STEP" == "실제 데이터 검사" ]]; then
    printf 'configs/pilot.json의 data_root/seg_root 또는 --data-root/--seg-root 경로를 확인하세요.\n'
    printf '데이터가 없다면 --download-data, 이미 받은 mask archive는 --segmentation-archive PATH를 사용할 수 있습니다.\n'
  fi
  if ((STEP_INDEX > 0)); then STEP_STATUS[$((STEP_INDEX-1))]="FAILED"; fi
  write_report FAILED "$status"
  printf 'GitHub 검증 보고서: %s\n' "$REPORT_DIR/README.md"
  exit "$status"
}
trap failed ERR

interrupted() {
  trap - ERR INT TERM
  if ((STEP_INDEX > 0)); then STEP_STATUS[$((STEP_INDEX-1))]="INTERRUPTED"; fi
  write_report INTERRUPTED "$1"
  printf '\n중단됨. GitHub 검증 보고서: %s\n' "$REPORT_DIR/README.md"
  exit "$1"
}
trap 'interrupted 130' INT
trap 'interrupted 143' TERM

run() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '  $ '
    printf '%q ' "$@"
    printf '\n'
  else
    { printf '$ '; printf '%q ' "$@"; printf '\n'; } >> "$STEP_LOG"
    "$@" 2>&1 | tee -a "$STEP_LOG"
  fi
}

start_step() {
  CURRENT_STEP="$1"
  STEP_INDEX=$((STEP_INDEX+1))
  STEP_NAMES+=("$CURRENT_STEP")
  STEP_STATUS+=("RUNNING")
  printf -v STEP_LOG '%s/%02d.log' "$REPORT_DIR" "$STEP_INDEX"
  write_report RUNNING pending
  printf '\n=== %s ===\n' "$CURRENT_STEP"
}

finish_step() {
  FINISHED+=("$CURRENT_STEP")
  STEP_STATUS[$((STEP_INDEX-1))]="PASSED"
  write_report RUNNING pending
  printf '[%s] %s\n' "$([[ "$DRY_RUN" == 1 ]] && printf '예정' || printf '통과')" "$CURRENT_STEP"
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

printf '프로젝트: %s\n가상환경: %s\nGPU 선택: %s\n' "$PROJECT_DIR" "$PROJECT_DIR/.venv" "$CUDA_VISIBLE_DEVICES"
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'Dry run: 아래 명령을 출력만 합니다.\n'
else
  printf '진행 내용은 터미널과 로그에 함께 출력됩니다: %s\n' "$LOG_FILE"
fi

start_step "서버 사양 확인"
run nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
if command -v free >/dev/null 2>&1; then run free -h; fi
run df -h "$PROJECT_DIR"
finish_step

start_step "uv 및 Python 3.11 환경 구성"
UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then UV_BIN="$HOME/.local/bin/uv"; fi
if [[ -z "$UV_BIN" ]]; then
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '  uv가 없으면 공식 installer(https://astral.sh/uv/install.sh)로 설치합니다.\n'
    UV_BIN="uv"
  else
    INSTALLER_FILE="$PROJECT_DIR/logs/uv-installer_${RUN_ID}.sh"
    run curl --fail --silent --show-error --location https://astral.sh/uv/install.sh --output "$INSTALLER_FILE"
    run sh "$INSTALLER_FILE"
    UV_BIN="${UV_INSTALL_DIR:-$HOME/.local/bin}/uv"
    if [[ ! -x "$UV_BIN" ]]; then
      printf 'uv 설치 경로를 찾지 못했습니다. uv를 PATH에 추가한 뒤 다시 실행하세요.\n'
      false
    fi
  fi
fi
run "$UV_BIN" --version
if [[ ! -d "$PROJECT_DIR/.venv" ]]; then
  run "$UV_BIN" venv --python 3.11 "$PROJECT_DIR/.venv"
else
  printf '기존 .venv를 재사용합니다.\n'
fi
run "$PYTHON" -c 'import sys; print("Python:", sys.version); assert sys.version_info[:2] == (3, 11), "Python 3.11 가상환경이 필요합니다. 기존 .venv를 별도 보관한 뒤 다시 실행하세요."'
finish_step

start_step "CUDA 패키지 설치"
run "$UV_BIN" pip install --python "$PYTHON" 'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121' --index-url https://download.pytorch.org/whl/cu121
run "$UV_BIN" pip install --python "$PYTHON" -r "$PROJECT_DIR/requirements.txt"
run "$UV_BIN" pip check --python "$PYTHON"
run "$PYTHON" -c 'import torch; print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda); assert torch.cuda.is_available(), "CUDA 사용 불가: NVIDIA 드라이버와 PyTorch 설치를 확인하세요."; print("GPU:", torch.cuda.get_device_name(0)); print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))'
finish_step

start_step "자동 테스트"
run "$PYTHON" -m pytest -q
finish_step

start_step "CUDA 전체 흐름 smoke test"
# Unique output prevents a previous successful run from skipping GPU execution.
run "$PYTHON" run.py smoke --device cuda --output-root "$SMOKE_OUTPUT"
finish_step

start_step "실제 DeiT VRAM 측정 (조건별 3 step)"
run "$PYTHON" run.py benchmark --config "$CONFIG" --device cuda "${COMMON_ARGS[@]}" --steps 3
finish_step

if [[ "$DOWNLOAD" == 1 ]]; then
  start_step "데이터 준비"
  run "$PYTHON" scripts/prepare_data.py --data-dir "$PROJECT_DIR/data" "${DOWNLOAD_ARGS[@]}"
  finish_step
fi

start_step "실제 데이터 검사"
run "$PYTHON" run.py preflight --config "$CONFIG" --device cuda "${COMMON_ARGS[@]}" --full-image-check
finish_step

if [[ "$TRAIN" == 1 ]]; then
  start_step "전체 실험 학습 및 분석"
  run "$PYTHON" -u run.py pipeline --config "$CONFIG" --device cuda "${COMMON_ARGS[@]}" --stage all
  finish_step
fi

printf '\n=== %s ===\n' "$([[ "$DRY_RUN" == 1 ]] && printf '실행 예정 순서' || printf '완료 결과')"
printf '  - %s\n' "${FINISHED[@]}"
if [[ "$DRY_RUN" == 0 ]]; then
  printf '\n로그: %s\nSmoke 결과: %s\n' "$LOG_FILE" "$SMOKE_OUTPUT"
fi
if [[ "$TRAIN" == 0 ]]; then
  printf '\n전체 학습 시작 명령:\n  '
  printf 'cd %q && ' "$PROJECT_DIR"
  printf '%q ' "$PYTHON" -u "$PROJECT_DIR/run.py" pipeline --config "$CONFIG" --device cuda "${COMMON_ARGS[@]}" --stage all
  printf '\n'
fi
printf '\n이 스크립트는 .venv Python을 직접 사용하므로 실행 전에 activate할 필요가 없습니다.\n'
write_report PASSED 0
if [[ "$DRY_RUN" == 0 ]]; then
  printf '\nGitHub에 올릴 검증 보고서: %s\n' "$REPORT_DIR/README.md"
  printf 'reports/setup/ 폴더를 git add / commit / push한 뒤 저장소 URL과 commit SHA를 알려주세요.\n'
fi

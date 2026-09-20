# 검증 기록

검증일: 2026-09-20. 환경: macOS arm64, Python 3.11.15, PyTorch 2.5.1 CPU.

- 자동 테스트 **45개 통과**: upstream student logits/CLS attention/top-k/gradient parity,
  upstream teacher full/masked logits parity, 실제 DeiT-Tiny/Small 구조,
  rescue token budget/중복/실제 교체 수/matched random control,
  KL 방향/GT probability/group WGA/가중 평균/FG 0개 처리,
  데이터 split/probe/paired 공간 변환/mask 크기 검사,
  CE-only teacher 미호출, 불균등 마지막 microbatch의 gradient accumulation,
  interruption/resume의 parameter 동일성, 설정 변경 시 resume 거부.
- CUDA 장치 선택 회귀 검사 5개 포함: 번호 없는 `cuda`는 현재 장치 번호로 변환하고,
  명시한 `cuda:1`은 유지하며, CPU/사용 불가능한 CUDA 경로도 검사.
  서버에서 보고된 ValueError를 실제 PyTorch 인자 검사 함수로 재현한 뒤 수정 확인.
  GPU 드라이버 호출은 mock으로 대체했으므로 실제 CUDA 실행 검증은 아님.
- 통합 setup 스크립트 검사 18개 포함: 외부 폴더에서 시작해도 프로젝트 경로 자동 선택,
  공백이 있는 데이터 경로 전달, 검사 이후 학습 순서, 데이터 검사 실패 시 학습 차단,
  기본 실행에서 장기 학습 미시작, dry-run 무변경, 누락된 옵션 값 거부.
  서버 명령은 mock으로 검사했고 실제 CUDA 설치/실행으로 간주하지 않음.
- GitHub 공유 보고서의 성공/실패 상태, 종료 코드, 원본 명령 출력 보관,
  실패한 데이터 검사 로그 유지, Python 설치 전 GPU 확인 실패의 기록,
  dry-run에서 report를 생성하지 않는 동작도 검사.
- 성공 단계 재사용 및 실패 단계 재시도, 코드/설정/GPU 선택/패키지 변경 시 재검사,
  smoke 결과 삭제/강제 검사 시 재실행, 데이터 변경 시 전체 이미지 검사 재실행 확인.
- 저장 경로의 symlink 해석, CLI output 경로 반영, 여유 공간 기록 확인.
  Linux findmnt/lsblk 저장 장치 결과는 서버 보고서로 추가 확인해야 함.
- 로컬 bare Git remote로 성공/실패 보고서 자동 commit/push 검증.
  다른 staged 파일을 포함하지 않고, 원격에 최종 PASSED/FAILED 상태가 보존됨을 확인.
- CUDA 12.1용 torch/torchvision의 uv 의존성 해결을 Linux 대상으로 dry-run 검증.
- 합성 데이터에서 teacher + student **8조건 전체** 학습 완료.
- 이전용 source archive를 별도 폴더에 풀어 동일한 전체 smoke test 재실행 성공
  (완료된 run 9개: teacher 1 + student 8; summary/예시 그림 생성 확인).
- 체크포인트, 매 epoch NPZ, 분석 CSV, 요청한 7종 그래프 생성 및 예시 그림 시각 확인.
- 공식 ImageNet pretrained DeiT-Tiny/Small checkpoint 다운로드·로딩 성공.
  두 모델 모두 2분류 logits `(1,2)`와 CLS-to-patch attention `(1,196)` 출력 확인.
- 실제 DeiT 구조로 teacher 학습 / full KD / masked KD의 forward·backward·optimizer step 확인
  (CPU, batch 1, 각 1 step; GPU 메모리 측정 결과가 아님).
- Ruff 정적 검사 통과, bash 스크립트 문법 검사 통과.
- Waterbirds 공식 archive의 상위 폴더 구조를 스트리밍으로 확인.
- CUB segmentation 다운로드는 이 환경에서 HTTP 403 발생. 이미 받은 archive를
  `scripts/prepare_data.py --segmentation-archive ...`로 사용하는 경로 제공.

수행하지 않은 검증: A5000/CUDA/AMP에서의 실측, 실제 Waterbirds 학습 성능,
실제 Waterbirds–CUB 전체 파일의 정합성, multi-seed 가설 검증.
서버에서 `preflight --full-image-check`, `benchmark`, `smoke --device cuda`를 먼저 실행하세요.
현재 합성 데이터 결과는 과학적 가설을 지지하거나 반박하는 증거가 아닙니다.

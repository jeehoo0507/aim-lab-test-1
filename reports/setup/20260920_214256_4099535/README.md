# Server setup validation

- Status: **FAILED**
- Exit code: 1
- Run ID: `20260920_214256_4099535`
- Recorded at (UTC): 2026-09-20T12:45:03Z
- Git commit: `752294d605128755485d7e3f1d638f393786b6f2`
- Selected CUDA devices: `0`
- Setup arguments: `--push-report `

| Step | Result | Evidence |
|---|---|---|
| 서버 사양 확인 | PASSED | [01.log](01.log) |
| uv 및 Python 3.11 환경 구성 | PASSED | [02.log](02.log) |
| 저장 경로 및 마운트 확인 | PASSED | [03.log](03.log) |
| CUDA 패키지 설치 | PASSED | [04.log](04.log) |
| 자동 테스트 | PASSED | [05.log](05.log) |
| CUDA 전체 흐름 smoke test | PASSED | [06.log](06.log) |
| 실제 DeiT VRAM 측정 (조건별 3 step) | PASSED | [07.log](07.log) |
| 실제 데이터 검사 | FAILED | [08.log](08.log) |

Each log contains the executed command and its combined stdout/stderr.
config.json is the input config; command-line overrides are recorded in the logs.
git_status.txt records local changes when this is a Git checkout.

[Storage paths and mounted devices](storage.json) / [Effective configuration](effective_config.json)
REUSED means prior successful evidence was copied into the current step log; it is not a fresh execution.

This report verifies setup and diagnostics. Synthetic smoke accuracy is not Waterbirds research evidence.
Dataset files, model checkpoints and environment variables are not bundled in this report.

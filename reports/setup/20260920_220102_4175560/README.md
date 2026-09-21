# Server setup validation

- Status: **PASSED**
- Exit code: 0
- Run ID: `20260920_220102_4175560`
- Recorded at (UTC): 2026-09-20T13:05:44Z
- Git commit: `b93f6499c366efec6a9b0f46abf14bf23b0449dd`
- Selected CUDA devices: `0`
- Setup arguments: `--download-data `

| Step | Result | Evidence |
|---|---|---|
| 서버 사양 확인 | PASSED | [01.log](01.log) |
| uv 및 Python 3.11 환경 구성 | PASSED | [02.log](02.log) |
| 저장 경로 및 마운트 확인 | PASSED | [03.log](03.log) |
| CUDA 패키지 설치 | PASSED | [04.log](04.log) |
| 자동 테스트 | REUSED | [05.log](05.log) |
| CUDA 전체 흐름 smoke test | REUSED | [06.log](06.log) |
| 실제 DeiT VRAM 측정 (조건별 3 step) | REUSED | [07.log](07.log) |
| 데이터 준비 | PASSED | [08.log](08.log) |
| 실제 데이터 검사 | PASSED | [09.log](09.log) |

Each log contains the executed command and its combined stdout/stderr.
config.json is the input config; command-line overrides are recorded in the logs.
git_status.txt records local changes when this is a Git checkout.

[Storage paths and mounted devices](storage.json) / [Effective configuration](effective_config.json)
REUSED means prior successful evidence was copied into the current step log; it is not a fresh execution.

This report verifies setup and diagnostics. Synthetic smoke accuracy is not Waterbirds research evidence.
Dataset files, model checkpoints and environment variables are not bundled in this report.

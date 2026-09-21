# Server setup validation

- Status: **FAILED**
- Exit code: 1
- Run ID: `20260920_212556_4035547`
- Recorded at (UTC): 2026-09-20T12:28:35Z
- Git commit: `d777cf77e746f3555208bc56bf2c44b3423df8f1`
- Selected CUDA devices: `0`
- Setup arguments: `'' `

| Step | Result | Evidence |
|---|---|---|
| 서버 사양 확인 | PASSED | [01.log](01.log) |
| uv 및 Python 3.11 환경 구성 | PASSED | [02.log](02.log) |
| CUDA 패키지 설치 | PASSED | [03.log](03.log) |
| 자동 테스트 | PASSED | [04.log](04.log) |
| CUDA 전체 흐름 smoke test | FAILED | [05.log](05.log) |

Each log contains the executed command and its combined stdout/stderr.
config.json is the input config; command-line overrides are recorded in the logs.
git_status.txt records local changes when this is a Git checkout.

This report verifies setup and diagnostics. Synthetic smoke accuracy is not Waterbirds research evidence.
Dataset files, model checkpoints and environment variables are not bundled in this report.

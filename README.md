# Waterbirds MaskedKD bottleneck 실험

Student-guided masking이 teacher의 foreground 관측을 제한하는지 검증하는 코드입니다.
동일한 98개 patch budget에서 foreground rescue와 random rescue를 비교합니다.
새 selector나 loss를 제안하지 않습니다.

## A5000 서버에서 빠르게 시작

Linux, NVIDIA GPU 한 장, Python **3.11**을 기준으로 합니다.
학습은 항상 순차 실행됩니다. 기본 batch 32, accumulation 4, AMP,
DataLoader worker 4, CPU thread 4입니다. 유효 배치는 128입니다.
실제 VRAM과 속도는 아래 `benchmark`에서 확인하세요. 서버의 전력 제한은 변경하지 않습니다.

```bash
tar -xzf maskedkd_waterbirds_a5000.tar.gz
cd maskedkd_waterbirds
bash scripts/setup_server.sh
```

`bash setup.sh`도 같은 명령입니다. 실행 위치를 자동으로 프로젝트 최상위로 맞추고,
uv가 없다면 설치한 뒤 Python 3.11 `.venv`를 생성합니다. 환경 생성과 패키지 설치는
**uv venv / uv pip** 방식입니다. `uv.lock` 기반 `uv sync` 프로젝트는 아닙니다.
별도로 `cd`를 맞추거나 `source .venv/bin/activate`를 실행할 필요 없이 스크립트가
프로젝트 `.venv`의 Python으로 모든 검사를 수행합니다. 이미 존재하는 환경은 재사용합니다.

기본 명령은 사양 확인 → uv 환경/패키지 설치 → CUDA 확인 → 자동 테스트 → CUDA smoke test
→ 실제 모델 VRAM 측정 → 전체 데이터 파일 검사를 수행합니다.
각 단계의 통과/실패, GPU 이름과 여유 메모리, 검사 결과, 다음 실행 명령이 출력됩니다.
콘솔 출력은 `logs/setup_날짜_시간_PID.log`에도 저장됩니다. 실패하면 즉시 멈춥니다.
Smoke 결과는 `outputs/setup_smoke/날짜_시간_PID/`에 생성되어 이전 결과 때문에 검사를 건너뛰지 않습니다.
검증 공유용 보고서는 `reports/setup/날짜_시간_PID/`에 별도로 저장합니다.
성공 시 PASSED, 실패 시 FAILED와 종료 코드/실패 단계가 남습니다.
프로세스가 강제 종료되어 완료 기록이 없으면 RUNNING 상태를 성공으로 해석하면 안 됩니다.

```bash
# 준비된 데이터 경로를 지정하고 세팅·검사
bash setup.sh --data-root /datasets/waterbirds --seg-root /datasets/segmentations

# 설정 파일에 데이터 경로가 준비되어 있다면, 세팅·검사·전체 학습을 한 번에
bash setup.sh --train

# 데이터 다운로드도 필요한 첫 실행 (기본 프로젝트 data/에 준비)
bash setup.sh --download-data --train

# 이미 받은 segmentation archive 사용
bash setup.sh --segmentation-archive /path/to/segmentations.tgz --train

# 변경 없이 어떤 명령을 실행할지 확인
bash setup.sh --dry-run --train
```

`--download-data`를 주지 않으면 데이터를 자동 다운로드하지 않습니다.
기본 데이터 경로는 `configs/pilot.json`에 있습니다. 데이터가 없으면 데이터 검사 단계에서
필요한 경로를 안내하며 멈춥니다. Caltech에서 HTTP 403이 나면 아래 archive 입력 방법을 사용하세요.
`--download-data`는 프로젝트 `data/`에 풀기 때문에 별도의 custom config를 사용한다면 경로도 맞춰야 합니다.

세팅 이후 직접 명령을 실행할 때는 `source .venv/bin/activate` 후
`python run.py pipeline --stage all`을 사용합니다. Baseline만 실행하려면
`bash scripts/run_pilot.sh`, rescue만 추가하려면 `python run.py pipeline --stage rescue`입니다.
기본 총량은 teacher 30 epochs + student 8조건 × 100 epochs입니다.
한 번에 여러 GPU 작업을 띄우지 않지만, 전체 실험에는 시간이 필요합니다.
구체적인 소요 시간은 서버에서 측정하기 전에는 보장하지 않습니다.

메모리가 부족하거나 다른 작업과 함께 실행해야 한다면 **시작 전에** config를
`batch_size: 16`, `accumulation_steps: 8`, `eval_batch_size: 16`, `num_workers: 2`로
바꾸세요. 유효 배치 128은 유지됩니다. 진행 중 설정 변경은 새 output_root에서 실행합니다.

CUDA 12.1용 PyTorch 2.5.1 / torchvision 0.20.1을 설치합니다.
드라이버가 지원하지 않으면 관리자에게 확인하거나
[PyTorch 공식 이전 버전 설치 안내](https://pytorch.org/get-started/previous-versions/)
에서 드라이버에 맞는 같은 버전의 wheel을 선택하세요.
GPU 선택은 `CUDA_VISIBLE_DEVICES=0 bash setup.sh --train`처럼 지정하며 기본은 0번입니다.

## 검증 결과를 GitHub로 전달

서버의 clone 폴더에서 아래 순서로 실행합니다. 이 단계에서는 `--train` 없이
환경/데이터 검증 결과를 먼저 공유하는 것을 권합니다. Setup이 실패해도 보고서를 올리면
단계별 로그를 통해 원인을 확인할 수 있습니다.

```bash
bash setup.sh
git add reports/setup/
git commit -m "chore: add A5000 setup report"
git push
git rev-parse HEAD
```

Push 후 저장소 URL과 마지막 commit SHA를 전달하세요. 보고서에는 GPU/VRAM/드라이버,
Python/PyTorch/CUDA 출력, 테스트 결과, CUDA smoke test, 실제 모델의 최대 VRAM,
데이터 정합성 검사, 실행한 코드의 commit과 초기 Git 변경 상태가 포함됩니다.
각 단계의 실행 명령과 stdout/stderr를 따로 보관하므로 성공/실패를 실제 로그로 검증할 수 있습니다.
`config.json`은 입력 설정이며 CLI override는 로그에 기록됩니다.

`reports/`는 Git에서 추적 가능한 경로입니다. `.venv`, `outputs`, `logs`, `data`는
계속 제외되므로 `git add -f outputs/` 같은 명령은 필요하지 않습니다.
보고서는 컴퓨터 경로/파일명 등 실행 정보를 포함하며, 환경변수 전체를 수집하지 않습니다.
합성 smoke test 통과는 실제 Waterbirds의 수렴·가설 검증을 의미하지 않습니다.
배포용 source archive에는 다른 서버의 reports를 포함하지 않습니다.

## 데이터 구조

```text
data/
  waterbird_complete95_forest2water2/
    metadata.csv
    001.Black_footed_Albatross/...jpg
    ...
  CUB_200_2011/
    segmentations/
      001.Black_footed_Albatross/...png
      ...
```

`metadata.csv`의 `img_filename`, `y`, `place`, `split`을 사용합니다.
`split=0/1/2`는 train/validation/test, `group_id=2*y+place`입니다.
Waterbirds의 상대 이미지 경로에서 `.jpg`를 `.png`로 바꾸어 CUB segmentation과 대응합니다.
두 파일의 원본 크기가 다르면 즉시 오류를 냅니다. 전처리된 다른 크기의 mask를 섞지 마세요.

공식 데이터의 명목 split 크기는 train 4,795 / validation 1,199 / test 5,794입니다.
`preflight`는 실제 개수와 네 그룹의 분포를 출력하고 전체 경로를 검사합니다.
`--full-image-check`는 모든 이미지/mask의 디코딩 가능 여부와 크기까지 검사합니다.
원본 사진 전체나 Places를 받을 필요는 없습니다. 배포된 Waterbirds와 CUB mask만 필요합니다.
다운로드 helper는 큰 Waterbirds archive를 받으므로 실제 다운로드는 서버에서 실행하세요.

Caltech 다운로드는 실행 환경에 따라 HTTP 403이 날 수 있습니다(현재 로컬 확인에서도 발생).
이 경우 공식 CUB segmentation 페이지에서 브라우저로 받은 파일을 서버에 옮겨 사용합니다.

```bash
python scripts/prepare_data.py --segmentation-archive /path/to/segmentations.tgz
# 두 archive를 모두 가지고 있다면 --waterbirds-archive /path/to/waterbird_complete95_forest2water2.tar.gz 도 지정
```

이미 저장된 데이터 경로를 CLI로도 지정할 수 있습니다. 이후 명령에도 같은 경로를 사용하세요.

```bash
python run.py pipeline --stage all \
  --data-root /datasets/waterbird_complete95_forest2water2 \
  --seg-root /datasets/CUB_200_2011/segmentations
```

Segmentation 없이 baseline만 확인하려면 config의 `seg_root`를 `""`로 설정합니다.
이 경우 foreground 분석·rescue는 실행할 수 없습니다.

## 실험 조건과 고정 규칙

| 설정 | 기본값 |
|---|---|
| Teacher | ImageNet pretrained DeiT-Small, Waterbirds 2분류 head fine-tuning |
| Student | DeiT-Tiny, **scratch 초기화** (공식 MaskedKD 기본과 일치) |
| 입력 / patch | 224×224 / 16×16 |
| Student token | 항상 196 patches + CLS |
| Masked teacher token | 98 patches + CLS |
| Teacher / student epochs | 30 / 100 |
| Optimizer | AdamW, weight decay 0.05, bias/norm/CLS/position은 decay 제외 |
| LR | teacher 5e-5, student 5e-4 × 유효 배치/512 = 1.25e-4 |
| Scheduler | 5-epoch warmup + cosine; 최소 LR 1e-6 |
| CE | label smoothing 0.1, 모든 조건 동일 |
| Soft KD | alpha 0.5, temperature 1; KL(teacher || student) × T² |
| Augmentation | paired random resized crop(scale 0.7–1.0), horizontal flip |
| 평가 / probe | resize short edge 256, center crop 224, 고정 변환 |
| Group metric | 각 그룹 accuracy, minimum WGA, raw average, train-group-weighted average |
| Checkpoint 선택 | **validation WGA 최대**, 동률이면 먼저 도달한 epoch |
| Test | 선택된 best checkpoint로 최종 평가; 선택·스케줄에 사용하지 않음 |

작은 데이터에서 scratch student 학습이 약하면 새로운 output_root에서
`student_init: "imagenet"`으로 전체 조건을 다시 비교할 수 있습니다.
이는 별도 실험이며, scratch와 pretrained 결과를 같은 표로 섞지 마세요.
모든 조건은 seed별 동일한 student 초기화와 학습 image 순서/augmentation을 사용합니다.
mask sampling은 별도의 RNG를 사용하여 image augmentation RNG를 바꾸지 않습니다.

Baseline:

- `ce`: CE만 사용합니다. 학습 중 teacher forward가 없습니다. Pipeline에서는 비교용 probe에만 teacher를 사용합니다.
- `full`: student/teacher 모두 full input.
- `random`: teacher에 무작위 98 patches.
- `student`: 마지막 block의 CLS-to-patch attention을 head 평균한 후 top-98.

Rescue:

- `foreground_rescue_5`, `foreground_rescue_10`, `foreground_rescue_20`.
- `random_rescue_10`.
- Foreground rescue는 선택된 BG를 균등 무작위로 빼고, 미선택 FG를 균등 무작위로 넣습니다.
- 실제 교체 수는 `min(요청 수, 선택된 BG 수, 미선택 FG 수)`이며 반드시 기록합니다.
- Random rescue도 해당 이미지의 **같은 실제 교체 수**로 선택/미선택 token을 무작위 교체합니다.
  FG가 부족한 이미지에서 명목 10개를 강제로 바꾸지 않습니다.
- 항상 중복 없이 98 patches를 유지합니다. CLS와 원래 위치 임베딩은 보존합니다.
- 동일한 것은 teacher transformer의 token budget입니다. Segmentation/분석 비용을 포함한
  전체 wall time이 완전히 같다는 뜻은 아닙니다.

## Attention / teacher 분석

Validation에서 그룹당 50개, 총 200개를 고정 선택합니다. `probe/manifest.json`에 ID를 저장합니다.

- **매 epoch**: attention `[N,196]`, top-k index, binary mask, student logits/probabilities,
  정답·배경·group·sample ID, foreground patch mask와 연속 coverage.
- Epoch `0,5,10,20,40,60,80,100` 및 final: full teacher, student mask, random mask,
  FG rescue-5/10/20, matched random rescue-5/10/20의 teacher logits / indices / 실제 교체 수.
- Epoch 0은 학습 이전입니다. Diagnostic random mask는 epoch 간 같은 RNG seed로 비교합니다.
- Full teacher는 probe 입력이 고정되어 있으므로 run 안에서 결과를 캐시합니다.
- Logger는 `eval()` + inference mode이며 학습 batch의 augmentation 결과를 probe로 사용하지 않습니다.

Foreground는 patch 내 mask pixel 평균이 0.5 이상인 patch입니다.
Threshold는 `foreground_threshold`로 설정합니다.

```text
Foreground Selection Ratio = 선택된 FG patch 수 / 선택된 전체 patch 수
Foreground Missing Ratio   = 미선택 FG patch 수 / 전체 FG patch 수
Foreground Coverage Retained = 선택된 patch의 연속 bird coverage 합 / 전체 bird coverage 합
```

FG patch가 0개면 missing ratio는 NaN으로 남깁니다. 0으로 채워 결과를 왜곡하지 않습니다.
배경 비율은 원래 배경 면적과 random mask 기준을 함께 해석해야 합니다.

분석은 Spearman / top-k overlap, `KL(full teacher || masked teacher)`, prediction flip,
GT probability drop, 그룹별 상관관계, rescue 전후 paired 차이를 생성합니다.
Full teacher가 맞고 student가 틀린 **correction opportunity subset**도 따로 집계합니다.
해당 subset의 sample 수가 작거나 0이면 강한 결론을 내리지 마세요.

Teacher-level rescue는 **같은 student checkpoint, 같은 이미지**에서 mask만 바꾸는 비교입니다.
Student-level rescue는 조건별로 다시 학습한 학생의 test WGA 비교입니다.
Probe 그래프의 final은 마지막 학습 epoch이며, 성능 표의 validation-best와 구분합니다.

## 출력과 재시작

```text
outputs/pilot/
  preflight.json
  seed_0/
    teacher/{config.json,history.json,best.pt,last.pt,result.json}
    student/
      config.json, history.json, best.pt, last.pt, result.json
      probe/manifest.json, epoch_000.npz, ...
      analysis/*.csv, *.png
    ce/, full/, random/, foreground_rescue_5/, ...
  summary/
    results_per_seed.csv
    results_mean_std.csv
    paired_wga_differences.csv
    05_group_accuracy.png
    06_worst_group_accuracy.png
```

학생별 `analysis/`에서 요청한 Plot 1–4 및 Plot 7, 추가 rescue plot을 확인합니다.
Plot 5–6은 `summary/`에 있습니다. 모든 성능은 0–1 범위입니다.

같은 명령을 다시 실행하면 완료된 run은 건너뛰고 미완료 run은 `last.pt`에서 이어갑니다.
Checkpoint는 매 epoch 원자적으로 저장하고 optimizer·AMP scaler·history를 포함합니다.
중단된 epoch는 처음부터 다시 수행합니다. Epoch별 RNG를 재설정하여 데이터 순서를 복원합니다.
같은 머신/설정의 CPU 테스트에서는 중단 없는 학습과 parameter가 정확히 일치했습니다.
서로 다른 장치나 CUDA kernel에서 bitwise 동일성을 보장하지는 않습니다.

메타데이터 SHA-256 및 teacher checkpoint SHA-256이 달라지면 재개를 거부합니다.
학습 설정이 달라도 새 output_root를 요구합니다. 서버 이전 시 데이터 경로·device·worker·thread 수는 바꿀 수 있습니다.
동일 output_root/seed/method에 두 프로세스를 동시에 실행하지 마세요.

```bash
# 단일 조건
python run.py train --role teacher
python run.py train --method student
python run.py train --method foreground_rescue_10

# 추가 2 seeds: 각각 teacher를 새로 학습합니다. 여전히 순차 실행입니다.
for seed in 1 2; do
  python run.py pipeline --stage all --seed "$seed"
done
python run.py summarize

# 분석 재생성 / 다른 위치로 데이터 이동 후
python run.py analyze --data-root /datasets/waterbirds --seg-root /datasets/segmentations

# 체크포인트 평가
python run.py evaluate --checkpoint outputs/pilot/seed_0/student/best.pt
```

3 seeds는 pilot 확장이지 강한 통계적 유의성을 자동 보장하지 않습니다.
`results_mean_std.csv`는 seed 간 sample SD, `paired_wga_differences.csv`는 같은 seed의
MaskedKD 대비 WGA 차이를 제공합니다.

## 공식 MaskedKD와의 관계

[공식 저장소](https://github.com/effl-lab/MaskedKD)의 커밋
`96d052da7346441e2425876a9ad37ff8c87fe383`를 기준으로 합니다.
수정하지 않은 참조 소스와 provenance는 `vendor/MaskedKD/`에 보관했습니다.
실행 코드는 `waterbirds_kd/models.py`의 **PyTorch-only 최소 adaptation**입니다.
오래된 timm import/registry 의존성을 제거하고, 원본 parameter 이름과 ViT 연산 순서를 유지합니다.
원본 CLI 자체를 실행하는 구성은 아닙니다.

공식 클래스와 동일한 가중치를 넣고 logits, CLS attention, top-k, teacher masking,
student gradient를 비교하는 parity test가 있습니다. ImageNet pretrained weight의
이름/shape도 그대로 대응하며, 1,000분류 head만 버리고 2분류 head를 초기화합니다.

Waterbirds를 위해 바꾼 것은 loader/group metric, binary head, CE-only 경로,
저부하 학습 루프, probe/rescue 분석입니다. Mixup/CutMix/Random Erasing/RandAugment/
repeated augmentation은 첫 pilot에서 제외합니다. 공간 mask 정합성과 해석을 우선한
공통 설정이며, 원 논문의 ImageNet 학습 recipe 전체를 재현한다는 의미는 아닙니다.

데이터 출처:
[Waterbirds / group DRO](https://github.com/kohpangwei/group_DRO#waterbirds),
[CUB segmentation](https://data.caltech.edu/records/w9d68-gec53).
Waterbirds 관행의 average accuracy는 train group 비율로 가중한 값이므로 raw average와 모두 저장합니다.

## 로컬 검증 및 이전용 패키지

```bash
python -m pytest -q
python run.py smoke --device cpu
python scripts/package.py
```

`maskedkd_waterbirds_a5000.tar.gz`에는 코드·설정·설명·테스트·upstream 참조만 포함됩니다.
`.venv`, 실제 데이터, 학습 결과, downloaded pretrained weights는 포함하지 않습니다.
서버에서 인터넷이 안 된다면 ImageNet weight cache도 별도로 준비해야 합니다.
기본 cache 위치는 `~/.cache/torch/hub/checkpoints/`입니다.

본 코드 작성 시 실제 Waterbirds 학습과 A5000 CUDA 실행은 수행하지 않았습니다.
합성 데이터의 수치/그래프는 파이프라인 동작 확인용이며 가설의 증거가 아닙니다.

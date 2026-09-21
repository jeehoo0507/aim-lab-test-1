# Waterbirds MaskedKD — seed 0 scratch 결과

2026-09-20 A5000 서버에서 실행한 Waterbirds MaskedKD pilot 결과를 정리한 폴더입니다.
Student는 ImageNet 사전학습 없이 초기화한 DeiT-Tiny이고, teacher는 ImageNet 사전학습된
DeiT-Small입니다. 모든 student 조건은 seed 0과 동일한 초기화/데이터 순서를 사용했습니다.

## 바로 볼 파일

- [`CONCLUSIONS.md`](CONCLUSIONS.md): 결과 해석, 통계 검정, 한계와 다음 실험
- [`figures/worst_group_accuracy.png`](figures/worst_group_accuracy.png): 조건별 test WGA
- [`figures/group_accuracy.png`](figures/group_accuracy.png): 네 그룹별 test 정확도
- [`figures/training_curves.png`](figures/training_curves.png): epoch별 validation WGA와 train loss
- [`figures/teacher_signal_probe.png`](figures/teacher_signal_probe.png): correction-opportunity probe 비교
- [`tables/final_metrics_with_wilson_ci.csv`](tables/final_metrics_with_wilson_ci.csv): 최종 성능과 기술적 95% 신뢰구간
- [`tables/probe_paired_tests.csv`](tables/probe_paired_tests.csv): foreground/random rescue paired 검정
- [`../../reports/setup/20260920_220652_3875/results/`](../../reports/setup/20260920_220652_3875/results/): 서버에서 내보낸 config, history, result, CSV, 그림 전체

## 체크포인트 상태

서버가 내보낸 분석 압축에는 `.pt`, `.pth`, `.ckpt` 파일이 포함되지 않았습니다. 따라서 이
폴더에도 실제 모델 가중치는 없습니다. 서버에 남아 있는 체크포인트의 예상 위치와 보존
방법은 [`checkpoints/README.md`](checkpoints/README.md)에 기록했습니다.

## 실험 구성

- 데이터: `waterbird_complete95_forest2water2`
- seed: `0`
- teacher: DeiT-Small, ImageNet pretrained, 30 epochs
- student: DeiT-Tiny, scratch, 100 epochs
- student 조건: CE, full KD, random mask, student-guided mask, foreground rescue 5/10/20,
  matched random rescue 10
- 선택 규칙: validation WGA가 가장 높은 체크포인트, 동률이면 가장 이른 epoch
- probe: validation에서 그룹당 50장, 총 200장

## 출처와 무결성

원본 분석 export는 53.68 MiB이며 총 9개 실행이 완료된 상태입니다. 원본 archive SHA-256은
`6349fc...d4be`로 검증되었습니다. 내보낸 파일 목록은 canonical 원본인
[`reports/setup/20260920_220652_3875/results/manifest.json`](../../reports/setup/20260920_220652_3875/results/manifest.json)에
보존되어 있습니다. 동일한 54 MiB 원본을 이 정리 폴더에 다시 복제하지 않습니다.

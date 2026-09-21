# 결과 분석 및 결론

## 핵심 결론

Foreground rescue가 **student가 틀린 사례에서 teacher supervision을 개선한다는 증거**는
확인됐습니다. 하지만 현재 한 개 seed의 최종 test WGA만으로는 foreground rescue가 student의
최종 성능을 유의하게 향상시켰다고 결론 내릴 수 없습니다.

절대 WGA가 낮은 주된 원인은 epoch 부족보다는 Waterbirds의 심한 그룹 불균형과 scratch
DeiT-Tiny의 표현 학습 한계로 판단됩니다. 현재 결과는 가설의 중간 메커니즘을 지지하지만,
최종 student 성능 효과는 추가 검증이 필요합니다.

## 최종 test 결과

| 방법 | best epoch | Test accuracy | Test WGA | Student 대비 WGA |
|---|---:|---:|---:|---:|
| teacher | 11 | 88.80% | 77.38% | — |
| foreground rescue 5 | 42 | 60.86% | **21.65%** | **+1.25%p** |
| foreground rescue 20 | 61 | 58.75% | 20.72% | +0.31%p |
| foreground rescue 10 | 78 | 60.42% | 20.40% | 0.00%p |
| student-guided MaskedKD | 28 | 57.80% | 20.40% | 기준 |
| random mask | 31 | 58.65% | 19.94% | -0.47%p |
| CE | 56 | 59.87% | 19.00% | -1.40%p |
| full KD | 3 | 55.07% | 19.00% | -1.40%p |
| matched random rescue 10 | 50 | 59.29% | 19.00% | -1.40%p |

모든 student의 최악 그룹은 `waterbird_land`였습니다. Student-guided MaskedKD는 642장 중
131장, foreground rescue 5는 139장을 맞혀 차이는 8장입니다. 각 방법의 기술적 Wilson 95%
구간은 크게 겹칩니다. 동일 test 샘플에 대한 prediction이 export되지 않았고 훈련 seed도 하나라
최종 성능의 paired 유의성 검정은 할 수 없습니다.

세부 수치는 [`tables/final_metrics_with_wilson_ci.csv`](tables/final_metrics_with_wilson_ci.csv)에
있습니다.

## 데이터 불균형

훈련 그룹 수는 다음과 같습니다.

| 그룹 | 훈련 이미지 수 |
|---|---:|
| landbird / land | 3,498 |
| landbird / water | 184 |
| waterbird / land | **56** |
| waterbird / water | 1,057 |

정렬된 그룹 정확도는 대체로 88–94%지만 충돌 그룹은 19–33%였습니다. Student의 훈련분포
가중 test 정확도는 약 89%이고 WGA는 약 20%이므로 배경 shortcut 패턴이 강합니다. Teacher가
동일 데이터에서 WGA 77.38%를 얻었으므로 데이터/label pipeline 자체가 손상된 징후는 없습니다.

## Epoch 검토

Student 방법들의 best epoch는 3–78 사이였고, epoch 100이 best인 조건은 없습니다. 최적점 이후
validation WGA는 대부분 하락했습니다. 현재 cosine schedule은 epoch 100에서 learning rate가
약 `1e-6`까지 내려가므로 기존 체크포인트를 단순 연장하는 것은 우선순위가 낮습니다.

[`figures/training_curves.png`](figures/training_curves.png)에서 전체 수렴 경향을 확인할 수 있습니다.

## Teacher-signal paired 검정

고정 probe 200장 중 full teacher는 맞고 student는 틀린 65장을 correction opportunity로 정의했습니다.
Epoch 100 attention으로 동일 이미지에 foreground rescue와 matched random rescue를 적용했습니다.

| 비교 | Teacher correctness 차이 | Exact McNemar p | GT 확률 차이 | Paired Wilcoxon p | KL 차이 | Paired Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| foreground 5 − random 5 | +3.08%p | 0.6250 | +0.0306 | 0.0676 | -0.0584 | 0.1752 |
| foreground 10 − random 10 | +7.69%p | 0.0625 | **+0.0754** | **0.000047** | **-0.1197** | **0.0049** |
| foreground 20 − random 20 | +9.23%p | 0.0313 | **+0.0823** | **0.000078** | -0.1108 | 0.0667 |

세 rescue 크기를 보수적으로 Bonferroni 보정하면 correctness의 p=0.0313은 0.05 기준을 통과하지
못합니다. 반면 foreground 10/20의 정답 클래스 확률 개선은 보정 후에도 유지됩니다. 즉,
foreground rescue가 teacher 신호를 더 유용하게 만든다는 근거는 있으나 최종 WGA 개선과는 아직
연결되지 않았습니다.

전체 결과는 [`tables/probe_paired_tests.csv`](tables/probe_paired_tests.csv)에 있습니다.

## 해석상의 제한

1. 훈련 seed가 0 하나뿐이라 seed 간 변동을 추정할 수 없습니다.
2. 최종 test per-image prediction/logit이 없어 방법 간 paired test를 할 수 없습니다.
3. probe 분석은 epoch 100 attention을 사용하지만 test 성능은 각 방법의 best checkpoint를 사용합니다.
4. Teacher와 student는 모델 크기뿐 아니라 ImageNet 사전학습 여부도 달라, 크기 효과와 초기화 효과를
   분리할 수 없습니다.
5. 학습은 그룹 균형 sampler나 group DRO가 없는 일반 CE/KD이며 augmentation도 crop/flip 중심입니다.

## 다음 실험

현재 scratch 설정의 재현성을 확인하려면 `student`, `foreground_rescue_10`,
`random_rescue_10`만 seed 1과 2에서 먼저 반복합니다. 방향이 일관되면 seed 3과 4를 추가합니다.

절대 성능의 floor effect를 해결하려면 데이터와 masking 설정은 유지하고 student 초기화만
`imagenet`으로 바꾼 seed 0 pilot을 별도로 실행합니다. 그 결과 WGA가 충분히 상승하면 해당 설정에서
5 seeds를 실행합니다. 이후에는 best checkpoint probe와 per-image test prediction을 함께 저장해야
최종 성능과 teacher-signal 메커니즘을 직접 연결할 수 있습니다.


# Checkpoint 상태

분석 결과 압축에는 모델 체크포인트가 포함되지 않았습니다. 이 디렉터리는 체크포인트 상태를
명시하기 위해 존재하며, 현재 실제 `.pt`, `.pth`, `.ckpt` 파일은 없습니다.

서버 원본 저장소에서 예상되는 best 체크포인트 위치는 다음과 같습니다.

```text
outputs/pilot/seed_0/teacher/best.pt
outputs/pilot/seed_0/ce/best.pt
outputs/pilot/seed_0/full/best.pt
outputs/pilot/seed_0/random/best.pt
outputs/pilot/seed_0/student/best.pt
outputs/pilot/seed_0/foreground_rescue_5/best.pt
outputs/pilot/seed_0/foreground_rescue_10/best.pt
outputs/pilot/seed_0/foreground_rescue_20/best.pt
outputs/pilot/seed_0/random_rescue_10/best.pt
```

서버에서 존재 여부와 용량을 확인하려면 저장소 루트에서 실행합니다.

```bash
find outputs/pilot/seed_0 -mindepth 2 -maxdepth 2 -name 'best.pt' \
  -exec sha256sum {} \; \
  -exec du -h {} \;
```

`last.pt`는 optimizer/scaler 상태까지 포함해 더 크므로 재개가 필요하지 않으면 보존 대상에서
제외해도 됩니다. 일반 GitHub 저장소에는 체크포인트를 직접 커밋하지 말고, 필요할 때 Git LFS나
외부 대용량 저장소를 사용해야 합니다.


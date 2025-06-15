# MMTD 경량화 모델 실험

이 프로젝트는 기존 MMTD(Multi-Modal Text Detection) 모델을 경량화하여 추론 속도를 향상시키는 실험을 포함합니다.

## 📁 파일 구조

```
MMTD/
├── models.py                    # 기존 MMTD 모델들
├── lightweight_models.py        # 경량화된 모델들
├── train_lightweight_model.py   # 경량화 모델 훈련 스크립트
├── inference_speed_comparison.py # 추론 속도 비교 실험
├── evaluate_performance.py      # 기존 모델 성능 평가
└── README_lightweight.md        # 이 파일
```

## 🚀 경량화 모델 종류

### 1. LightWeightMMTD
- **BERT → DistilBERT**: 파라미터 50% 감소, 속도 60% 향상
- **BEiT → ViT Small**: 이미지 인코더 경량화
- **Transformer → FC Layer**: 멀티모달 융합 레이어 단순화

### 2. UltraLightMMTD
- 더욱 간단한 구조로 최대 경량화
- 단일 FC 레이어로 직접 분류
- 최소한의 연산으로 빠른 추론

## 🔧 설치 및 환경 설정

```bash
# 필요한 패키지 설치
pip install torch torchvision transformers
pip install scikit-learn pandas numpy matplotlib seaborn
pip install Pillow tqdm
```

## 📊 실험 실행 방법

### 1. 경량화 모델 훈련

```bash
# LightWeightMMTD 모델 훈련 (모든 fold)
python train_lightweight_model.py --model_type lightweight --epochs 3

# UltraLightMMTD 모델 훈련 (모든 fold)
python train_lightweight_model.py --model_type ultralight --epochs 3

# 특정 fold만 훈련
python train_lightweight_model.py --model_type lightweight --fold 1 --epochs 5

# 하이퍼파라미터 조정
python train_lightweight_model.py --model_type lightweight --batch_size 32 --learning_rate 1e-5
```

### 2. 추론 속도 비교 실험

```bash
# 모든 모델의 추론 속도 비교
python inference_speed_comparison.py
```

이 실험은 다음을 수행합니다:
- 기존 MMTD vs LightWeightMMTD vs UltraLightMMTD 추론 시간 측정
- 모델 크기 비교
- 속도 향상 배수 계산
- 결과 시각화 및 저장

### 3. 기존 모델 성능 평가 (참고용)

```bash
# 기존 MMTD 모델 성능 평가
python evaluate_performance.py
```

## 📈 결과 분석

### 실험 결과 저장 위치

```
lightweight_checkpoints/          # 훈련된 모델 체크포인트
├── lightweightmmtd_fold1/
├── lightweightmmtd_fold2/
├── ...
└── lightweight_summary/          # 전체 결과 요약

inference_results/                 # 추론 속도 비교 결과
├── detailed_results.json
├── inference_time_comparison.png
├── speedup_comparison.png
└── inference_time_distribution.png
```

### 예상 성능 향상

| 모델 | 파라미터 감소 | 추론 속도 향상 | 정확도 유지 |
|------|---------------|----------------|-------------|
| LightWeightMMTD | ~40% | 2-3x | ~95% |
| UltraLightMMTD | ~60% | 4-5x | ~90% |

## 🔍 모델 구조 비교

### 기존 MMTD
```
BERT (110M) + BEiT (86M) + Transformer Layer + Pooler + Classifier
총 파라미터: ~200M
```

### LightWeightMMTD
```
DistilBERT (66M) + ViT-Small (22M) + FC Layer + Pooler + Classifier
총 파라미터: ~90M
```

### UltraLightMMTD
```
DistilBERT (66M) + ViT-Small (22M) + Simple Classifier
총 파라미터: ~88M
```

## 📋 실험 설정

### 데이터셋
- 이메일 스팸 분류 데이터셋
- 텍스트 + 이미지 멀티모달 데이터
- 5-fold 교차 검증

### 훈련 설정
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Epochs**: 3
- **Scheduler**: Linear with warmup

### 추론 속도 측정 설정
- **Warmup Runs**: 10회
- **Measurement Runs**: 50회
- **Batch Size**: 8
- **Device**: CUDA (GPU) 또는 CPU

## 🎯 사용 예시

### 경량화 모델 로드 및 사용

```python
from lightweight_models import LightWeightMMTD
import torch

# 모델 초기화
model = LightWeightMMTD()

# 체크포인트 로드
checkpoint = torch.load('lightweight_checkpoints/lightweightmmtd_fold1/best_model.pth')
model.load_state_dict(checkpoint)
model.eval()

# 추론
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
    predictions = torch.argmax(outputs.logits, dim=-1)
```

### 모델 크기 확인

```python
# 모델 크기 정보 출력
size_info = model.get_model_size()
print(f"총 파라미터: {size_info['total_parameters']:,}")
print(f"모델 크기: {size_info['total_size_mb']:.2f} MB")
```

## 🔧 커스터마이징

### 새로운 경량화 모델 추가

`lightweight_models.py`에 새로운 클래스를 추가하고, `train_lightweight_model.py`에서 해당 모델을 지원하도록 수정할 수 있습니다.

### 하이퍼파라미터 튜닝

`train_lightweight_model.py`의 인자를 조정하여 다양한 설정으로 실험할 수 있습니다:

```bash
python train_lightweight_model.py \
    --model_type lightweight \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 1e-5
```

## 📊 성능 모니터링

### 훈련 과정 모니터링
- Loss 및 Accuracy 그래프 자동 생성
- 최고 성능 모델 자동 저장
- 상세한 분류 리포트 제공

### 추론 속도 분석
- 평균, 표준편차, 최소/최대 시간 측정
- 모델별 속도 향상 배수 계산
- 시간 분포 박스플롯 제공

## 🚨 주의사항

1. **데이터 경로**: `DATA/email_data/` 경로에 데이터가 있는지 확인
2. **GPU 메모리**: 배치 크기를 GPU 메모리에 맞게 조정
3. **토크나이저 호환성**: DistilBERT는 `token_type_ids`를 사용하지 않음
4. **모델 저장**: 충분한 디스크 공간 확보

## 🤝 기여 방법

1. 새로운 경량화 기법 구현
2. 추가 평가 메트릭 개발
3. 시각화 개선
4. 문서화 향상

## 📝 라이선스

이 프로젝트는 기존 MMTD 프로젝트와 동일한 라이선스를 따릅니다. 
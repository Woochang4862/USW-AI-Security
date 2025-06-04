# 멀티모달 스팸 탐지 모델 프로토타입 🔍

이 프로토타입은 ViT(Vision Transformer)와 BERT를 결합하여 이미지와 텍스트를 동시에 분석하는 멀티모달 스팸 탐지 모델을 구현합니다.

## 모델 구성 📊

1. **이미지 처리**: Vision Transformer (ViT)
   - 모델: google/vit-base-patch16-224
   - 이미지를 패치 단위로 분할하여 Transformer 구조로 처리

2. **텍스트 처리**: BERT
   - 모델: bert-base-uncased
   - 텍스트의 문맥적 의미를 임베딩으로 변환

3. **분류기**: Support Vector Machine (SVM)
   - ViT와 BERT의 특징을 결합하여 최종 분류 수행
   - RBF 커널 사용

## 필요 라이브러리 📚

```bash
pip install -r requirements.txt
```

## Google Colab에서 실행하기 🚀

1. `spam_detection_prototype.ipynb` 파일을 Google Colab에서 열기
2. Google Drive 마운트
3. 필요한 라이브러리 설치
4. 데이터셋 준비 및 모델 학습 실행

## 데이터셋 구조 📁

```
drive/MyDrive/spam_detection_project/
├── data/
│   ├── images/          # 스팸/정상 이미지
│   └── texts/           # 관련 텍스트 데이터
├── models/              # 저장된 모델
└── checkpoints/         # 학습 체크포인트
```

## 주요 기능 ⚙️

- 이미지와 텍스트의 멀티모달 특징 추출
- SVM을 사용한 스팸 분류
- 모델 학습 및 평가
- 학습된 모델 저장 및 로드

## 성능 평가 📈

다음 지표들을 사용하여 모델 성능을 평가합니다:
- Accuracy
- Precision
- Recall
- F1-Score

## 참고사항 ⚠️

1. 이 프로토타입은 실험적 구현이며, 실제 사용을 위해서는 추가적인 최적화가 필요할 수 있습니다.
2. GPU 사용을 권장합니다 (Google Colab Pro 환경 추천).
3. 대용량 데이터셋 처리 시 메모리 사용량에 주의하세요. 
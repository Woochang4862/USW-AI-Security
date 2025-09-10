#!/bin/bash

# MMTD 모델 실험 순차 실행 스크립트
# 각 실험이 완료된 후 다음 실험을 시작합니다

echo "🚀 MMTD 모델 실험 순차 실행 시작"
echo "📅 시작 시간: $(date)"
echo "=" * 60

# 실험 목록 정의
experiments=(
    "distilbert_mobilevit"
    "bert_deit"
    "bert_mobilevit"
    "bert_vit-tiny"
    "mobilebert_beit"
    "distilbert_beit"
    "tinybert_beit"
    "tinybert_vit-tiny"
    "mobilebert_mobilevit"
    "mobilebert_deit"
    "distilbert_deit"
    "bert_beit_pretrained"
    "tinybert_mobilevit"
    "tinybert_deit"
    "mobilebert_vit-tiny"
    "distilbert_vit-tiny"
)

# 공통 하이퍼파라미터
EPOCHS=3
LEARNING_RATE=5e-4

# 로그 디렉토리 생성
mkdir -p logs

# 각 실험 순차 실행
for i in "${!experiments[@]}"; do
    experiment="${experiments[$i]}"
    log_file="logs/train_${experiment}_$(date +%Y%m%d_%H%M%S).log"
    
    echo ""
    echo "🎯 실험 $((i+1))/${#experiments[@]}: $experiment"
    echo "📝 로그 파일: $log_file"
    echo "⏰ 시작 시간: $(date)"
    echo "-" * 50
    
    # 실험 실행
    python train_lightweight_model.py \
        --experiment "$experiment" \
        --epochs $EPOCHS \
        --learning_rate $LEARNING_RATE \
        > "$log_file" 2>&1
    
    # 실험 완료 확인
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ 실험 $experiment 완료 (종료 코드: $exit_code)"
    else
        echo "❌ 실험 $experiment 실패 (종료 코드: $exit_code)"
        echo "📋 로그 확인: $log_file"
    fi
    
    echo "⏰ 완료 시간: $(date)"
    echo "=" * 60
    
    # 다음 실험 전 잠시 대기 (선택사항)
    sleep 5
done

echo ""
echo "🎉 모든 실험 완료!"
echo "📅 종료 시간: $(date)"
echo "📊 총 실험 수: ${#experiments[@]}"
echo ""
echo "📋 로그 파일들:"
ls -la logs/train_*_$(date +%Y%m%d)*.log 2>/dev/null || echo "로그 파일을 찾을 수 없습니다."

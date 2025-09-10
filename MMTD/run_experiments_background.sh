#!/bin/bash

# MMTD 모델 실험 백그라운드 순차 실행 스크립트
# nohup으로 백그라운드 실행하면서도 순차적으로 진행

echo "🚀 MMTD 모델 실험 백그라운드 순차 실행 시작"
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

# 전체 실행 로그 파일
MASTER_LOG="logs/master_experiment_$(date +%Y%m%d_%H%M%S).log"

echo "📝 마스터 로그: $MASTER_LOG"
echo ""

# 각 실험 순차 실행 (백그라운드)
for i in "${!experiments[@]}"; do
    experiment="${experiments[$i]}"
    log_file="logs/train_${experiment}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "🎯 실험 $((i+1))/${#experiments[@]}: $experiment" | tee -a "$MASTER_LOG"
    echo "📝 로그 파일: $log_file" | tee -a "$MASTER_LOG"
    echo "⏰ 시작 시간: $(date)" | tee -a "$MASTER_LOG"
    echo "-" * 50 | tee -a "$MASTER_LOG"
    
    # 실험 실행 (백그라운드)
    nohup python train_lightweight_model.py \
        --experiment "$experiment" \
        --epochs $EPOCHS \
        --learning_rate $LEARNING_RATE \
        > "$log_file" 2>&1 &
    
    # 프로세스 ID 저장
    pid=$!
    echo "🔄 프로세스 ID: $pid" | tee -a "$MASTER_LOG"
    
    # 실험이 완료될 때까지 대기
    echo "⏳ 실험 완료 대기 중..." | tee -a "$MASTER_LOG"
    wait $pid
    
    # 실험 완료 확인
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ 실험 $experiment 완료 (종료 코드: $exit_code)" | tee -a "$MASTER_LOG"
    else
        echo "❌ 실험 $experiment 실패 (종료 코드: $exit_code)" | tee -a "$MASTER_LOG"
        echo "📋 로그 확인: $log_file" | tee -a "$MASTER_LOG"
    fi
    
    echo "⏰ 완료 시간: $(date)" | tee -a "$MASTER_LOG"
    echo "=" * 60 | tee -a "$MASTER_LOG"
    
    # 다음 실험 전 잠시 대기
    sleep 10
done

echo ""
echo "🎉 모든 실험 완료!" | tee -a "$MASTER_LOG"
echo "📅 종료 시간: $(date)" | tee -a "$MASTER_LOG"
echo "📊 총 실험 수: ${#experiments[@]}" | tee -a "$MASTER_LOG"
echo ""
echo "📋 로그 파일들:"
ls -la logs/train_*_$(date +%Y%m%d)*.log 2>/dev/null || echo "로그 파일을 찾을 수 없습니다."
echo ""
echo "📝 마스터 로그: $MASTER_LOG"

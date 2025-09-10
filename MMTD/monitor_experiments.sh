#!/bin/bash

# 실험 진행 상황 모니터링 스크립트

echo "📊 MMTD 실험 진행 상황 모니터링"
echo "=" * 50

# 현재 실행 중인 Python 프로세스 확인
echo "🔄 현재 실행 중인 실험:"
ps aux | grep "train_lightweight_model.py" | grep -v grep | while read line; do
    echo "  $line"
done

echo ""

# 최근 로그 파일들 확인
echo "📝 최근 로그 파일들 (최근 10개):"
ls -lt logs/train_*.log 2>/dev/null | head -10 | while read line; do
    echo "  $line"
done

echo ""

# 실험 결과 디렉토리 확인
echo "📁 실험 결과 디렉토리:"
if [ -d "outputs" ]; then
    ls -la outputs/ | while read line; do
        echo "  $line"
    done
else
    echo "  outputs 디렉토리가 없습니다."
fi

echo ""

# GPU 사용률 확인 (CUDA가 사용 가능한 경우)
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU 사용률:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
        echo "  GPU $line"
    done
fi

echo ""

# 시스템 리소스 확인
echo "💻 시스템 리소스:"
echo "  CPU 사용률: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "  메모리 사용률: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
echo "  디스크 사용률: $(df -h . | tail -1 | awk '{print $5}')"

echo ""
echo "⏰ 현재 시간: $(date)"

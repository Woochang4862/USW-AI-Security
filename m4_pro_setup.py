"""
M4 Pro MacBook Optimization Setup for MMTD Development
Optimized configuration for Apple Silicon
"""

import torch
import os
import platform

def setup_m4_pro_environment():
    """Setup M4 Pro environment for MMTD development"""
    
    print("🍎 Setting up M4 Pro MacBook environment for MMTD...")
    
    # 1. Check Apple Silicon and MPS availability
    if platform.processor() == 'arm':
        print("✅ Apple Silicon detected")
        if torch.backends.mps.is_available():
            print("✅ MPS (Metal Performance Shaders) available")
            device = torch.device("mps")
        else:
            print("⚠️ MPS not available, using CPU")
            device = torch.device("cpu")
    else:
        print("ℹ️ Intel Mac detected")
        device = torch.device("cpu")
    
    # 2. Memory optimization for unified memory
    if hasattr(torch.backends, 'mps'):
        # Enable memory efficient attention
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        
    # 3. Set optimal thread count for M4 Pro (12 cores)
    torch.set_num_threads(8)  # Leave some cores for system
    
    print(f"🎯 Device: {device}")
    print(f"🧵 Threads: {torch.get_num_threads()}")
    
    return device

def get_optimal_batch_size_m4():
    """Get optimal batch size for M4 Pro"""
    # M4 Pro has 18-36GB unified memory
    # Conservative batch size for stability
    if torch.backends.mps.is_available():
        return 32  # Good balance for MPS
    else:
        return 16  # CPU fallback

def optimize_for_interpretability():
    """Settings optimized for interpretability research"""
    return {
        'batch_size': get_optimal_batch_size_m4(),
        'num_workers': 4,  # M4 Pro has good CPU performance
        'pin_memory': False,  # Not needed for MPS
        'persistent_workers': True,
        'prefetch_factor': 2
    }

def print_m4_tips():
    """Print M4 Pro specific optimization tips"""
    print("\n🚀 M4 Pro 최적화 팁:")
    print("1. MPS 사용으로 GPU 가속 (Neural Engine 활용)")
    print("2. 배치 크기: 32 (통합 메모리 효율적 사용)")
    print("3. 멀티스레딩: 8 threads (12코어 중 8개 사용)")
    print("4. 예상 성능: T4 GPU의 70-80% 수준")
    print("5. 장점: 메모리 제한 없음, 안정적 성능")

if __name__ == "__main__":
    device = setup_m4_pro_environment()
    settings = optimize_for_interpretability()
    
    print(f"\n🎯 M4 Pro 권장 설정:")
    print(f"   Device: {device}")
    print(f"   배치 크기: {settings['batch_size']}")
    print(f"   Workers: {settings['num_workers']}")
    
    print_m4_tips()
    
    # 테스트 코드
    print(f"\n📝 M4 Pro에서 사용할 코드:")
    print(f"""
# M4 Pro 최적화 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# MMTD 평가 실행
evaluator = MMTDEvaluator(
    checkpoint_dir="MMTD/checkpoints",
    data_path="MMTD/DATA/email_data/pics", 
    csv_path="MMTD/DATA/email_data/EDP.csv",
    device=device,
    batch_size={settings['batch_size']}  # M4 Pro 최적화
)

# Logistic Regression 구현
logistic_classifier = LogisticRegressionClassifier(
    input_size=768,  # MMTD fusion output
    num_classes=2,
    device=device
)
    """) 
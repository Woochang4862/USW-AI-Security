import torch
import sys
import os
from pathlib import Path

# 현재 디렉터리를 Python path에 추가
sys.path.append('.')

try:
    from models import MMTD
    print("✅ MMTD 모델 클래스 import 성공")
except ImportError as e:
    print(f"❌ MMTD 모델 import 실패: {e}")
    sys.exit(1)

class RealModelTester:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🔧 디바이스: {self.device}")
        
    def test_model_creation(self):
        """기본 MMTD 모델 생성 테스트"""
        print("\n🏗️ MMTD 모델 생성 테스트...")
        
        try:
            # 기본 모델 생성 시도
            model = MMTD(
                bert_pretrain_weight='bert-base-multilingual-cased',
                beit_pretrain_weight='microsoft/dit-base'
            )
            
            print("✅ 기본 MMTD 모델 생성 성공")
            print(f"   모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
            
            # 모델을 평가 모드로 설정
            model.eval()
            model.to(self.device)
            print(f"✅ 모델을 {self.device}에 로딩 완료")
            
            return model
            
        except Exception as e:
            print(f"❌ 모델 생성 실패: {e}")
            return None
    
    def test_checkpoint_loading(self, model):
        """체크포인트 로딩 테스트"""
        print("\n💾 체크포인트 로딩 테스트...")
        
        checkpoint_path = "checkpoints/fold1/checkpoint-939/pytorch_model.bin"
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ 체크포인트 파일 없음: {checkpoint_path}")
            return False
            
        try:
            print(f"📁 체크포인트 로딩 시도: {checkpoint_path}")
            
            # 체크포인트 로드
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"✅ 체크포인트 파일 로딩 성공")
            print(f"   체크포인트 키 수: {len(checkpoint.keys())}")
            
            # 몇 개 키 확인
            print("   상위 5개 키:")
            for i, key in enumerate(list(checkpoint.keys())[:5]):
                shape = checkpoint[key].shape if hasattr(checkpoint[key], 'shape') else 'scalar'
                print(f"     {i+1}. {key}: {shape}")
            
            # 모델에 로딩 시도
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            
            print(f"✅ 모델에 체크포인트 로딩 완료")
            print(f"   Missing keys: {len(missing_keys)}")
            print(f"   Unexpected keys: {len(unexpected_keys)}")
            
            if missing_keys:
                print("   일부 missing keys:")
                for key in missing_keys[:3]:
                    print(f"     - {key}")
                    
            if unexpected_keys:
                print("   일부 unexpected keys:")
                for key in unexpected_keys[:3]:
                    print(f"     - {key}")
            
            return True
            
        except Exception as e:
            print(f"❌ 체크포인트 로딩 실패: {e}")
            return False
    
    def test_model_forward(self, model):
        """모델 forward pass 테스트"""
        print("\n🔄 모델 Forward Pass 테스트...")
        
        try:
            # 더미 입력 데이터 생성
            batch_size = 1
            seq_length = 64
            image_size = 224
            
            dummy_input = {
                'input_ids': torch.randint(0, 1000, (batch_size, seq_length), dtype=torch.long),
                'attention_mask': torch.ones(batch_size, seq_length, dtype=torch.long),
                'token_type_ids': torch.zeros(batch_size, seq_length, dtype=torch.long),
                'pixel_values': torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32)
            }
            
            # 디바이스로 이동
            for key in dummy_input:
                dummy_input[key] = dummy_input[key].to(self.device)
            
            print(f"📊 입력 데이터 생성 완료:")
            for key, tensor in dummy_input.items():
                print(f"   {key}: {tensor.shape} on {tensor.device}")
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**dummy_input)
                
            print(f"✅ Forward pass 성공!")
            print(f"   출력 형태: {outputs.logits.shape}")
            print(f"   예측 확률: {torch.softmax(outputs.logits, dim=-1)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Forward pass 실패: {e}")
            import traceback
            print(f"   상세 오류: {traceback.format_exc()}")
            return False
    
    def run_full_test(self):
        """전체 테스트 실행"""
        print("🚀 실제 MMTD 모델 로딩 및 동작 테스트 시작")
        print("="*60)
        
        # 1. 모델 생성
        model = self.test_model_creation()
        if model is None:
            print("❌ 모델 생성 실패로 테스트 중단")
            return False
        
        # 2. 체크포인트 로딩
        checkpoint_loaded = self.test_checkpoint_loading(model)
        if not checkpoint_loaded:
            print("⚠️ 체크포인트 로딩 실패, 기본 모델로 계속")
        
        # 3. Forward pass 테스트
        forward_success = self.test_model_forward(model)
        
        # 결과 요약
        print("\n📋 테스트 결과 요약:")
        print("-" * 40)
        print(f"모델 생성: ✅ 성공")
        print(f"체크포인트 로딩: {'✅ 성공' if checkpoint_loaded else '❌ 실패'}")
        print(f"Forward pass: {'✅ 성공' if forward_success else '❌ 실패'}")
        
        if checkpoint_loaded and forward_success:
            print("\n🎉 실제 MMTD 모델 로딩 및 동작 완전 성공!")
            print("   이제 실제 attention 분석을 시작할 수 있습니다.")
            return True
        else:
            print("\n⚠️ 일부 문제가 있지만 진단 정보를 얻었습니다.")
            return False


if __name__ == "__main__":
    tester = RealModelTester()
    success = tester.run_full_test()
    
    if success:
        print("\n✅ 다음 단계: 실제 attention 가중치 추출 구현")
    else:
        print("\n🔧 다음 단계: 식별된 문제들 해결 필요") 
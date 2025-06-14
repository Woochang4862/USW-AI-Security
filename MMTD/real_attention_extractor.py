import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('.')

from models import MMTD
from transformers import AutoTokenizer, AutoFeatureExtractor
import warnings
warnings.filterwarnings("ignore")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class RealAttentionExtractor:
    """
    실제 MMTD 모델에서 attention 가중치를 추출하는 클래스
    - BERT text encoder attention
    - BEiT image encoder attention  
    - Multi-modality fusion layer attention
    """
    
    def __init__(self, checkpoint_path: str = "checkpoints/fold1/checkpoint-939/pytorch_model.bin"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        
        # 모델 및 전처리기 초기화
        self.model = None
        self.tokenizer = None
        self.feature_extractor = None
        
        # Attention 저장용
        self.attention_weights = {}
        self.hooks = []
        
        print(f"🔧 실제 MMTD Attention 추출기 초기화 (디바이스: {self.device})")
    
    def load_model_and_checkpoint(self):
        """실제 MMTD 모델과 체크포인트 로딩"""
        print("\n📂 실제 MMTD 모델 로딩...")
        
        try:
            # 모델 생성
            self.model = MMTD(
                bert_pretrain_weight='bert-base-multilingual-cased',
                beit_pretrain_weight='microsoft/dit-base'
            )
            
            # 체크포인트 로딩
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint, strict=False)
            
            # 모델을 평가 모드로 설정하고 디바이스로 이동
            self.model.eval()
            self.model.to(self.device)
            
            print(f"✅ MMTD 모델 로딩 성공 ({sum(p.numel() for p in self.model.parameters()):,} 파라미터)")
            print(f"   Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
            
            # 토크나이저와 특징 추출기 로딩
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/dit-base')
            
            print("✅ 토크나이저 및 특징 추출기 로딩 완료")
            return True
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            return False
    
    def register_attention_hooks(self):
        """Attention 가중치를 캡처하기 위한 Hook 등록"""
        print("\n🔗 Attention Hook 등록...")
        
        def bert_attention_hook(name):
            def hook(module, input, output):
                # BERT self-attention의 attention_probs 캡처
                if hasattr(module, 'attention_probs') and module.attention_probs is not None:
                    self.attention_weights[f'bert_{name}'] = module.attention_probs.detach().cpu()
                # output이 튜플인 경우 attention weights 확인
                elif isinstance(output, tuple) and len(output) >= 2:
                    # output[1]이 attention weights인 경우가 많음
                    if hasattr(output[1], 'shape') and len(output[1].shape) >= 3:
                        self.attention_weights[f'bert_{name}'] = output[1].detach().cpu()
            return hook
        
        def beit_attention_hook(name):
            def hook(module, input, output):
                # BEiT self-attention의 attention_probs 캡처
                if hasattr(module, 'attention_probs') and module.attention_probs is not None:
                    self.attention_weights[f'beit_{name}'] = module.attention_probs.detach().cpu()
                # output이 튜플인 경우
                elif isinstance(output, tuple) and len(output) >= 2:
                    if hasattr(output[1], 'shape') and len(output[1].shape) >= 3:
                        self.attention_weights[f'beit_{name}'] = output[1].detach().cpu()
            return hook
        
        def attention_probs_hook(name, module_type):
            """실제 attention probabilities을 직접 캡처"""
            def hook(module, input, output):
                # attention_probs 속성이 있는지 확인
                if hasattr(module, 'attention_probs'):
                    attention_probs = module.attention_probs
                    if attention_probs is not None:
                        self.attention_weights[f'{module_type}_{name}'] = attention_probs.detach().cpu()
                        print(f"   캡처됨: {module_type}_{name} - {attention_probs.shape}")
                
                # output에서 attention weights 찾기
                if isinstance(output, tuple):
                    for i, out in enumerate(output):
                        if hasattr(out, 'shape') and len(out.shape) >= 3:
                            # attention weights로 보이는 텐서
                            if 'attention' in str(type(out)).lower() or len(out.shape) == 4:
                                self.attention_weights[f'{module_type}_{name}_output_{i}'] = out.detach().cpu()
                                print(f"   캡처됨: {module_type}_{name}_output_{i} - {out.shape}")
            return hook
        
        # BERT text encoder attention hooks - 더 구체적으로
        print("  BERT 레이어 Hook 등록...")
        for i, layer in enumerate(self.model.text_encoder.bert.encoder.layer):
            # self-attention hook
            if hasattr(layer.attention, 'self'):
                hook = layer.attention.self.register_forward_hook(
                    attention_probs_hook(f'layer_{i}', 'bert')
                )
                self.hooks.append(hook)
        
        # BEiT image encoder attention hooks - 더 구체적으로  
        print("  BEiT 레이어 Hook 등록...")
        for i, layer in enumerate(self.model.image_encoder.beit.encoder.layer):
            # self-attention hook
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'attention'):
                hook = layer.attention.attention.register_forward_hook(
                    attention_probs_hook(f'layer_{i}', 'beit')
                )
                self.hooks.append(hook)
        
        # Multi-modality transformer에 대해서도 더 정교한 hook
        print("  융합 레이어 Hook 등록...")
        def fusion_attention_hook(name):
            def hook(module, input, output):
                # TransformerEncoderLayer의 self-attention 부분 캡처
                if len(input) > 0:
                    self.attention_weights[f'fusion_{name}_input'] = input[0].detach().cpu()
                if isinstance(output, torch.Tensor):
                    self.attention_weights[f'fusion_{name}_output'] = output.detach().cpu()
            return hook
        
        hook = self.model.multi_modality_transformer_layer.register_forward_hook(
            fusion_attention_hook('fusion')
        )
        self.hooks.append(hook)
        
        print(f"✅ {len(self.hooks)}개 Attention Hook 등록 완료")
    
    def create_sample_input(self, text: str = "스팸 이메일 테스트 샘플", image_size: int = 224):
        """샘플 입력 데이터 생성"""
        print(f"\n📝 샘플 입력 생성: '{text}'")
        
        # 텍스트 토크나이징
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # 더미 이미지 생성 (실제로는 이메일 이미지 사용)
        dummy_image = torch.randn(1, 3, image_size, image_size)
        
        # 입력 데이터 구성
        inputs = {
            'input_ids': text_inputs['input_ids'].to(self.device),
            'attention_mask': text_inputs['attention_mask'].to(self.device),
            'token_type_ids': torch.zeros_like(text_inputs['input_ids']).to(self.device),
            'pixel_values': dummy_image.to(self.device)
        }
        
        print(f"✅ 입력 데이터 생성 완료:")
        for key, tensor in inputs.items():
            print(f"   {key}: {tensor.shape}")
        
        return inputs, text
    
    def extract_attention_weights(self, inputs: Dict[str, torch.Tensor]):
        """실제 attention 가중치 추출"""
        print("\n🔍 실제 Attention 가중치 추출 중...")
        
        # 이전 attention 가중치 초기화
        self.attention_weights.clear()
        
        # 모델 forward pass (attention hook이 자동으로 호출됨)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 결과 확인
        prediction = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(prediction, dim=-1).item()
        confidence = prediction.max().item()
        
        print(f"✅ Attention 추출 완료!")
        print(f"   추출된 attention 종류: {len(self.attention_weights)}개")
        print(f"   예측: {'스팸' if predicted_class == 1 else '정상'} (신뢰도: {confidence:.4f})")
        
        for key, attention in self.attention_weights.items():
            if hasattr(attention, 'shape'):
                print(f"   {key}: {attention.shape}")
        
        return {
            'attention_weights': self.attention_weights.copy(),
            'prediction': prediction.cpu().numpy(),
            'predicted_class': predicted_class,
            'confidence': confidence
        }
    
    def visualize_text_attention(self, text: str, attention_data: Dict, layer_idx: int = 11):
        """텍스트 attention 시각화"""
        print(f"\n📊 텍스트 Attention 시각화 (레이어 {layer_idx})")
        
        # 토큰 분리
        tokens = self.tokenizer.tokenize(text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # BERT attention 데이터 찾기
        bert_key = f'bert_layer_{layer_idx}'
        if bert_key not in attention_data['attention_weights']:
            print(f"⚠️ {bert_key} attention을 찾을 수 없음")
            return
        
        attention = attention_data['attention_weights'][bert_key]
        
        # Attention 평균 (모든 헤드의 평균)
        if len(attention.shape) == 4:  # [batch, heads, seq, seq]
            attention_avg = attention[0].mean(dim=0)  # 헤드 평균
        else:
            attention_avg = attention[0]
        
        # 시각화
        plt.figure(figsize=(12, 10))
        
        # 텍스트 길이에 맞게 조정
        seq_len = min(len(tokens), attention_avg.shape[0])
        attention_matrix = attention_avg[:seq_len, :seq_len]
        
        sns.heatmap(
            attention_matrix, 
            xticklabels=tokens[:seq_len], 
            yticklabels=tokens[:seq_len],
            cmap='Blues',
            annot=False,
            fmt='.3f'
        )
        
        plt.title(f'BERT 텍스트 Attention (레이어 {layer_idx})\n'
                 f'예측: {"스팸" if attention_data["predicted_class"] == 1 else "정상"} '
                 f'(신뢰도: {attention_data["confidence"]:.3f})')
        plt.xlabel('Tokens (To)')
        plt.ylabel('Tokens (From)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        filename = f'real_bert_attention_layer_{layer_idx}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 텍스트 attention 시각화 저장: {filename}")
    
    def visualize_image_attention(self, attention_data: Dict, layer_idx: int = 11):
        """이미지 attention 시각화"""
        print(f"\n🖼️ 이미지 Attention 시각화 (레이어 {layer_idx})")
        
        # BEiT attention 데이터 찾기
        beit_key = f'beit_layer_{layer_idx}'
        if beit_key not in attention_data['attention_weights']:
            print(f"⚠️ {beit_key} attention을 찾을 수 없음")
            return
        
        attention = attention_data['attention_weights'][beit_key]
        
        # Attention 평균
        if len(attention.shape) == 4:  # [batch, heads, patches, patches]
            attention_avg = attention[0].mean(dim=0)  # 헤드 평균
        else:
            attention_avg = attention[0]
        
        # 패치 수 계산 (보통 14x14 = 196개 패치)
        num_patches = attention_avg.shape[0]
        patch_size = int(np.sqrt(num_patches))
        
        # CLS 토큰 제외하고 이미지 패치만 사용
        if attention_avg.shape[0] > patch_size**2:
            # CLS 토큰이 있는 경우
            image_attention = attention_avg[1:, 1:]  # CLS 제거
        else:
            image_attention = attention_avg
        
        # 패치 attention을 이미지 형태로 재구성
        actual_patches = int(np.sqrt(image_attention.shape[0]))
        if actual_patches**2 == image_attention.shape[0]:
            # 각 패치에서 자기 자신에 대한 attention
            self_attention = torch.diag(image_attention)
            attention_map = self_attention.reshape(actual_patches, actual_patches)
        else:
            # 전체 attention의 평균
            attention_map = image_attention.mean(dim=1).reshape(actual_patches, actual_patches)
        
        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # 원본 attention 히트맵
        im1 = axes[0].imshow(attention_map, cmap='viridis', interpolation='nearest')
        axes[0].set_title(f'BEiT 이미지 Attention Map (레이어 {layer_idx})')
        axes[0].set_xlabel('이미지 패치 (X)')
        axes[0].set_ylabel('이미지 패치 (Y)')
        plt.colorbar(im1, ax=axes[0])
        
        # 업샘플링된 attention 맵
        upsampled = torch.nn.functional.interpolate(
            attention_map.unsqueeze(0).unsqueeze(0), 
            size=(224, 224), 
            mode='bilinear'
        )[0, 0]
        
        im2 = axes[1].imshow(upsampled, cmap='hot', interpolation='bilinear')
        axes[1].set_title('고해상도 Attention 오버레이')
        axes[1].set_xlabel('픽셀 X')
        axes[1].set_ylabel('픽셀 Y')
        plt.colorbar(im2, ax=axes[1])
        
        plt.suptitle(f'실제 BEiT 이미지 Attention 분석\n'
                    f'예측: {"스팸" if attention_data["predicted_class"] == 1 else "정상"} '
                    f'(신뢰도: {attention_data["confidence"]:.3f})')
        plt.tight_layout()
        
        filename = f'real_beit_attention_layer_{layer_idx}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 이미지 attention 시각화 저장: {filename}")
    
    def analyze_cross_modal_attention(self, attention_data: Dict):
        """크로스 모달 attention 분석"""
        print(f"\n🔗 크로스 모달 Attention 분석")
        
        # 융합 레이어 데이터 확인
        fusion_key = 'fusion_fusion'
        if fusion_key not in attention_data['attention_weights']:
            print(f"⚠️ 융합 레이어 attention을 찾을 수 없음")
            return
        
        fusion_data = attention_data['attention_weights'][fusion_key]
        
        print(f"✅ 융합 레이어 데이터 형태: {fusion_data.shape}")
        
        # 간단한 통계 분석
        text_region = fusion_data[:, :64]  # 텍스트 영역 (가정)
        image_region = fusion_data[:, 64:]  # 이미지 영역 (가정)
        
        text_activation = text_region.mean().item()
        image_activation = image_region.mean().item()
        
        print(f"   텍스트 영역 평균 활성화: {text_activation:.4f}")
        print(f"   이미지 영역 평균 활성화: {image_activation:.4f}")
        print(f"   모달리티 비율 (텍스트/이미지): {text_activation/image_activation:.3f}")
    
    def cleanup_hooks(self):
        """등록된 Hook 정리"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        print("🧹 Hook 정리 완료")
    
    def run_real_attention_analysis(self, text: str = "무료 상품 받기! 지금 클릭하세요!"):
        """실제 attention 분석 전체 파이프라인 실행"""
        print("🚀 실제 MMTD Attention 분석 시작")
        print("="*60)
        
        try:
            # 1. 모델 로딩
            if not self.load_model_and_checkpoint():
                return False
            
            # 2. Hook 등록
            self.register_attention_hooks()
            
            # 3. 샘플 입력 생성
            inputs, processed_text = self.create_sample_input(text)
            
            # 4. Attention 추출
            attention_data = self.extract_attention_weights(inputs)
            
            # 5. 시각화
            self.visualize_text_attention(processed_text, attention_data, layer_idx=11)
            self.visualize_image_attention(attention_data, layer_idx=11)
            self.analyze_cross_modal_attention(attention_data)
            
            print("\n🎉 실제 MMTD Attention 분석 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            # 7. 정리
            self.cleanup_hooks()


if __name__ == "__main__":
    # 실제 attention 분석 실행
    extractor = RealAttentionExtractor()
    
    # 여러 샘플 테스트
    test_samples = [
        "무료 상품을 받으세요! 지금 클릭하세요!",  # 스팸 예상
        "안녕하세요. 회의 일정을 알려드립니다.",     # 정상 예상
        "FREE MONEY! Click here NOW!!!",          # 영어 스팸
        "おめでとうございます！賞品が当選"          # 일본어 스팸
    ]
    
    for i, sample_text in enumerate(test_samples):
        print(f"\n{'='*80}")
        print(f"샘플 {i+1}: {sample_text}")
        print(f"{'='*80}")
        
        success = extractor.run_real_attention_analysis(sample_text)
        if not success:
            print(f"⚠️ 샘플 {i+1} 분석 실패")
            break
    
    print("\n✅ 모든 실제 attention 분석 완료!") 
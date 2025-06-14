import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('.')

from models import MMTD
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class RealAttentionExtractorV2:
    """
    실제 MMTD 모델에서 attention 가중치를 정확히 추출하는 개선된 클래스
    - output_attentions=True 사용
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
        
        print(f"🔧 실제 MMTD Attention 추출기 V2 초기화 (디바이스: {self.device})")
    
    def load_model_and_checkpoint(self):
        """실제 MMTD 모델과 체크포인트 로딩"""
        print("\n📂 실제 MMTD 모델 로딩...")
        
        try:
            # 모델 생성
            self.model = MMTD(
                bert_pretrain_weight='bert-base-multilingual-cased',
                beit_pretrain_weight='microsoft/dit-base'
            )
            
            # ★ 핵심: output_attentions=True 설정
            self.model.text_encoder.config.output_attentions = True
            self.model.image_encoder.config.output_attentions = True
            
            # 체크포인트 로딩
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint, strict=False)
            
            # 모델을 평가 모드로 설정하고 디바이스로 이동
            self.model.eval()
            self.model.to(self.device)
            
            print(f"✅ MMTD 모델 로딩 성공 ({sum(p.numel() for p in self.model.parameters()):,} 파라미터)")
            print(f"   Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
            print(f"   🎯 Attention 출력 활성화됨: BERT={self.model.text_encoder.config.output_attentions}, BEiT={self.model.image_encoder.config.output_attentions}")
            
            # 토크나이저 로딩
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
            
            print("✅ 토크나이저 로딩 완료")
            return True
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            return False
    
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
    
    def extract_attention_weights_step_by_step(self, inputs: Dict[str, torch.Tensor]):
        """단계별로 attention 가중치 추출"""
        print("\n🔍 단계별 실제 Attention 가중치 추출...")
        
        attention_data = {}
        
        with torch.no_grad():
            # 1. BERT 텍스트 인코더 실행
            print("   1. BERT 텍스트 인코더 실행...")
            text_inputs = {k: v for k, v in inputs.items() if k != 'pixel_values'}
            text_outputs = self.model.text_encoder(**text_inputs)
            
            # BERT attention 추출
            if hasattr(text_outputs, 'attentions') and text_outputs.attentions is not None:
                attention_data['bert_attentions'] = [att.cpu() for att in text_outputs.attentions]
                print(f"     ✅ BERT attention 추출 완료: {len(text_outputs.attentions)}개 레이어")
                print(f"     마지막 레이어 형태: {text_outputs.attentions[-1].shape}")
            else:
                print(f"     ❌ BERT attention 없음")
            
            # 2. BEiT 이미지 인코더 실행
            print("   2. BEiT 이미지 인코더 실행...")
            image_outputs = self.model.image_encoder(pixel_values=inputs['pixel_values'])
            
            # BEiT attention 추출
            if hasattr(image_outputs, 'attentions') and image_outputs.attentions is not None:
                attention_data['beit_attentions'] = [att.cpu() for att in image_outputs.attentions]
                print(f"     ✅ BEiT attention 추출 완료: {len(image_outputs.attentions)}개 레이어")
                print(f"     마지막 레이어 형태: {image_outputs.attentions[-1].shape}")
            else:
                print(f"     ❌ BEiT attention 없음")
            
            # 3. 융합 레이어 실행 (전체 모델)
            print("   3. 전체 모델 실행...")
            full_outputs = self.model(**inputs)
            
            # 예측 결과
            prediction = torch.softmax(full_outputs.logits, dim=-1)
            predicted_class = torch.argmax(prediction, dim=-1).item()
            confidence = prediction.max().item()
            
            attention_data.update({
                'prediction': prediction.cpu().numpy(),
                'predicted_class': predicted_class,
                'confidence': confidence,
                'text_hidden_states': text_outputs.hidden_states,
                'image_hidden_states': image_outputs.hidden_states
            })
            
            print(f"   🎯 예측 결과: {'스팸' if predicted_class == 1 else '정상'} (신뢰도: {confidence:.4f})")
        
        return attention_data
    
    def visualize_bert_attention(self, text: str, attention_data: Dict, layer_idx: int = 11):
        """BERT 텍스트 attention 시각화"""
        print(f"\n📊 BERT 텍스트 Attention 시각화 (레이어 {layer_idx})")
        
        if 'bert_attentions' not in attention_data:
            print("❌ BERT attention 데이터 없음")
            return
        
        # 토큰 분리
        tokens = self.tokenizer.tokenize(text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # 지정된 레이어의 attention
        attention = attention_data['bert_attentions'][layer_idx]  # [batch, heads, seq, seq]
        
        # 평균 attention (모든 헤드의 평균)
        attention_avg = attention[0].mean(dim=0)  # [seq, seq]
        
        # 시각화
        plt.figure(figsize=(12, 10))
        
        # 텍스트 길이에 맞게 조정
        seq_len = min(len(tokens), attention_avg.shape[0])
        attention_matrix = attention_avg[:seq_len, :seq_len].numpy()
        
        sns.heatmap(
            attention_matrix, 
            xticklabels=tokens[:seq_len], 
            yticklabels=tokens[:seq_len],
            cmap='Blues',
            annot=False,
            fmt='.3f',
            cbar_kws={'label': 'Attention 가중치'}
        )
        
        plt.title(f'실제 BERT 텍스트 Attention (레이어 {layer_idx})\n'
                 f'예측: {"스팸" if attention_data["predicted_class"] == 1 else "정상"} '
                 f'(신뢰도: {attention_data["confidence"]:.3f})')
        plt.xlabel('To Tokens')
        plt.ylabel('From Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        filename = f'real_bert_attention_v2_layer_{layer_idx}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ BERT attention 시각화 저장: {filename}")
    
    def visualize_beit_attention(self, attention_data: Dict, layer_idx: int = 11):
        """BEiT 이미지 attention 시각화"""
        print(f"\n🖼️ BEiT 이미지 Attention 시각화 (레이어 {layer_idx})")
        
        if 'beit_attentions' not in attention_data:
            print("❌ BEiT attention 데이터 없음")
            return
        
        # 지정된 레이어의 attention
        attention = attention_data['beit_attentions'][layer_idx]  # [batch, heads, patches, patches]
        
        # 평균 attention (모든 헤드의 평균)
        attention_avg = attention[0].mean(dim=0)  # [patches, patches]
        
        # CLS 토큰 제외 (첫 번째 토큰이 CLS인 경우)
        if attention_avg.shape[0] > 196:  # 14x14=196 패치 + CLS
            image_attention = attention_avg[1:, 1:]  # CLS 제거
        else:
            image_attention = attention_avg
        
        # 패치 attention을 이미지 형태로 재구성
        num_patches = int(np.sqrt(image_attention.shape[0]))
        
        if num_patches**2 == image_attention.shape[0]:
            # 각 패치의 self-attention (대각선)
            self_attention = torch.diag(image_attention)
            attention_map = self_attention.reshape(num_patches, num_patches).numpy()
        else:
            # 전체 attention의 평균
            attention_map = image_attention.mean(dim=1).reshape(num_patches, num_patches).numpy()
        
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
            torch.tensor(attention_map).unsqueeze(0).unsqueeze(0).float(), 
            size=(224, 224), 
            mode='bilinear'
        )[0, 0].numpy()
        
        im2 = axes[1].imshow(upsampled, cmap='hot', interpolation='bilinear')
        axes[1].set_title('고해상도 Attention 오버레이')
        axes[1].set_xlabel('픽셀 X')
        axes[1].set_ylabel('픽셀 Y')
        plt.colorbar(im2, ax=axes[1])
        
        plt.suptitle(f'실제 BEiT 이미지 Attention 분석\n'
                    f'예측: {"스팸" if attention_data["predicted_class"] == 1 else "정상"} '
                    f'(신뢰도: {attention_data["confidence"]:.3f})')
        plt.tight_layout()
        
        filename = f'real_beit_attention_v2_layer_{layer_idx}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ BEiT attention 시각화 저장: {filename}")
    
    def analyze_attention_patterns(self, attention_data: Dict):
        """Attention 패턴 분석"""
        print(f"\n🔬 Attention 패턴 분석")
        
        if 'bert_attentions' in attention_data:
            # BERT attention 통계
            bert_attention = attention_data['bert_attentions'][-1][0]  # 마지막 레이어, 첫 번째 샘플
            bert_entropy = -torch.sum(bert_attention * torch.log(bert_attention + 1e-12), dim=-1).mean()
            print(f"   BERT Attention 엔트로피: {bert_entropy:.4f}")
            print(f"   BERT Attention 집중도 (최대값): {bert_attention.max():.4f}")
        
        if 'beit_attentions' in attention_data:
            # BEiT attention 통계
            beit_attention = attention_data['beit_attentions'][-1][0]  # 마지막 레이어, 첫 번째 샘플
            beit_entropy = -torch.sum(beit_attention * torch.log(beit_attention + 1e-12), dim=-1).mean()
            print(f"   BEiT Attention 엔트로피: {beit_entropy:.4f}")
            print(f"   BEiT Attention 집중도 (최대값): {beit_attention.max():.4f}")
        
        # 예측 신뢰도와 attention 패턴 관계
        confidence = attention_data['confidence']
        print(f"   예측 신뢰도: {confidence:.4f}")
        
        if confidence > 0.9:
            print("   📊 높은 신뢰도: 명확한 패턴 탐지")
        elif confidence > 0.7:
            print("   📊 중간 신뢰도: 모호한 패턴")
        else:
            print("   📊 낮은 신뢰도: 불확실한 패턴")
    
    def run_real_attention_analysis(self, text: str = "무료 상품 받기! 지금 클릭하세요!"):
        """실제 attention 분석 전체 파이프라인 실행"""
        print("🚀 실제 MMTD Attention 분석 V2 시작")
        print("="*60)
        
        try:
            # 1. 모델 로딩
            if not self.load_model_and_checkpoint():
                return False
            
            # 2. 샘플 입력 생성
            inputs, processed_text = self.create_sample_input(text)
            
            # 3. Attention 추출
            attention_data = self.extract_attention_weights_step_by_step(inputs)
            
            # 4. 시각화
            self.visualize_bert_attention(processed_text, attention_data, layer_idx=11)
            self.visualize_beit_attention(attention_data, layer_idx=11)
            
            # 5. 패턴 분석
            self.analyze_attention_patterns(attention_data)
            
            print("\n🎉 실제 MMTD Attention 분석 V2 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # 실제 attention 분석 실행
    extractor = RealAttentionExtractorV2()
    
    # 테스트 샘플들
    test_samples = [
        "무료 상품을 받으세요! 지금 클릭하세요!",  # 한국어 스팸
        "안녕하세요. 회의 일정을 알려드립니다.",     # 한국어 정상
        "FREE MONEY! Click here NOW!!!",          # 영어 스팸
    ]
    
    for i, sample_text in enumerate(test_samples):
        print(f"\n{'='*80}")
        print(f"샘플 {i+1}: {sample_text}")
        print(f"{'='*80}")
        
        success = extractor.run_real_attention_analysis(sample_text)
        if not success:
            print(f"⚠️ 샘플 {i+1} 분석 실패")
            break
    
    print("\n✅ 모든 실제 attention 분석 V2 완료!") 
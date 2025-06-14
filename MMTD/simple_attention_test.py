import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import MMTD
from Email_dataset import EDPDataset, EDPCollator
from utils import SplitData
import warnings
warnings.filterwarnings('ignore')

class SimpleAttentionAnalyzer:
    """
    간단한 Attention 분석기 - PyTorch 버전 호환성 문제 우회
    """
    
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 기존 모델을 로드하되 사전 훈련된 가중치 없이 초기화
        print("모델 초기화 중...")
        self.model = MMTD()  # 사전 훈련된 가중치 없이 초기화
        
        # 체크포인트 로드
        self.load_checkpoint()
        
        # 데이터 로드
        self.load_test_data()
        
    def load_checkpoint(self):
        """체크포인트를 로드합니다."""
        model_path = os.path.join(self.checkpoint_path, "pytorch_model.bin")
        if os.path.exists(model_path):
            print(f"체크포인트 로딩: {model_path}")
            state_dict = torch.load(model_path, map_location='cpu')
            
            # 엄격하지 않은 로딩으로 가능한 한 많이 로드
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            print(f"로딩 완료 - Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
            
            self.model.eval()
            self.model.to(self.device)
        else:
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {model_path}")
    
    def load_test_data(self):
        """테스트 데이터를 로드합니다."""
        print("테스트 데이터 로딩...")
        split_data = SplitData('DATA/email_data/EDP.csv', 5)
        train_df, test_df = split_data()
        
        # 작은 샘플만 사용
        test_sample = test_df.head(5)
        self.test_dataset = EDPDataset('DATA/email_data/pics', test_sample)
        self.collator = EDPCollator()
        
        print(f"테스트 샘플 수: {len(test_sample)}")
    
    def analyze_modality_contributions(self, sample_idx=0):
        """모달리티별 기여도를 분석합니다."""
        print(f"\n샘플 {sample_idx} 모달리티 기여도 분석:")
        
        sample = self.test_dataset[sample_idx]
        batch = self.collator([sample])
        
        # GPU로 이동
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        with torch.no_grad():
            # 전체 예측
            full_output = self.model(**batch)
            full_probs = torch.softmax(full_output.logits, dim=-1)
            
            # 텍스트와 이미지 인코더의 출력 얻기
            text_outputs = self.model.text_encoder(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask']
            )
            image_outputs = self.model.image_encoder(
                pixel_values=batch['pixel_values']
            )
            
            # Hidden states 추출
            text_hidden = text_outputs.hidden_states[12]  # 마지막 레이어
            image_hidden = image_outputs.hidden_states[12]
            
            # 모달리티 임베딩 추가
            text_hidden_mod = text_hidden + torch.zeros(text_hidden.size()).to(self.device)
            image_hidden_mod = image_hidden + torch.ones(image_hidden.size()).to(self.device)
            
            # 텍스트만 사용 (이미지를 0으로 마스킹)
            text_only_fused = torch.cat([
                text_hidden_mod,
                torch.zeros_like(image_hidden_mod)
            ], dim=1)
            text_only_output = self.model.multi_modality_transformer_layer(text_only_fused)
            text_only_pooled = self.model.pooler(text_only_output[:, 0, :])
            text_only_logits = self.model.classifier(text_only_pooled)
            text_only_probs = torch.softmax(text_only_logits, dim=-1)
            
            # 이미지만 사용 (텍스트를 0으로 마스킹)
            image_only_fused = torch.cat([
                torch.zeros_like(text_hidden_mod),
                image_hidden_mod
            ], dim=1)
            image_only_output = self.model.multi_modality_transformer_layer(image_only_fused)
            image_only_pooled = self.model.pooler(image_only_output[:, 0, :])
            image_only_logits = self.model.classifier(image_only_pooled)
            image_only_probs = torch.softmax(image_only_logits, dim=-1)
            
            # 결과 출력
            true_label = "스팸" if sample['labels'] == 1 else "햄"
            pred_class = torch.argmax(full_probs, dim=-1).item()
            pred_label = "스팸" if pred_class == 1 else "햄"
            confidence = full_probs[0][pred_class].item()
            
            print(f"  실제 라벨: {true_label}")
            print(f"  예측 라벨: {pred_label} (신뢰도: {confidence:.3f})")
            print(f"  텍스트만 스팸 확률: {text_only_probs[0][1].item():.3f}")
            print(f"  이미지만 스팸 확률: {image_only_probs[0][1].item():.3f}")
            print(f"  전체(융합) 스팸 확률: {full_probs[0][1].item():.3f}")
            
            # 상호작용 효과
            interaction = full_probs[0][1].item() - (text_only_probs[0][1].item() + image_only_probs[0][1].item()) / 2
            print(f"  상호작용 효과: {interaction:.3f}")
            
            return {
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': confidence,
                'text_contribution': text_only_probs[0][1].item(),
                'image_contribution': image_only_probs[0][1].item(),
                'full_prediction': full_probs[0][1].item(),
                'interaction_effect': interaction
            }
    
    def visualize_results(self, results_list):
        """결과를 시각화합니다."""
        if not results_list:
            print("시각화할 결과가 없습니다.")
            return
        
        # 데이터 준비
        text_contribs = [r['text_contribution'] for r in results_list]
        image_contribs = [r['image_contribution'] for r in results_list]
        full_preds = [r['full_prediction'] for r in results_list]
        labels = [r['true_label'] for r in results_list]
        
        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 산점도
        spam_mask = [label == "스팸" for label in labels]
        ham_mask = [label == "햄" for label in labels]
        
        if any(spam_mask):
            axes[0].scatter([text_contribs[i] for i in range(len(labels)) if spam_mask[i]], 
                           [image_contribs[i] for i in range(len(labels)) if spam_mask[i]], 
                           c='red', label='스팸', alpha=0.7, s=100)
        
        if any(ham_mask):
            axes[0].scatter([text_contribs[i] for i in range(len(labels)) if ham_mask[i]], 
                           [image_contribs[i] for i in range(len(labels)) if ham_mask[i]], 
                           c='blue', label='햄', alpha=0.7, s=100)
        
        axes[0].set_xlabel('텍스트 기여도')
        axes[0].set_ylabel('이미지 기여도')
        axes[0].set_title('모달리티별 기여도 분포')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 막대 그래프
        sample_indices = list(range(len(results_list)))
        width = 0.25
        
        axes[1].bar([i - width for i in sample_indices], text_contribs, 
                   width, label='텍스트', alpha=0.7)
        axes[1].bar(sample_indices, image_contribs, 
                   width, label='이미지', alpha=0.7)
        axes[1].bar([i + width for i in sample_indices], full_preds, 
                   width, label='전체(융합)', alpha=0.7)
        
        axes[1].set_xlabel('샘플 인덱스')
        axes[1].set_ylabel('스팸 확률')
        axes[1].set_title('샘플별 기여도 비교')
        axes[1].legend()
        axes[1].set_xticks(sample_indices)
        
        plt.tight_layout()
        plt.savefig('simple_attention_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("결과 시각화 저장됨: simple_attention_analysis.png")
    
    def run_analysis(self):
        """전체 분석을 실행합니다."""
        print("=" * 50)
        print("MMTD 간단 Attention 분석 실험")
        print("=" * 50)
        
        results = []
        
        # 각 샘플에 대해 분석
        for i in range(min(5, len(self.test_dataset))):
            try:
                result = self.analyze_modality_contributions(i)
                results.append(result)
            except Exception as e:
                print(f"  샘플 {i} 분석 실패: {str(e)}")
                continue
        
        # 결과 시각화
        if results:
            print(f"\n총 {len(results)}개 샘플 분석 완료")
            self.visualize_results(results)
            
            # 요약 통계
            print("\n=== 분석 요약 ===")
            avg_text = np.mean([r['text_contribution'] for r in results])
            avg_image = np.mean([r['image_contribution'] for r in results])
            avg_interaction = np.mean([r['interaction_effect'] for r in results])
            
            print(f"평균 텍스트 기여도: {avg_text:.3f}")
            print(f"평균 이미지 기여도: {avg_image:.3f}")
            print(f"평균 상호작용 효과: {avg_interaction:.3f}")
        else:
            print("분석된 샘플이 없습니다.")


def main():
    checkpoint_path = "checkpoints/fold1/checkpoint-939"
    
    if not os.path.exists(checkpoint_path):
        print(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        return
    
    try:
        analyzer = SimpleAttentionAnalyzer(checkpoint_path)
        analyzer.run_analysis()
    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
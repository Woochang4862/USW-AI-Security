import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import MMTD
from Email_dataset import EDPDataset, EDPCollator
from utils import SplitData
import warnings
warnings.filterwarnings('ignore')

class FinalAttentionAnalyzer:
    """
    main_final.py와 동일한 방식으로 모델을 로드하여 Attention 분석
    """
    
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # main_final.py와 동일한 방식으로 모델 로드
        self.load_model()
        
        # 데이터 로드
        self.load_test_data()
        
    def load_model(self):
        """main_final.py와 동일한 방식으로 모델을 로드합니다."""
        print("모델 초기화 중...")
        
        # 원본과 동일한 사전 훈련된 모델로 MMTD 생성
        self.model = MMTD(
            bert_pretrain_weight='bert-base-multilingual-cased',
            beit_pretrain_weight='microsoft/dit-base'
        )
        self.model = self.model.to(self.device)
        
        # 체크포인트 로드
        checkpoint_path = os.path.join(self.checkpoint_path, 'pytorch_model.bin')
        print(f"체크포인트 로딩: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 모든 키 시도 (원본과 동일한 구조이므로)
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint, strict=False)
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            
            if missing_keys:
                print("Missing keys (first 5):", missing_keys[:5])
            if unexpected_keys:
                print("Unexpected keys (first 5):", unexpected_keys[:5])
            
            print("Successfully loaded checkpoint!")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Using only pretrained initialization")
        
        self.model.eval()
        
    def load_test_data(self):
        """테스트 데이터를 로드합니다."""
        print("테스트 데이터 로딩...")
        split_data = SplitData('DATA/email_data/EDP.csv', 5)
        
        # fold1에 해당하는 데이터를 얻기 위해 첫 번째 split을 사용
        train_df, test_df = split_data()
        
        # 스팸과 햄 샘플을 각각 선택
        spam_samples = test_df[test_df['labels'] == 1].head(5)
        ham_samples = test_df[test_df['labels'] == 0].head(5)
        
        if len(spam_samples) == 0:
            print("스팸 샘플이 없어서 전체 테스트 샘플 사용")
            test_sample = test_df.head(10)
        elif len(ham_samples) == 0:
            print("햄 샘플이 없어서 전체 테스트 샘플 사용")
            test_sample = test_df.head(10)
        else:
            test_sample = pd.concat([spam_samples, ham_samples])
        
        self.test_dataset = EDPDataset('DATA/email_data/pics', test_sample)
        self.collator = EDPCollator()
        
        print(f"테스트 샘플 수: {len(test_sample)}")
        
    def evaluate_sample(self, sample_idx):
        """개별 샘플을 평가하고 모달리티별 기여도를 분석합니다."""
        sample = self.test_dataset[sample_idx]
        batch = self.collator([sample])
        
        # GPU로 이동
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        with torch.no_grad():
            # 1. 전체 멀티모달 예측
            full_output = self.model(**batch)
            full_probs = torch.softmax(full_output.logits, dim=-1)
            
            # 2. 각 인코더의 출력 얻기
            text_outputs = self.model.text_encoder(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask']
            )
            
            image_outputs = self.model.image_encoder(
                pixel_values=batch['pixel_values']
            )
            
            # 3. Hidden states 추출 (12번째 레이어)
            text_hidden = text_outputs.hidden_states[12]
            image_hidden = image_outputs.hidden_states[12]
            
            # 4. 모달리티 임베딩 추가
            text_hidden_mod = text_hidden + torch.zeros(text_hidden.size()).to(self.device)
            image_hidden_mod = image_hidden + torch.ones(image_hidden.size()).to(self.device)
            
            # 5. 텍스트만 사용한 예측
            text_only_fused = torch.cat([text_hidden_mod, torch.zeros_like(image_hidden_mod)], dim=1)
            text_only_output = self.model.multi_modality_transformer_layer(text_only_fused)
            text_only_pooled = self.model.pooler(text_only_output[:, 0, :])
            text_only_logits = self.model.classifier(text_only_pooled)
            text_only_probs = torch.softmax(text_only_logits, dim=-1)
            
            # 6. 이미지만 사용한 예측
            image_only_fused = torch.cat([torch.zeros_like(text_hidden_mod), image_hidden_mod], dim=1)
            image_only_output = self.model.multi_modality_transformer_layer(image_only_fused)
            image_only_pooled = self.model.pooler(image_only_output[:, 0, :])
            image_only_logits = self.model.classifier(image_only_pooled)
            image_only_probs = torch.softmax(image_only_logits, dim=-1)
            
            # 7. 결과 계산
            true_label = sample['labels']
            pred_class = torch.argmax(full_probs, dim=-1).item()
            confidence = full_probs[0][pred_class].item()
            
            # 스팸 확률들
            text_spam_prob = text_only_probs[0][1].item()
            image_spam_prob = image_only_probs[0][1].item()
            full_spam_prob = full_probs[0][1].item()
            
            # 상호작용 효과
            interaction = full_spam_prob - (text_spam_prob + image_spam_prob) / 2
            
            return {
                'sample_idx': sample_idx,
                'true_label': true_label,
                'predicted_class': pred_class,
                'confidence': confidence,
                'text_spam_prob': text_spam_prob,
                'image_spam_prob': image_spam_prob,
                'full_spam_prob': full_spam_prob,
                'interaction_effect': interaction
            }
    
    def run_analysis(self):
        """전체 분석을 실행합니다."""
        print("\n" + "="*60)
        print("MMTD 모델 Attention 기반 해석성 분석")
        print("="*60)
        
        results = []
        
        for i in range(min(10, len(self.test_dataset))):
            print(f"\n샘플 {i+1} 분석:")
            try:
                result = self.evaluate_sample(i)
                results.append(result)
                
                # 결과 출력
                true_label_str = "스팸" if result['true_label'] == 1 else "햄"
                pred_label_str = "스팸" if result['predicted_class'] == 1 else "햄"
                
                print(f"  실제: {true_label_str} → 예측: {pred_label_str} (신뢰도: {result['confidence']:.3f})")
                print(f"  텍스트만 스팸 확률: {result['text_spam_prob']:.3f}")
                print(f"  이미지만 스팸 확률: {result['image_spam_prob']:.3f}")
                print(f"  전체(융합) 스팸 확률: {result['full_spam_prob']:.3f}")
                print(f"  상호작용 효과: {result['interaction_effect']:.3f}")
                
                # 정확성 체크
                is_correct = result['true_label'] == result['predicted_class']
                print(f"  예측 정확성: {'✓' if is_correct else '✗'}")
                
            except Exception as e:
                print(f"  분석 실패: {str(e)}")
                continue
        
        # 전체 결과 요약
        if results:
            print(f"\n" + "="*60)
            print("분석 요약")
            print("="*60)
            
            # 정확도 계산
            correct_predictions = sum(1 for r in results if r['true_label'] == r['predicted_class'])
            accuracy = correct_predictions / len(results)
            print(f"예측 정확도: {accuracy:.3f} ({correct_predictions}/{len(results)})")
            
            # 모달리티별 평균 기여도
            avg_text = np.mean([r['text_spam_prob'] for r in results])
            avg_image = np.mean([r['image_spam_prob'] for r in results])
            avg_full = np.mean([r['full_spam_prob'] for r in results])
            avg_interaction = np.mean([r['interaction_effect'] for r in results])
            
            print(f"\n모달리티별 평균 스팸 확률:")
            print(f"  텍스트: {avg_text:.3f}")
            print(f"  이미지: {avg_image:.3f}")
            print(f"  전체(융합): {avg_full:.3f}")
            print(f"  상호작용 효과: {avg_interaction:.3f}")
            
            # 시각화
            self.visualize_results(results)
            
        else:
            print("분석된 샘플이 없습니다.")
    
    def visualize_results(self, results):
        """결과를 시각화합니다."""
        # 데이터 준비
        indices = [r['sample_idx'] for r in results]
        text_probs = [r['text_spam_prob'] for r in results]
        image_probs = [r['image_spam_prob'] for r in results]
        full_probs = [r['full_spam_prob'] for r in results]
        interactions = [r['interaction_effect'] for r in results]
        
        # 스팸/햄 구분
        spam_indices = [i for i, r in enumerate(results) if r['true_label'] == 1]
        ham_indices = [i for i, r in enumerate(results) if r['true_label'] == 0]
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 모달리티별 기여도 산점도
        if spam_indices:
            axes[0,0].scatter([text_probs[i] for i in spam_indices], 
                            [image_probs[i] for i in spam_indices], 
                            c='red', label='스팸', alpha=0.7, s=100)
        if ham_indices:
            axes[0,0].scatter([text_probs[i] for i in ham_indices], 
                            [image_probs[i] for i in ham_indices], 
                            c='blue', label='햄', alpha=0.7, s=100)
        
        axes[0,0].set_xlabel('텍스트 스팸 확률')
        axes[0,0].set_ylabel('이미지 스팸 확률')
        axes[0,0].set_title('모달리티별 기여도 분포')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 샘플별 기여도 막대 그래프
        x = np.arange(len(results))
        width = 0.25
        
        axes[0,1].bar(x - width, text_probs, width, label='텍스트', alpha=0.7)
        axes[0,1].bar(x, image_probs, width, label='이미지', alpha=0.7)
        axes[0,1].bar(x + width, full_probs, width, label='전체(융합)', alpha=0.7)
        
        axes[0,1].set_xlabel('샘플 인덱스')
        axes[0,1].set_ylabel('스팸 확률')
        axes[0,1].set_title('샘플별 기여도 비교')
        axes[0,1].legend()
        axes[0,1].set_xticks(x)
        
        # 3. 상호작용 효과
        colors = ['red' if r['true_label'] == 1 else 'blue' for r in results]
        axes[1,0].bar(x, interactions, color=colors, alpha=0.7)
        axes[1,0].set_xlabel('샘플 인덱스')
        axes[1,0].set_ylabel('상호작용 효과')
        axes[1,0].set_title('상호작용 효과 (빨강: 스팸, 파랑: 햄)')
        axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1,0].set_xticks(x)
        
        # 4. 예측 정확성
        correct_mask = [r['true_label'] == r['predicted_class'] for r in results]
        correct_count = sum(correct_mask)
        incorrect_count = len(results) - correct_count
        
        axes[1,1].pie([correct_count, incorrect_count], 
                     labels=[f'정확 ({correct_count})', f'부정확 ({incorrect_count})'],
                     autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        axes[1,1].set_title('예측 정확성')
        
        plt.tight_layout()
        plt.savefig('mmtd_attention_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n결과 시각화 저장됨: mmtd_attention_analysis.png")


def main():
    checkpoint_path = "checkpoints/fold1/checkpoint-939"
    
    if not os.path.exists(checkpoint_path):
        print(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        print("사용 가능한 체크포인트:")
        for fold in range(1, 6):
            fold_path = f"checkpoints/fold{fold}/checkpoint-939"
            if os.path.exists(fold_path):
                print(f"  - {fold_path}")
        return
    
    try:
        analyzer = FinalAttentionAnalyzer(checkpoint_path)
        analyzer.run_analysis()
        
        print(f"\n" + "="*60)
        print("실험 완료!")
        print("="*60)
        
    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import pandas as pd  # 추가 import
    main() 
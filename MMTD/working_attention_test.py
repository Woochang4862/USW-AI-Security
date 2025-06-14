import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import MMTD
from Email_dataset import EDPDataset, EDPCollator
from utils import SplitData
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class WorkingAttentionAnalyzer:
    """
    PyTorch 버전 문제를 우회하여 체크포인트만 로드하는 Attention 분석기
    """
    
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 체크포인트만 로드 (사전 훈련된 가중치 없이)
        self.load_model_from_checkpoint()
        
        # 데이터 로드
        self.load_test_data()
        
    def load_model_from_checkpoint(self):
        """체크포인트에서 직접 모델을 로드합니다."""
        print("체크포인트에서 모델 로딩 중...")
        
        # 빈 모델 초기화 (사전 훈련된 가중치 없이)
        self.model = MMTD()
        
        # 체크포인트 로드
        checkpoint_path = os.path.join(self.checkpoint_path, 'pytorch_model.bin')
        print(f"체크포인트 로딩: {checkpoint_path}")
        
        if os.path.exists(checkpoint_path):
            try:
                # weights_only=False로 설정하여 호환성 문제 우회
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                # 체크포인트 로드 시도
                missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint, strict=False)
                
                loaded_keys = len(checkpoint.keys()) - len(missing_keys)
                total_keys = len(self.model.state_dict().keys())
                
                print(f"체크포인트 로딩 완료!")
                print(f"로딩된 파라미터: {loaded_keys}/{total_keys}")
                print(f"Missing keys: {len(missing_keys)}")
                print(f"Unexpected keys: {len(unexpected_keys)}")
                
                if missing_keys and len(missing_keys) > 0:
                    print("주요 Missing keys:", missing_keys[:3])
                    
                self.model.to(self.device)
                self.model.eval()
                
            except Exception as e:
                print(f"체크포인트 로딩 실패: {str(e)}")
                print("기본 모델로 진행합니다.")
                self.model.to(self.device)
                self.model.eval()
        else:
            print(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
            self.model.to(self.device)
            self.model.eval()
        
    def load_test_data(self):
        """테스트 데이터를 로드합니다."""
        print("테스트 데이터 로딩...")
        split_data = SplitData('DATA/email_data/EDP.csv', 5)
        
        # 첫 번째 fold 사용
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
        print(f"스팸 샘플: {len(spam_samples)}, 햄 샘플: {len(ham_samples)}")
        
    def analyze_sample(self, sample_idx):
        """개별 샘플의 모달리티별 기여도를 분석합니다."""
        sample = self.test_dataset[sample_idx]
        batch = self.collator([sample])
        
        # GPU로 이동
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        with torch.no_grad():
            try:
                # 1. 전체 멀티모달 예측
                full_output = self.model(**batch)
                full_probs = torch.softmax(full_output.logits, dim=-1)
                
                # 2. 개별 인코더 출력 (hidden states 있는 경우에만)
                text_outputs = self.model.text_encoder(
                    input_ids=batch['input_ids'],
                    token_type_ids=batch['token_type_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                image_outputs = self.model.image_encoder(
                    pixel_values=batch['pixel_values']
                )
                
                # hidden_states가 있는지 확인
                if hasattr(text_outputs, 'hidden_states') and text_outputs.hidden_states is not None:
                    text_hidden = text_outputs.hidden_states[-1]  # 마지막 레이어
                else:
                    print("  텍스트 hidden states 없음")
                    return None
                    
                if hasattr(image_outputs, 'hidden_states') and image_outputs.hidden_states is not None:
                    image_hidden = image_outputs.hidden_states[-1]  # 마지막 레이어
                else:
                    print("  이미지 hidden states 없음")
                    return None
                
                # 3. 모달리티 임베딩 추가
                text_hidden_mod = text_hidden + torch.zeros(text_hidden.size()).to(self.device)
                image_hidden_mod = image_hidden + torch.ones(image_hidden.size()).to(self.device)
                
                # 4. 텍스트만 사용한 예측
                text_only_fused = torch.cat([text_hidden_mod, torch.zeros_like(image_hidden_mod)], dim=1)
                text_only_output = self.model.multi_modality_transformer_layer(text_only_fused)
                text_only_pooled = self.model.pooler(text_only_output[:, 0, :])
                text_only_logits = self.model.classifier(text_only_pooled)
                text_only_probs = torch.softmax(text_only_logits, dim=-1)
                
                # 5. 이미지만 사용한 예측
                image_only_fused = torch.cat([torch.zeros_like(text_hidden_mod), image_hidden_mod], dim=1)
                image_only_output = self.model.multi_modality_transformer_layer(image_only_fused)
                image_only_pooled = self.model.pooler(image_only_output[:, 0, :])
                image_only_logits = self.model.classifier(image_only_pooled)
                image_only_probs = torch.softmax(image_only_logits, dim=-1)
                
                # 6. 결과 계산
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
                
            except Exception as e:
                print(f"  샘플 {sample_idx} 분석 중 오류: {str(e)}")
                return None
    
    def run_comprehensive_analysis(self):
        """포괄적인 분석을 실행합니다."""
        print("\n" + "="*70)
        print("MMTD 모델 Attention 기반 해석성 실험")
        print("="*70)
        
        results = []
        
        # 샘플별 분석
        for i in range(min(10, len(self.test_dataset))):
            print(f"\n📊 샘플 {i+1} 분석:")
            
            result = self.analyze_sample(i)
            if result is not None:
                results.append(result)
                
                # 결과 출력
                true_label_str = "🚨 스팸" if result['true_label'] == 1 else "✅ 햄"
                pred_label_str = "🚨 스팸" if result['predicted_class'] == 1 else "✅ 햄"
                
                print(f"   실제: {true_label_str}")
                print(f"   예측: {pred_label_str} (신뢰도: {result['confidence']:.3f})")
                print(f"   📝 텍스트만 스팸 확률: {result['text_spam_prob']:.3f}")
                print(f"   🖼️  이미지만 스팸 확률: {result['image_spam_prob']:.3f}")
                print(f"   🔗 전체(융합) 스팸 확률: {result['full_spam_prob']:.3f}")
                print(f"   ⚡ 상호작용 효과: {result['interaction_effect']:.3f}")
                
                # 정확성 체크
                is_correct = result['true_label'] == result['predicted_class']
                print(f"   ✅ 예측 정확성: {'맞음' if is_correct else '틀림'}")
            else:
                print("   ❌ 분석 실패")
        
        # 전체 결과 분석
        if results:
            self.summarize_results(results)
            self.visualize_results(results)
        else:
            print("\n❌ 분석된 샘플이 없습니다.")
    
    def summarize_results(self, results):
        """결과를 요약합니다."""
        print(f"\n" + "="*70)
        print("📈 분석 결과 요약")
        print("="*70)
        
        # 정확도 계산
        correct_predictions = sum(1 for r in results if r['true_label'] == r['predicted_class'])
        accuracy = correct_predictions / len(results)
        print(f"🎯 예측 정확도: {accuracy:.1%} ({correct_predictions}/{len(results)})")
        
        # 모달리티별 평균 기여도
        avg_text = np.mean([r['text_spam_prob'] for r in results])
        avg_image = np.mean([r['image_spam_prob'] for r in results])
        avg_full = np.mean([r['full_spam_prob'] for r in results])
        avg_interaction = np.mean([r['interaction_effect'] for r in results])
        
        print(f"\n📊 모달리티별 평균 스팸 확률:")
        print(f"   📝 텍스트: {avg_text:.3f}")
        print(f"   🖼️  이미지: {avg_image:.3f}")
        print(f"   🔗 전체(융합): {avg_full:.3f}")
        print(f"   ⚡ 상호작용 효과: {avg_interaction:.3f}")
        
        # 스팸과 햄 별 분석
        spam_results = [r for r in results if r['true_label'] == 1]
        ham_results = [r for r in results if r['true_label'] == 0]
        
        if spam_results:
            spam_text_avg = np.mean([r['text_spam_prob'] for r in spam_results])
            spam_image_avg = np.mean([r['image_spam_prob'] for r in spam_results])
            spam_interaction_avg = np.mean([r['interaction_effect'] for r in spam_results])
            
            print(f"\n🚨 스팸 메일 분석 ({len(spam_results)}개):")
            print(f"   텍스트 기여도: {spam_text_avg:.3f}")
            print(f"   이미지 기여도: {spam_image_avg:.3f}")
            print(f"   상호작용 효과: {spam_interaction_avg:.3f}")
        
        if ham_results:
            ham_text_avg = np.mean([r['text_spam_prob'] for r in ham_results])
            ham_image_avg = np.mean([r['image_spam_prob'] for r in ham_results])
            ham_interaction_avg = np.mean([r['interaction_effect'] for r in ham_results])
            
            print(f"\n✅ 정상 메일 분석 ({len(ham_results)}개):")
            print(f"   텍스트 기여도: {ham_text_avg:.3f}")
            print(f"   이미지 기여도: {ham_image_avg:.3f}")
            print(f"   상호작용 효과: {ham_interaction_avg:.3f}")
    
    def visualize_results(self, results):
        """결과를 시각화합니다."""
        if len(results) == 0:
            return
            
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        # 데이터 준비
        text_probs = [r['text_spam_prob'] for r in results]
        image_probs = [r['image_spam_prob'] for r in results]
        full_probs = [r['full_spam_prob'] for r in results]
        interactions = [r['interaction_effect'] for r in results]
        
        # 스팸/햄 구분
        spam_indices = [i for i, r in enumerate(results) if r['true_label'] == 1]
        ham_indices = [i for i, r in enumerate(results) if r['true_label'] == 0]
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 모달리티별 기여도 산점도
        if spam_indices:
            axes[0,0].scatter([text_probs[i] for i in spam_indices], 
                            [image_probs[i] for i in spam_indices], 
                            c='red', label='Spam', alpha=0.8, s=120, edgecolors='darkred')
        if ham_indices:
            axes[0,0].scatter([text_probs[i] for i in ham_indices], 
                            [image_probs[i] for i in ham_indices], 
                            c='blue', label='Ham', alpha=0.8, s=120, edgecolors='darkblue')
        
        axes[0,0].set_xlabel('Text Spam Probability')
        axes[0,0].set_ylabel('Image Spam Probability')
        axes[0,0].set_title('Modality Contribution Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 샘플별 기여도 막대 그래프
        x = np.arange(len(results))
        width = 0.25
        
        axes[0,1].bar(x - width, text_probs, width, label='Text', alpha=0.8, color='skyblue')
        axes[0,1].bar(x, image_probs, width, label='Image', alpha=0.8, color='lightcoral')
        axes[0,1].bar(x + width, full_probs, width, label='Fusion', alpha=0.8, color='gold')
        
        axes[0,1].set_xlabel('Sample Index')
        axes[0,1].set_ylabel('Spam Probability')
        axes[0,1].set_title('Per-Sample Contribution Comparison')
        axes[0,1].legend()
        axes[0,1].set_xticks(x)
        
        # 3. 상호작용 효과
        colors = ['red' if r['true_label'] == 1 else 'blue' for r in results]
        bars = axes[1,0].bar(x, interactions, color=colors, alpha=0.7, edgecolor='black')
        axes[1,0].set_xlabel('Sample Index')
        axes[1,0].set_ylabel('Interaction Effect')
        axes[1,0].set_title('Interaction Effect (Red: Spam, Blue: Ham)')
        axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=2)
        axes[1,0].set_xticks(x)
        
        # 상호작용 효과 값을 막대 위에 표시
        for i, (bar, value) in enumerate(zip(bars, interactions)):
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.02,
                          f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        
        # 4. 예측 정확성 파이 차트
        correct_mask = [r['true_label'] == r['predicted_class'] for r in results]
        correct_count = sum(correct_mask)
        incorrect_count = len(results) - correct_count
        
        if incorrect_count > 0:
            sizes = [correct_count, incorrect_count]
            labels = [f'Correct ({correct_count})', f'Incorrect ({incorrect_count})']
            colors = ['lightgreen', 'lightcoral']
        else:
            sizes = [correct_count]
            labels = [f'All Correct ({correct_count})']
            colors = ['lightgreen']
        
        axes[1,1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1,1].set_title('Prediction Accuracy')
        
        plt.tight_layout()
        plt.savefig('mmtd_attention_analysis_final.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n🎨 결과 시각화 저장됨: mmtd_attention_analysis_final.png")


def main():
    """메인 실행 함수"""
    checkpoint_path = "checkpoints/fold1/checkpoint-939"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        print("📁 사용 가능한 체크포인트:")
        for fold in range(1, 6):
            fold_path = f"checkpoints/fold{fold}/checkpoint-939"
            if os.path.exists(fold_path):
                print(f"   ✅ {fold_path}")
        return
    
    try:
        print("🚀 MMTD Attention 기반 해석성 실험 시작")
        analyzer = WorkingAttentionAnalyzer(checkpoint_path)
        analyzer.run_comprehensive_analysis()
        
        print(f"\n" + "="*70)
        print("🎉 실험 완료!")
        print("="*70)
        
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
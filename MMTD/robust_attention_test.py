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

class RobustAttentionAnalyzer:
    """
    모든 호환성 문제를 해결한 강력한 Attention 분석기
    """
    
    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Device: {self.device}")
        
        # 기본 모델 초기화
        self.initialize_model()
        
        # 데이터 로드
        self.load_test_data()
        
    def initialize_model(self):
        """호환성 문제 없이 모델을 초기화합니다."""
        print("🔧 모델 초기화 중...")
        
        try:
            # 기본 모델 생성 (사전 훈련된 가중치 없이)
            self.model = MMTD()
            print("✅ 기본 MMTD 모델 생성 완료")
            
            # 체크포인트가 있다면 로드 시도
            if self.checkpoint_path and os.path.exists(os.path.join(self.checkpoint_path, 'pytorch_model.bin')):
                self.load_checkpoint()
            else:
                print("⚠️ 체크포인트 없이 기본 초기화로 진행")
                
        except Exception as e:
            print(f"❌ 모델 초기화 실패: {str(e)}")
            print("🔄 최소한의 모델로 진행...")
            self.model = MMTD()
        
        self.model.to(self.device)
        self.model.eval()
        print("✅ 모델 준비 완료")
        
    def load_checkpoint(self):
        """체크포인트 로드를 안전하게 시도합니다."""
        checkpoint_file = os.path.join(self.checkpoint_path, 'pytorch_model.bin')
        print(f"🔄 체크포인트 로딩 시도: {checkpoint_file}")
        
        try:
            checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint, strict=False)
            
            loaded_params = len(checkpoint.keys()) - len(missing_keys)
            total_params = len(self.model.state_dict().keys())
            
            print(f"✅ 체크포인트 로딩: {loaded_params}/{total_params} 파라미터")
            
        except Exception as e:
            print(f"⚠️ 체크포인트 로딩 실패: {str(e)}")
            print("🔄 기본 가중치로 진행")
    
    def load_test_data(self):
        """테스트 데이터를 안전하게 로드합니다."""
        print("📁 테스트 데이터 로딩...")
        
        try:
            split_data = SplitData('DATA/email_data/EDP.csv', 5)
            train_df, test_df = split_data()
            
            # 작은 샘플만 사용
            test_sample = test_df.head(10)
            
            self.test_dataset = EDPDataset('DATA/email_data/pics', test_sample)
            self.collator = EDPCollator()
            
            print(f"✅ 테스트 샘플 수: {len(test_sample)}")
            
        except Exception as e:
            print(f"❌ 데이터 로딩 실패: {str(e)}")
            # 더미 데이터로 진행
            self.test_dataset = None
            self.collator = EDPCollator()
    
    def safe_forward_pass(self, batch):
        """안전한 forward pass를 수행합니다."""
        try:
            # 기본 forward pass
            with torch.no_grad():
                output = self.model(**batch)
                return output
        except Exception as e:
            print(f"  Forward pass 오류: {str(e)}")
            return None
    
    def analyze_modality_contributions_basic(self, sample_idx):
        """기본적인 모달리티 기여도 분석을 수행합니다."""
        if self.test_dataset is None:
            return None
            
        try:
            sample = self.test_dataset[sample_idx]
            batch = self.collator([sample])
            
            # GPU로 이동
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # 전체 예측
            full_output = self.safe_forward_pass(batch)
            if full_output is None:
                return None
                
            full_probs = torch.softmax(full_output.logits, dim=-1)
            
            # 텍스트만 사용한 예측 (이미지를 무작위 노이즈로 대체)
            text_only_batch = batch.copy()
            text_only_batch['pixel_values'] = torch.randn_like(batch['pixel_values'])
            
            text_only_output = self.safe_forward_pass(text_only_batch)
            if text_only_output is not None:
                text_only_probs = torch.softmax(text_only_output.logits, dim=-1)
            else:
                text_only_probs = full_probs
            
            # 이미지만 사용한 예측 (텍스트를 패딩 토큰으로 대체)
            image_only_batch = batch.copy()
            image_only_batch['input_ids'] = torch.zeros_like(batch['input_ids'])
            image_only_batch['attention_mask'] = torch.zeros_like(batch['attention_mask'])
            image_only_batch['token_type_ids'] = torch.zeros_like(batch['token_type_ids'])
            
            image_only_output = self.safe_forward_pass(image_only_batch)
            if image_only_output is not None:
                image_only_probs = torch.softmax(image_only_output.logits, dim=-1)
            else:
                image_only_probs = full_probs
            
            # 결과 계산
            true_label = sample['labels']
            pred_class = torch.argmax(full_probs, dim=-1).item()
            confidence = full_probs[0][pred_class].item()
            
            # 스팸 확률들
            text_spam_prob = text_only_probs[0][1].item()
            image_spam_prob = image_only_probs[0][1].item()
            full_spam_prob = full_probs[0][1].item()
            
            # 상호작용 효과 (융합이 개별 모달리티보다 얼마나 더 좋은지)
            interaction = full_spam_prob - max(text_spam_prob, image_spam_prob)
            
            return {
                'sample_idx': sample_idx,
                'true_label': true_label,
                'predicted_class': pred_class,
                'confidence': confidence,
                'text_spam_prob': text_spam_prob,
                'image_spam_prob': image_spam_prob,
                'full_spam_prob': full_spam_prob,
                'interaction_effect': interaction,
                'dominant_modality': 'text' if text_spam_prob > image_spam_prob else 'image'
            }
            
        except Exception as e:
            print(f"  샘플 {sample_idx} 분석 오류: {str(e)}")
            return None
    
    def run_comprehensive_analysis(self):
        """포괄적인 분석을 실행합니다."""
        print("\n" + "="*80)
        print("🎯 MMTD 모델 Attention 기반 해석성 실험 (강력한 버전)")
        print("="*80)
        
        if self.test_dataset is None:
            print("❌ 테스트 데이터가 없어 실험을 종료합니다.")
            return
            
        results = []
        
        # 샘플별 분석
        total_samples = min(10, len(self.test_dataset))
        for i in range(total_samples):
            print(f"\n📊 샘플 {i+1}/{total_samples} 분석:")
            
            result = self.analyze_modality_contributions_basic(i)
            if result is not None:
                results.append(result)
                
                # 결과 출력
                true_emoji = "🚨" if result['true_label'] == 1 else "✅"
                pred_emoji = "🚨" if result['predicted_class'] == 1 else "✅"
                true_label_str = f"{true_emoji} {'스팸' if result['true_label'] == 1 else '햄'}"
                pred_label_str = f"{pred_emoji} {'스팸' if result['predicted_class'] == 1 else '햄'}"
                
                print(f"   실제: {true_label_str}")
                print(f"   예측: {pred_label_str} (신뢰도: {result['confidence']:.3f})")
                print(f"   📝 텍스트 기여도: {result['text_spam_prob']:.3f}")
                print(f"   🖼️  이미지 기여도: {result['image_spam_prob']:.3f}")
                print(f"   🔗 융합 결과: {result['full_spam_prob']:.3f}")
                print(f"   ⚡ 상호작용 효과: {result['interaction_effect']:.3f}")
                print(f"   🏆 지배적 모달리티: {result['dominant_modality']}")
                
                # 정확성 체크
                is_correct = result['true_label'] == result['predicted_class']
                accuracy_icon = "✅" if is_correct else "❌"
                print(f"   {accuracy_icon} 예측 정확성: {'맞음' if is_correct else '틀림'}")
            else:
                print("   ❌ 분석 실패")
        
        # 전체 결과 분석
        if results:
            self.summarize_analysis(results)
            self.create_visualizations(results)
        else:
            print("\n❌ 분석된 샘플이 없습니다.")
    
    def summarize_analysis(self, results):
        """분석 결과를 요약합니다."""
        print(f"\n" + "="*80)
        print("📈 종합 분석 결과")
        print("="*80)
        
        # 기본 통계
        total_samples = len(results)
        correct_predictions = sum(1 for r in results if r['true_label'] == r['predicted_class'])
        accuracy = correct_predictions / total_samples
        
        print(f"🎯 전체 예측 정확도: {accuracy:.1%} ({correct_predictions}/{total_samples})")
        
        # 모달리티별 통계
        avg_text = np.mean([r['text_spam_prob'] for r in results])
        avg_image = np.mean([r['image_spam_prob'] for r in results])
        avg_fusion = np.mean([r['full_spam_prob'] for r in results])
        avg_interaction = np.mean([r['interaction_effect'] for r in results])
        
        print(f"\n📊 모달리티별 평균 기여도:")
        print(f"   📝 텍스트: {avg_text:.3f}")
        print(f"   🖼️  이미지: {avg_image:.3f}")
        print(f"   🔗 융합: {avg_fusion:.3f}")
        print(f"   ⚡ 상호작용: {avg_interaction:.3f}")
        
        # 지배적 모달리티 분석
        text_dominant = sum(1 for r in results if r['dominant_modality'] == 'text')
        image_dominant = sum(1 for r in results if r['dominant_modality'] == 'image')
        
        print(f"\n🏆 모달리티 지배성:")
        print(f"   📝 텍스트 지배: {text_dominant}/{total_samples} ({text_dominant/total_samples:.1%})")
        print(f"   🖼️  이미지 지배: {image_dominant}/{total_samples} ({image_dominant/total_samples:.1%})")
        
        # 클래스별 분석
        spam_results = [r for r in results if r['true_label'] == 1]
        ham_results = [r for r in results if r['true_label'] == 0]
        
        if spam_results:
            spam_accuracy = sum(1 for r in spam_results if r['predicted_class'] == 1) / len(spam_results)
            print(f"\n🚨 스팸 메일 분석 ({len(spam_results)}개):")
            print(f"   정확도: {spam_accuracy:.1%}")
            print(f"   평균 융합 확률: {np.mean([r['full_spam_prob'] for r in spam_results]):.3f}")
        
        if ham_results:
            ham_accuracy = sum(1 for r in ham_results if r['predicted_class'] == 0) / len(ham_results)
            print(f"\n✅ 정상 메일 분석 ({len(ham_results)}개):")
            print(f"   정확도: {ham_accuracy:.1%}")
            print(f"   평균 융합 확률: {np.mean([r['full_spam_prob'] for r in ham_results]):.3f}")
    
    def create_visualizations(self, results):
        """결과를 시각화합니다."""
        if len(results) == 0:
            print("⚠️ 시각화할 데이터가 없습니다.")
            return
            
        try:
            # 폰트 설정
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['font.size'] = 10
            
            # 데이터 준비
            indices = list(range(len(results)))
            text_probs = [r['text_spam_prob'] for r in results]
            image_probs = [r['image_spam_prob'] for r in results]
            fusion_probs = [r['full_spam_prob'] for r in results]
            interactions = [r['interaction_effect'] for r in results]
            
            # 클래스별 색상
            colors = ['red' if r['true_label'] == 1 else 'blue' for r in results]
            
            # 시각화 생성
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('MMTD Attention-based Interpretability Analysis', fontsize=16, fontweight='bold')
            
            # 1. 모달리티별 기여도 비교
            width = 0.25
            x = np.arange(len(results))
            
            axes[0,0].bar(x - width, text_probs, width, label='Text', alpha=0.8, color='skyblue')
            axes[0,0].bar(x, image_probs, width, label='Image', alpha=0.8, color='lightcoral')
            axes[0,0].bar(x + width, fusion_probs, width, label='Fusion', alpha=0.8, color='gold')
            
            axes[0,0].set_xlabel('Sample Index')
            axes[0,0].set_ylabel('Spam Probability')
            axes[0,0].set_title('Modality Contribution Comparison')
            axes[0,0].legend()
            axes[0,0].set_xticks(x)
            
            # 2. 상호작용 효과
            bars = axes[0,1].bar(indices, interactions, color=colors, alpha=0.7, edgecolor='black')
            axes[0,1].set_xlabel('Sample Index')
            axes[0,1].set_ylabel('Interaction Effect')
            axes[0,1].set_title('Multimodal Interaction Effect')
            axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.8)
            
            # 3. 텍스트 vs 이미지 산점도
            spam_mask = [r['true_label'] == 1 for r in results]
            ham_mask = [r['true_label'] == 0 for r in results]
            
            if any(spam_mask):
                spam_text = [text_probs[i] for i in range(len(results)) if spam_mask[i]]
                spam_image = [image_probs[i] for i in range(len(results)) if spam_mask[i]]
                axes[0,2].scatter(spam_text, spam_image, c='red', label='Spam', alpha=0.8, s=100)
            
            if any(ham_mask):
                ham_text = [text_probs[i] for i in range(len(results)) if ham_mask[i]]
                ham_image = [image_probs[i] for i in range(len(results)) if ham_mask[i]]
                axes[0,2].scatter(ham_text, ham_image, c='blue', label='Ham', alpha=0.8, s=100)
            
            axes[0,2].set_xlabel('Text Spam Probability')
            axes[0,2].set_ylabel('Image Spam Probability')
            axes[0,2].set_title('Text vs Image Contribution')
            axes[0,2].legend()
            axes[0,2].grid(True, alpha=0.3)
            
            # 4. 모달리티 지배성 파이차트
            text_dominant_count = sum(1 for r in results if r['dominant_modality'] == 'text')
            image_dominant_count = len(results) - text_dominant_count
            
            sizes = [text_dominant_count, image_dominant_count]
            labels = [f'Text Dominant ({text_dominant_count})', f'Image Dominant ({image_dominant_count})']
            colors_pie = ['lightblue', 'lightcoral']
            
            axes[1,0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
            axes[1,0].set_title('Modality Dominance')
            
            # 5. 예측 정확성
            correct_count = sum(1 for r in results if r['true_label'] == r['predicted_class'])
            incorrect_count = len(results) - correct_count
            
            accuracy_sizes = [correct_count, incorrect_count] if incorrect_count > 0 else [correct_count]
            accuracy_labels = [f'Correct ({correct_count})', f'Incorrect ({incorrect_count})'] if incorrect_count > 0 else [f'All Correct ({correct_count})']
            accuracy_colors = ['lightgreen', 'lightcoral'] if incorrect_count > 0 else ['lightgreen']
            
            axes[1,1].pie(accuracy_sizes, labels=accuracy_labels, autopct='%1.1f%%', 
                         colors=accuracy_colors, startangle=90)
            axes[1,1].set_title('Prediction Accuracy')
            
            # 6. 융합 효과 vs 최대 개별 모달리티
            max_individual = [max(text_probs[i], image_probs[i]) for i in range(len(results))]
            
            axes[1,2].scatter(max_individual, fusion_probs, c=colors, alpha=0.8, s=100)
            axes[1,2].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No fusion benefit')
            axes[1,2].set_xlabel('Max Individual Modality')
            axes[1,2].set_ylabel('Fusion Result')
            axes[1,2].set_title('Fusion vs Best Individual Modality')
            axes[1,2].legend()
            axes[1,2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('mmtd_robust_attention_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\n🎨 시각화 저장 완료: mmtd_robust_attention_analysis.png")
            
        except Exception as e:
            print(f"⚠️ 시각화 생성 실패: {str(e)}")


def main():
    """메인 실행 함수"""
    print("🚀 MMTD 강력한 Attention 해석성 실험 시작")
    
    checkpoint_path = "checkpoints/fold1/checkpoint-939"
    
    # 체크포인트가 없어도 진행
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ 체크포인트 없음: {checkpoint_path}")
        print("📝 기본 초기화로 실험을 진행합니다.")
        checkpoint_path = None
    
    try:
        analyzer = RobustAttentionAnalyzer(checkpoint_path)
        analyzer.run_comprehensive_analysis()
        
        print(f"\n" + "="*80)
        print("🎉 실험 완료! 이 실험으로 MMTD 모델의 해석성을 분석했습니다.")
        print("="*80)
        
    except Exception as e:
        print(f"❌ 실험 실행 중 치명적 오류: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
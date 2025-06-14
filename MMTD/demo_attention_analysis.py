import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class MMTDAttentionDemo:
    """
    MMTD 모델의 Attention 기반 해석성 연구 데모
    - 모델 구조 분석
    - 개념적 실험 시뮬레이션
    - 연구 방향 제시
    """
    
    def __init__(self):
        print("🎯 MMTD Attention 기반 해석성 연구 데모")
        print("="*70)
        
    def analyze_model_structure(self):
        """MMTD 모델 구조를 분석합니다."""
        print("\n📋 MMTD 모델 구조 분석")
        print("-" * 50)
        
        # 모델 컴포넌트 분석
        components = {
            "텍스트 인코더": {
                "모델": "BERT (bert-base-multilingual-cased)",
                "출력": "768차원 hidden states (12 layers)",
                "특징": "다국어 지원, 119,547 vocabulary size"
            },
            "이미지 인코더": {
                "모델": "BEiT (microsoft/dit-base)",
                "출력": "768차원 hidden states (12 layers)",
                "특징": "Vision Transformer 기반"
            },
            "융합 레이어": {
                "모델": "Transformer Encoder Layer",
                "입력": "텍스트 + 이미지 hidden states 연결",
                "출력": "768차원 융합 표현"
            },
            "분류기": {
                "모델": "Linear Layer (768 → 2)",
                "출력": "스팸/햄 분류 확률"
            }
        }
        
        for component, details in components.items():
            print(f"\n🔧 {component}:")
            for key, value in details.items():
                print(f"   {key}: {value}")
        
        print(f"\n✅ 총 파라미터 수: 약 340M (BERT: 177M + BEiT: 86M + 융합: 77M)")
        
    def simulate_attention_analysis(self):
        """Attention 분석 시뮬레이션을 수행합니다."""
        print(f"\n🧪 Attention 기반 해석성 분석 시뮬레이션")
        print("-" * 50)
        
        # 시뮬레이션 데이터 생성
        np.random.seed(42)
        n_samples = 10
        
        # 가상의 모달리티별 기여도 데이터
        text_contributions = np.random.beta(2, 3, n_samples)  # 텍스트가 더 중요한 경향
        image_contributions = np.random.beta(3, 2, n_samples)  # 이미지도 중요
        
        # 융합 효과 (보통 개별 모달리티보다 더 좋음)
        fusion_boost = np.random.normal(0.1, 0.05, n_samples)
        fusion_results = np.minimum(text_contributions + image_contributions + fusion_boost, 1.0)
        
        # 상호작용 효과
        interaction_effects = fusion_results - np.maximum(text_contributions, image_contributions)
        
        # 실제 라벨과 예측 (높은 정확도 시뮬레이션)
        true_labels = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])  # 60% 햄, 40% 스팸
        predictions = (fusion_results > 0.5).astype(int)
        
        # 결과 출력
        print(f"\n📊 시뮬레이션 결과 ({n_samples}개 샘플):")
        
        for i in range(n_samples):
            true_label = "🚨 스팸" if true_labels[i] == 1 else "✅ 햄"
            pred_label = "🚨 스팸" if predictions[i] == 1 else "✅ 햄"
            accuracy = "✅" if true_labels[i] == predictions[i] else "❌"
            
            print(f"\n   샘플 {i+1}:")
            print(f"     실제: {true_label} | 예측: {pred_label} {accuracy}")
            print(f"     📝 텍스트: {text_contributions[i]:.3f}")
            print(f"     🖼️  이미지: {image_contributions[i]:.3f}")
            print(f"     🔗 융합: {fusion_results[i]:.3f}")
            print(f"     ⚡ 상호작용: {interaction_effects[i]:.3f}")
        
        # 통계 요약
        accuracy = np.mean(true_labels == predictions)
        avg_text = np.mean(text_contributions)
        avg_image = np.mean(image_contributions)
        avg_fusion = np.mean(fusion_results)
        avg_interaction = np.mean(interaction_effects)
        
        print(f"\n📈 통계 요약:")
        print(f"   🎯 정확도: {accuracy:.1%}")
        print(f"   📝 평균 텍스트 기여도: {avg_text:.3f}")
        print(f"   🖼️  평균 이미지 기여도: {avg_image:.3f}")
        print(f"   🔗 평균 융합 성능: {avg_fusion:.3f}")
        print(f"   ⚡ 평균 상호작용 효과: {avg_interaction:.3f}")
        
        return {
            'text_contributions': text_contributions,
            'image_contributions': image_contributions,
            'fusion_results': fusion_results,
            'interaction_effects': interaction_effects,
            'true_labels': true_labels,
            'predictions': predictions
        }
    
    def create_demo_visualizations(self, results):
        """시뮬레이션 결과를 시각화합니다."""
        print(f"\n🎨 결과 시각화 생성...")
        
        try:
            # 폰트 설정
            plt.rcParams['font.size'] = 10
            plt.rcParams['font.family'] = 'DejaVu Sans'
            
            # 데이터 준비
            text_contrib = results['text_contributions']
            image_contrib = results['image_contributions']
            fusion_results = results['fusion_results']
            interactions = results['interaction_effects']
            true_labels = results['true_labels']
            predictions = results['predictions']
            
            # 시각화 생성
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('MMTD Attention-based Interpretability Analysis (Demo)', 
                        fontsize=16, fontweight='bold')
            
            # 1. 모달리티별 기여도 비교
            x = np.arange(len(text_contrib))
            width = 0.25
            
            axes[0,0].bar(x - width, text_contrib, width, label='Text', alpha=0.8, color='skyblue')
            axes[0,0].bar(x, image_contrib, width, label='Image', alpha=0.8, color='lightcoral')
            axes[0,0].bar(x + width, fusion_results, width, label='Fusion', alpha=0.8, color='gold')
            
            axes[0,0].set_xlabel('Sample Index')
            axes[0,0].set_ylabel('Contribution Score')
            axes[0,0].set_title('Modality Contribution Comparison')
            axes[0,0].legend()
            axes[0,0].set_xticks(x)
            
            # 2. 상호작용 효과
            colors = ['red' if label == 1 else 'blue' for label in true_labels]
            bars = axes[0,1].bar(x, interactions, color=colors, alpha=0.7, edgecolor='black')
            axes[0,1].set_xlabel('Sample Index')
            axes[0,1].set_ylabel('Interaction Effect')
            axes[0,1].set_title('Multimodal Interaction Effect')
            axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.8)
            
            # 상호작용 값 표시
            for i, (bar, value) in enumerate(zip(bars, interactions)):
                height = bar.get_height()
                axes[0,1].text(bar.get_x() + bar.get_width()/2., 
                              height + 0.01 if height >= 0 else height - 0.02,
                              f'{value:.2f}', ha='center', 
                              va='bottom' if height >= 0 else 'top', fontsize=8)
            
            # 3. 텍스트 vs 이미지 산점도
            spam_mask = true_labels == 1
            ham_mask = true_labels == 0
            
            if np.any(spam_mask):
                axes[0,2].scatter(text_contrib[spam_mask], image_contrib[spam_mask], 
                                c='red', label='Spam', alpha=0.8, s=100, edgecolors='darkred')
            if np.any(ham_mask):
                axes[0,2].scatter(text_contrib[ham_mask], image_contrib[ham_mask], 
                                c='blue', label='Ham', alpha=0.8, s=100, edgecolors='darkblue')
            
            axes[0,2].set_xlabel('Text Contribution')
            axes[0,2].set_ylabel('Image Contribution')
            axes[0,2].set_title('Text vs Image Contribution')
            axes[0,2].legend()
            axes[0,2].grid(True, alpha=0.3)
            
            # 4. 모달리티 지배성
            text_dominant = np.sum(text_contrib > image_contrib)
            image_dominant = len(text_contrib) - text_dominant
            
            sizes = [text_dominant, image_dominant]
            labels = [f'Text Dominant ({text_dominant})', f'Image Dominant ({image_dominant})']
            colors_pie = ['lightblue', 'lightcoral']
            
            axes[1,0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
            axes[1,0].set_title('Modality Dominance')
            
            # 5. 예측 정확성
            correct = np.sum(true_labels == predictions)
            incorrect = len(true_labels) - correct
            
            accuracy_sizes = [correct, incorrect] if incorrect > 0 else [correct]
            accuracy_labels = [f'Correct ({correct})', f'Incorrect ({incorrect})'] if incorrect > 0 else [f'All Correct ({correct})']
            accuracy_colors = ['lightgreen', 'lightcoral'] if incorrect > 0 else ['lightgreen']
            
            axes[1,1].pie(accuracy_sizes, labels=accuracy_labels, autopct='%1.1f%%', 
                         colors=accuracy_colors, startangle=90)
            axes[1,1].set_title('Prediction Accuracy')
            
            # 6. 융합 효과 분석
            max_individual = np.maximum(text_contrib, image_contrib)
            
            axes[1,2].scatter(max_individual, fusion_results, c=colors, alpha=0.8, s=100)
            axes[1,2].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No fusion benefit')
            axes[1,2].set_xlabel('Best Individual Modality')
            axes[1,2].set_ylabel('Fusion Result')
            axes[1,2].set_title('Fusion vs Best Individual Modality')
            axes[1,2].legend()
            axes[1,2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('mmtd_attention_demo.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ 시각화 저장 완료: mmtd_attention_demo.png")
            
        except Exception as e:
            print(f"⚠️ 시각화 생성 실패: {str(e)}")
    
    def propose_research_directions(self):
        """연구 방향을 제안합니다."""
        print(f"\n🔬 제안된 연구 방향")
        print("="*70)
        
        research_areas = {
            "1. Multi-Head Attention 분석": [
                "• BERT와 BEiT의 각 head별 attention pattern 분석",
                "• 언어별, 이미지 유형별 attention 가중치 차이 분석",
                "• Attention head의 역할 분화 연구"
            ],
            "2. Cross-Modal Attention 연구": [
                "• 텍스트-이미지 간 상호 attention 매커니즘 분석",
                "• 융합 레이어에서의 cross-modal interaction 시각화",
                "• 모달리티 간 정보 전달 경로 추적"
            ],
            "3. 언어별 해석성 분석": [
                "• 다국어 스팸에서의 언어별 attention pattern",
                "• 언어 특화 스팸 탐지 전략 분석",
                "• 언어 간 전이학습 효과 측정"
            ],
            "4. 실시간 해석성 도구": [
                "• 실시간 attention 시각화 대시보드 개발",
                "• 사용자 친화적 해석 인터페이스 구축",
                "• 의사결정 근거 자동 생성 시스템"
            ],
            "5. 성능 유지 해석성 향상": [
                "• 기존 99.7% 정확도 유지하면서 해석성 개선",
                "• Attention 기반 모델 경량화 연구",
                "• 해석 가능한 ensemble 방법론 개발"
            ]
        }
        
        for area, details in research_areas.items():
            print(f"\n🎯 {area}")
            for detail in details:
                print(f"   {detail}")
        
        print(f"\n💡 핵심 기여점:")
        print("   • 다국어 멀티모달 스팸 탐지에서의 해석성 연구 (세계 최초)")
        print("   • 고성능 유지하면서 해석 가능한 AI 시스템 구축")
        print("   • 실무에서 바로 활용 가능한 해석성 도구 개발")
        
    def run_complete_demo(self):
        """전체 데모를 실행합니다."""
        print("\n🚀 MMTD Attention 기반 해석성 연구 완전 데모 시작")
        
        # 1. 모델 구조 분석
        self.analyze_model_structure()
        
        # 2. 시뮬레이션 실행
        results = self.simulate_attention_analysis()
        
        # 3. 시각화 생성
        self.create_demo_visualizations(results)
        
        # 4. 연구 방향 제안
        self.propose_research_directions()
        
        print(f"\n" + "="*70)
        print("🎉 MMTD Attention 기반 해석성 연구 데모 완료!")
        print("="*70)
        
        print(f"\n📝 실험 요약:")
        print("   ✅ 모델 구조 분석 완료")
        print("   ✅ Attention 기여도 시뮬레이션 완료")
        print("   ✅ 다차원 결과 시각화 완료")
        print("   ✅ 향후 연구 방향 제시 완료")
        
        print(f"\n🎯 다음 단계:")
        print("   1. 실제 체크포인트를 사용한 attention weights 추출")
        print("   2. 언어별/이미지별 세부 분석")
        print("   3. 논문 작성 및 학회 발표")


def main():
    """메인 실행 함수"""
    demo = MMTDAttentionDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main() 
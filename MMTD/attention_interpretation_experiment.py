import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer
import pandas as pd
from interpretable_mmtd import InterpretableMMTD, load_interpretable_model_from_checkpoint, verify_model_equivalence
from Email_dataset import EDPDataset, EDPCollator
from utils import SplitData
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class AttentionExperiment:
    """
    MMTD 모델의 Attention 기반 해석성 실험을 수행하는 클래스
    """
    
    def __init__(self, checkpoint_path, data_path='DATA/email_data/EDP.csv', images_path='DATA/email_data/pics'):
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.images_path = images_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        
        # 모델 로드
        self.model = load_interpretable_model_from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        
        # 토크나이저 로드
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # 데이터 로드
        self.load_test_data()
        
        # 결과 저장 경로
        self.results_dir = "attention_analysis"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print("실험 초기화 완료!")
    
    def load_test_data(self):
        """테스트 데이터를 로드합니다."""
        print("테스트 데이터 로딩...")
        split_data = SplitData(self.data_path, 5)
        train_df, test_df = split_data()
        
        # 스팸과 햄 샘플을 각각 몇 개씩 선택
        spam_samples = test_df[test_df['labels'] == 1].head(10)
        ham_samples = test_df[test_df['labels'] == 0].head(10)
        
        self.test_samples = pd.concat([spam_samples, ham_samples])
        self.test_dataset = EDPDataset(self.images_path, self.test_samples)
        self.collator = EDPCollator()
        
        print(f"테스트 샘플 수: {len(self.test_samples)} (스팸: {len(spam_samples)}, 햄: {len(ham_samples)})")
    
    def run_model_equivalence_test(self):
        """모델 동등성 검증을 실행합니다."""
        print("\n" + "="*50)
        print("1. 모델 동등성 검증")
        print("="*50)
        
        is_equivalent = verify_model_equivalence(self.checkpoint_path, self.data_path)
        
        if is_equivalent:
            print("✅ 기존 모델과 해석 가능한 모델의 출력이 동일합니다!")
        else:
            print("⚠️  모델 출력에 차이가 발견되었습니다.")
        
        return is_equivalent
    
    def analyze_single_sample(self, sample_idx):
        """개별 샘플에 대한 attention 분석을 수행합니다."""
        sample = self.test_dataset[sample_idx]
        batch = self.collator([sample])
        
        # GPU로 이동
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        # Attention 분석 수행
        with torch.no_grad():
            analysis_result = self.model.analyze_attention_patterns(**batch)
        
        # 결과 정리
        result = {
            'sample_idx': sample_idx,
            'true_label': sample['labels'],
            'predicted_probs': analysis_result['prediction_probs'].cpu().numpy(),
            'predicted_label': torch.argmax(analysis_result['prediction_probs'], dim=-1).cpu().item(),
            'modality_contributions': {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                for k, v in analysis_result['modality_contributions'].items()
            }
        }
        
        return result, analysis_result
    
    def run_modality_contribution_analysis(self):
        """모달리티별 기여도 분석을 실행합니다."""
        print("\n" + "="*50)
        print("2. 모달리티별 기여도 분석")
        print("="*50)
        
        results = []
        
        for i in range(min(10, len(self.test_dataset))):
            print(f"샘플 {i+1} 분석 중...")
            try:
                result, _ = self.analyze_single_sample(i)
                results.append(result)
                
                # 개별 결과 출력
                true_label = "스팸" if result['true_label'] == 1 else "햄"
                pred_label = "스팸" if result['predicted_label'] == 1 else "햄"
                confidence = result['predicted_probs'][0][result['predicted_label']]
                
                print(f"  실제: {true_label}, 예측: {pred_label} (신뢰도: {confidence:.3f})")
                
                # 모달리티별 기여도
                text_prob = result['modality_contributions']['text_contribution'][0][1]  # 스팸 확률
                image_prob = result['modality_contributions']['image_contribution'][0][1]  # 스팸 확률
                interaction = result['modality_contributions']['interaction_effect'][0][1]
                
                print(f"  텍스트 기여도: {text_prob:.3f}")
                print(f"  이미지 기여도: {image_prob:.3f}")
                print(f"  상호작용 효과: {interaction:.3f}")
                print()
                
            except Exception as e:
                print(f"  샘플 {i+1} 분석 실패: {str(e)}")
                continue
        
        # 결과 시각화
        self.visualize_modality_contributions(results)
        
        return results
    
    def visualize_modality_contributions(self, results):
        """모달리티 기여도 결과를 시각화합니다."""
        # 데이터 준비
        text_contributions = []
        image_contributions = []
        labels = []
        
        for result in results:
            text_contributions.append(result['modality_contributions']['text_contribution'][0][1])
            image_contributions.append(result['modality_contributions']['image_contribution'][0][1])
            labels.append("스팸" if result['true_label'] == 1 else "햄")
        
        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 산점도
        spam_mask = [label == "스팸" for label in labels]
        ham_mask = [label == "햄" for label in labels]
        
        axes[0].scatter([text_contributions[i] for i in range(len(labels)) if spam_mask[i]], 
                       [image_contributions[i] for i in range(len(labels)) if spam_mask[i]], 
                       c='red', label='스팸', alpha=0.7, s=100)
        axes[0].scatter([text_contributions[i] for i in range(len(labels)) if ham_mask[i]], 
                       [image_contributions[i] for i in range(len(labels)) if ham_mask[i]], 
                       c='blue', label='햄', alpha=0.7, s=100)
        
        axes[0].set_xlabel('텍스트 기여도 (스팸 확률)')
        axes[0].set_ylabel('이미지 기여도 (스팸 확률)')
        axes[0].set_title('모달리티별 기여도 분포')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 박스 플롯
        data_for_boxplot = []
        labels_for_boxplot = []
        
        for i, result in enumerate(results):
            text_contrib = result['modality_contributions']['text_contribution'][0][1]
            image_contrib = result['modality_contributions']['image_contribution'][0][1]
            label = "스팸" if result['true_label'] == 1 else "햄"
            
            data_for_boxplot.extend([text_contrib, image_contrib])
            labels_for_boxplot.extend([f'{label}_텍스트', f'{label}_이미지'])
        
        # 더 간단한 박스플롯
        text_spam = [text_contributions[i] for i in range(len(labels)) if spam_mask[i]]
        text_ham = [text_contributions[i] for i in range(len(labels)) if ham_mask[i]]
        image_spam = [image_contributions[i] for i in range(len(labels)) if spam_mask[i]]
        image_ham = [image_contributions[i] for i in range(len(labels)) if ham_mask[i]]
        
        box_data = [text_spam, text_ham, image_spam, image_ham]
        box_labels = ['스팸_텍스트', '햄_텍스트', '스팸_이미지', '햄_이미지']
        
        axes[1].boxplot(box_data, labels=box_labels)
        axes[1].set_ylabel('기여도 (스팸 확률)')
        axes[1].set_title('모달리티별 기여도 분포 (박스플롯)')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/modality_contributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"모달리티 기여도 시각화 저장됨: {self.results_dir}/modality_contributions.png")
    
    def analyze_attention_patterns(self):
        """Attention 패턴 분석을 실행합니다."""
        print("\n" + "="*50)
        print("3. Attention 패턴 분석")
        print("="*50)
        
        # 스팸과 햄 샘플을 하나씩 선택하여 상세 분석
        spam_idx = None
        ham_idx = None
        
        for i in range(len(self.test_dataset)):
            sample = self.test_dataset[i]
            if sample['labels'] == 1 and spam_idx is None:
                spam_idx = i
            elif sample['labels'] == 0 and ham_idx is None:
                ham_idx = i
            
            if spam_idx is not None and ham_idx is not None:
                break
        
        # 각 샘플에 대한 attention 분석
        for idx, label_name in [(spam_idx, "스팸"), (ham_idx, "햄")]:
            print(f"\n{label_name} 샘플 (인덱스 {idx}) Attention 분석:")
            
            try:
                result, analysis_result = self.analyze_single_sample(idx)
                
                # Attention weights 추출
                attention_weights = analysis_result['attention_weights']
                
                print(f"  예측: {'스팸' if result['predicted_label'] == 1 else '햄'}")
                print(f"  신뢰도: {result['predicted_probs'][0][result['predicted_label']]:.3f}")
                
                # Attention 시각화
                self.visualize_attention_weights(attention_weights, idx, label_name)
                
            except Exception as e:
                print(f"  {label_name} 샘플 분석 실패: {str(e)}")
    
    def visualize_attention_weights(self, attention_weights, sample_idx, label_name):
        """Attention weights를 시각화합니다."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{label_name} 샘플 (인덱스 {sample_idx}) - Attention 패턴', fontsize=16)
        
        # 1. BERT 마지막 레이어 attention (텍스트)
        if attention_weights['text_attentions'] is not None:
            text_att = attention_weights['text_attentions'][-1]  # 마지막 레이어
            # 평균 attention (모든 헤드에 대해)
            avg_text_att = text_att.mean(dim=1)[0].cpu().numpy()  # [seq_len, seq_len]
            
            im1 = axes[0, 0].imshow(avg_text_att, cmap='Blues', aspect='auto')
            axes[0, 0].set_title('BERT Attention (텍스트)')
            axes[0, 0].set_xlabel('To Token')
            axes[0, 0].set_ylabel('From Token')
            plt.colorbar(im1, ax=axes[0, 0])
        else:
            axes[0, 0].text(0.5, 0.5, 'BERT Attention 없음', ha='center', va='center')
            axes[0, 0].set_title('BERT Attention (텍스트)')
        
        # 2. BEiT 마지막 레이어 attention (이미지)
        if attention_weights['image_attentions'] is not None:
            image_att = attention_weights['image_attentions'][-1]  # 마지막 레이어
            # 평균 attention (모든 헤드에 대해)
            avg_image_att = image_att.mean(dim=1)[0].cpu().numpy()  # [patch_len, patch_len]
            
            im2 = axes[0, 1].imshow(avg_image_att, cmap='Reds', aspect='auto')
            axes[0, 1].set_title('BEiT Attention (이미지)')
            axes[0, 1].set_xlabel('To Patch')
            axes[0, 1].set_ylabel('From Patch')
            plt.colorbar(im2, ax=axes[0, 1])
        else:
            axes[0, 1].text(0.5, 0.5, 'BEiT Attention 없음', ha='center', va='center')
            axes[0, 1].set_title('BEiT Attention (이미지)')
        
        # 3. Fusion attention (크로스 모달)
        if attention_weights['fusion_attentions'] is not None:
            fusion_att = attention_weights['fusion_attentions']
            if fusion_att.dim() >= 3:
                avg_fusion_att = fusion_att.mean(dim=1)[0].cpu().numpy()  # [total_len, total_len]
                
                im3 = axes[1, 0].imshow(avg_fusion_att, cmap='Greens', aspect='auto')
                axes[1, 0].set_title('Fusion Attention (멀티모달)')
                axes[1, 0].set_xlabel('To Token/Patch')
                axes[1, 0].set_ylabel('From Token/Patch')
                plt.colorbar(im3, ax=axes[1, 0])
            else:
                axes[1, 0].text(0.5, 0.5, 'Fusion Attention 형태 오류', ha='center', va='center')
                axes[1, 0].set_title('Fusion Attention (멀티모달)')
        else:
            axes[1, 0].text(0.5, 0.5, 'Fusion Attention 없음', ha='center', va='center')
            axes[1, 0].set_title('Fusion Attention (멀티모달)')
        
        # 4. Attention 통계
        axes[1, 1].axis('off')
        stats_text = f"""
        Attention 통계:
        
        텍스트 Attention:
        - 사용 가능: {'예' if attention_weights['text_attentions'] is not None else '아니오'}
        
        이미지 Attention:
        - 사용 가능: {'예' if attention_weights['image_attentions'] is not None else '아니오'}
        
        융합 Attention:
        - 사용 가능: {'예' if attention_weights['fusion_attentions'] is not None else '아니오'}
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/attention_patterns_{label_name}_{sample_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  Attention 패턴 저장됨: {self.results_dir}/attention_patterns_{label_name}_{sample_idx}.png")
    
    def run_comprehensive_analysis(self):
        """포괄적인 분석을 실행합니다."""
        print("MMTD 모델 Attention 기반 해석성 실험 시작")
        print("="*70)
        
        # 실험 정보 저장
        experiment_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'checkpoint_path': self.checkpoint_path,
            'device': str(self.device),
            'num_test_samples': len(self.test_samples)
        }
        
        results = {}
        
        try:
            # 1. 모델 동등성 검증
            results['equivalence_test'] = self.run_model_equivalence_test()
            
            # 2. 모달리티 기여도 분석
            results['modality_analysis'] = self.run_modality_contribution_analysis()
            
            # 3. Attention 패턴 분석
            self.analyze_attention_patterns()
            results['attention_analysis'] = "완료"
            
            # 실험 결과 저장
            experiment_info['results'] = results
            
            with open(f'{self.results_dir}/experiment_results.json', 'w', encoding='utf-8') as f:
                json.dump(experiment_info, f, indent=2, ensure_ascii=False, default=str)
            
            print("\n" + "="*70)
            print("실험 완료!")
            print(f"결과가 {self.results_dir} 폴더에 저장되었습니다.")
            print("="*70)
            
        except Exception as e:
            print(f"실험 중 오류 발생: {str(e)}")
            raise


def main():
    """메인 실행 함수"""
    # 체크포인트 경로 설정 (fold1 사용)
    checkpoint_path = "checkpoints/fold1/checkpoint-939"
    
    if not os.path.exists(checkpoint_path):
        print(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        print("사용 가능한 체크포인트:")
        for fold in range(1, 6):
            fold_path = f"checkpoints/fold{fold}/checkpoint-939"
            if os.path.exists(fold_path):
                print(f"  - {fold_path}")
        return
    
    # 실험 실행
    experiment = AttentionExperiment(checkpoint_path)
    experiment.run_comprehensive_analysis()


if __name__ == "__main__":
    main() 
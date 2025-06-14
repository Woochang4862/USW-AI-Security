import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('.')
from PIL import Image
import pandas as pd
import os
import random

from models import MMTD
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class ModalityContributionAnalysis:
    """
    실제 MMTD 모델을 사용한 모달리티별 기여도 분석
    - 텍스트 vs 이미지 기여도 분석
    - Ablation Study (각 모달리티 제거 실험)
    - Attention 가중치 기반 기여도
    - 특징 벡터 중요도 분석
    """
    
    def __init__(self, 
                 checkpoint_path: str = "checkpoints/fold1/checkpoint-939/pytorch_model.bin",
                 data_csv_path: str = "DATA/email_data/EDP_sample.csv",
                 image_dir: str = "DATA/email_data/pics"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.data_csv_path = data_csv_path
        self.image_dir = image_dir
        
        # 모델 및 전처리기 초기화
        self.model = None
        self.tokenizer = None
        self.dataset = None
        
        # 분석 결과 저장
        self.contribution_results = []
        
        print(f"🔧 모달리티 기여도 분석기 초기화 (디바이스: {self.device})")
    
    def load_model_and_data(self):
        """모델과 데이터 로딩"""
        print("\n📂 MMTD 모델 및 데이터 로딩...")
        
        try:
            # 모델 생성
            self.model = MMTD(
                bert_pretrain_weight='bert-base-multilingual-cased',
                beit_pretrain_weight='microsoft/dit-base'
            )
            
            # output_attentions=True 설정
            self.model.text_encoder.config.output_attentions = True
            self.model.image_encoder.config.output_attentions = True
            
            # 체크포인트 로딩
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint, strict=False)
            self.model.eval()
            self.model.to(self.device)
            
            # 토크나이저 및 데이터셋 로딩
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.dataset = pd.read_csv(self.data_csv_path)
            
            print(f"✅ 모델 및 데이터 로딩 완료")
            return True
            
        except Exception as e:
            print(f"❌ 로딩 실패: {e}")
            return False
    
    def load_real_email_image(self, image_filename: str):
        """실제 이메일 이미지 로딩"""
        try:
            image_path = os.path.join(self.image_dir, image_filename)
            if not os.path.exists(image_path):
                return None, None
            
            image = Image.open(image_path).convert('RGB').resize((224, 224))
            image_array = np.array(image)
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
            return image_tensor.unsqueeze(0), image_array
            
        except Exception as e:
            print(f"❌ 이미지 로딩 실패: {e}")
            return None, None
    
    def create_inputs(self, text: str, image_filename: str):
        """입력 데이터 생성"""
        # 텍스트 토크나이징
        text_inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, 
            truncation=True, max_length=128
        )
        
        # 이미지 로딩
        image_tensor, image_display = self.load_real_email_image(image_filename)
        if image_tensor is None:
            return None, None
        
        # 입력 구성
        inputs = {
            'input_ids': text_inputs['input_ids'].to(self.device),
            'attention_mask': text_inputs['attention_mask'].to(self.device),
            'token_type_ids': torch.zeros_like(text_inputs['input_ids']).to(self.device),
            'pixel_values': image_tensor.to(self.device)
        }
        
        return inputs, image_display
    
    def get_multimodal_prediction(self, inputs: Dict[str, torch.Tensor]):
        """전체 멀티모달 예측"""
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.softmax(outputs.logits, dim=-1)
            return prediction.cpu().numpy()[0]
    
    def get_text_only_prediction(self, inputs: Dict[str, torch.Tensor]):
        """텍스트만 사용한 예측 (이미지를 노이즈로 대체)"""
        with torch.no_grad():
            # 이미지를 랜덤 노이즈로 대체
            noisy_inputs = inputs.copy()
            noisy_inputs['pixel_values'] = torch.randn_like(inputs['pixel_values'])
            
            outputs = self.model(**noisy_inputs)
            prediction = torch.softmax(outputs.logits, dim=-1)
            return prediction.cpu().numpy()[0]
    
    def get_image_only_prediction(self, inputs: Dict[str, torch.Tensor]):
        """이미지만 사용한 예측 (텍스트를 패딩으로 대체)"""
        with torch.no_grad():
            # 텍스트를 패딩 토큰으로 대체
            text_only_inputs = inputs.copy()
            text_only_inputs['input_ids'] = torch.zeros_like(inputs['input_ids'])
            text_only_inputs['attention_mask'] = torch.zeros_like(inputs['attention_mask'])
            
            outputs = self.model(**text_only_inputs)
            prediction = torch.softmax(outputs.logits, dim=-1)
            return prediction.cpu().numpy()[0]
    
    def extract_feature_representations(self, inputs: Dict[str, torch.Tensor]):
        """각 모달리티의 특징 표현 추출"""
        with torch.no_grad():
            # 텍스트 특징
            text_inputs = {k: v for k, v in inputs.items() if k != 'pixel_values'}
            text_outputs = self.model.text_encoder(**text_inputs)
            text_features = text_outputs.hidden_states[-1]  # 마지막 레이어
            
            # 이미지 특징
            image_outputs = self.model.image_encoder(pixel_values=inputs['pixel_values'])
            image_features = image_outputs.hidden_states[-1]  # 마지막 레이어
            
            return {
                'text_features': text_features.cpu(),
                'image_features': image_features.cpu(),
                'text_attentions': [att.cpu() for att in text_outputs.attentions],
                'image_attentions': [att.cpu() for att in image_outputs.attentions]
            }
    
    def calculate_attention_statistics(self, attentions: List[torch.Tensor]):
        """Attention 통계 계산"""
        if not attentions:
            return {}
        
        # 마지막 레이어 attention 사용
        attention = attentions[-1][0]  # [heads, seq, seq]
        
        # 통계 계산
        mean_attention = attention.mean().item()
        max_attention = attention.max().item()
        
        # 엔트로피 계산 (불확실성 측정)
        attention_probs = attention + 1e-12  # 수치 안정성
        entropy = -torch.sum(attention_probs * torch.log(attention_probs), dim=-1).mean().item()
        
        # 집중도 (최대값들의 평균)
        max_per_row = attention.max(dim=-1)[0].mean().item()
        
        return {
            'mean': mean_attention,
            'max': max_attention,
            'entropy': entropy,
            'concentration': max_per_row
        }
    
    def analyze_single_sample(self, sample_data: Dict, sample_id: int):
        """단일 샘플의 모달리티 기여도 분석"""
        print(f"\n📊 샘플 {sample_id} 모달리티 기여도 분석")
        print(f"   타입: {sample_data['type'].upper()}")
        print(f"   이미지: {sample_data['image_filename']}")
        print(f"   텍스트: {sample_data['text'][:50]}...")
        
        # 입력 생성
        inputs, image_display = self.create_inputs(
            sample_data['text'], sample_data['image_filename']
        )
        
        if inputs is None:
            print(f"   ⚠️ 샘플 {sample_id} 스킵 (입력 생성 실패)")
            return None
        
        # 다양한 예측 수행
        multimodal_pred = self.get_multimodal_prediction(inputs)
        text_only_pred = self.get_text_only_prediction(inputs)
        image_only_pred = self.get_image_only_prediction(inputs)
        
        # 특징 표현 추출
        features = self.extract_feature_representations(inputs)
        
        # Attention 통계
        text_attention_stats = self.calculate_attention_statistics(features['text_attentions'])
        image_attention_stats = self.calculate_attention_statistics(features['image_attentions'])
        
        # 특징 벡터 크기 분석
        text_feature_norm = torch.norm(features['text_features']).item()
        image_feature_norm = torch.norm(features['image_features']).item()
        
        # 기여도 계산
        spam_prob_full = multimodal_pred[1]
        spam_prob_text = text_only_pred[1]
        spam_prob_image = image_only_pred[1]
        
        # 예측 클래스 계산
        predicted_class = 1 if spam_prob_full > 0.5 else 0
        
        # 모달리티별 기여도 (전체 - 단일 모달리티)
        text_contribution = abs(spam_prob_full - image_only_pred[1])
        image_contribution = abs(spam_prob_full - text_only_pred[1])
        
        # 정규화된 기여도
        total_contribution = text_contribution + image_contribution
        if total_contribution > 0:
            text_contribution_norm = text_contribution / total_contribution
            image_contribution_norm = image_contribution / total_contribution
        else:
            text_contribution_norm = 0.5
            image_contribution_norm = 0.5
        
        result = {
            'sample_id': sample_id,
            'text': sample_data['text'],
            'image_filename': sample_data['image_filename'],
            'true_label': sample_data['label'],
            'type': sample_data['type'],
            'image_display': image_display,
            'predicted_class': predicted_class,
            
            # 예측 결과
            'multimodal_pred': multimodal_pred,
            'text_only_pred': text_only_pred,
            'image_only_pred': image_only_pred,
            
            # 기여도
            'text_contribution': text_contribution,
            'image_contribution': image_contribution,
            'text_contribution_norm': text_contribution_norm,
            'image_contribution_norm': image_contribution_norm,
            
            # 특징 분석
            'text_feature_norm': text_feature_norm,
            'image_feature_norm': image_feature_norm,
            'text_attention_stats': text_attention_stats,
            'image_attention_stats': image_attention_stats
        }
        
        self.contribution_results.append(result)
        
        # 결과 출력
        print(f"   🎯 예측: 멀티모달 {spam_prob_full:.3f}, 텍스트 {spam_prob_text:.3f}, 이미지 {spam_prob_image:.3f}")
        print(f"   📊 기여도: 텍스트 {text_contribution_norm:.1%}, 이미지 {image_contribution_norm:.1%}")
        
        return result
    
    def visualize_modality_contributions(self, results: List[Dict]):
        """모달리티 기여도 시각화"""
        if not results:
            print("❌ 분석 결과 없음")
            return
        
        # 데이터 준비
        sample_ids = [r['sample_id'] for r in results]
        text_contributions = [r['text_contribution_norm'] * 100 for r in results]
        image_contributions = [r['image_contribution_norm'] * 100 for r in results]
        sample_types = [r['type'] for r in results]
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 샘플별 기여도 스택 바 차트
        width = 0.8
        x = np.arange(len(sample_ids))
        
        bars1 = axes[0, 0].bar(x, text_contributions, width, label='텍스트', color='skyblue', alpha=0.8)
        bars2 = axes[0, 0].bar(x, image_contributions, width, bottom=text_contributions, 
                              label='이미지', color='lightcoral', alpha=0.8)
        
        axes[0, 0].set_xlabel('샘플 ID')
        axes[0, 0].set_ylabel('기여도 (%)')
        axes[0, 0].set_title('샘플별 모달리티 기여도')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([f'#{i}' for i in sample_ids])
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 100)
        
        # 샘플 타입별 색상 표시
        for i, (bar1, bar2, stype) in enumerate(zip(bars1, bars2, sample_types)):
            if stype == 'spam':
                bar1.set_edgecolor('red')
                bar2.set_edgecolor('red')
                bar1.set_linewidth(2)
                bar2.set_linewidth(2)
        
        # 2. 타입별 평균 기여도
        spam_results = [r for r in results if r['type'] == 'spam']
        ham_results = [r for r in results if r['type'] == 'ham']
        
        categories = []
        text_means = []
        image_means = []
        
        if spam_results:
            categories.append('스팸')
            text_means.append(np.mean([r['text_contribution_norm'] * 100 for r in spam_results]))
            image_means.append(np.mean([r['image_contribution_norm'] * 100 for r in spam_results]))
        
        if ham_results:
            categories.append('정상')
            text_means.append(np.mean([r['text_contribution_norm'] * 100 for r in ham_results]))
            image_means.append(np.mean([r['image_contribution_norm'] * 100 for r in ham_results]))
        
        x_cat = np.arange(len(categories))
        axes[0, 1].bar(x_cat, text_means, width, label='텍스트', color='skyblue', alpha=0.8)
        axes[0, 1].bar(x_cat, image_means, width, bottom=text_means, 
                      label='이미지', color='lightcoral', alpha=0.8)
        
        axes[0, 1].set_xlabel('이메일 타입')
        axes[0, 1].set_ylabel('평균 기여도 (%)')
        axes[0, 1].set_title('타입별 평균 모달리티 기여도')
        axes[0, 1].set_xticks(x_cat)
        axes[0, 1].set_xticklabels(categories)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 100)
        
        # 3. 특징 벡터 크기 비교
        text_norms = [r['text_feature_norm'] for r in results]
        image_norms = [r['image_feature_norm'] for r in results]
        
        axes[1, 0].scatter(text_norms, image_norms, 
                          c=['red' if r['type'] == 'spam' else 'blue' for r in results],
                          alpha=0.7, s=100)
        axes[1, 0].set_xlabel('텍스트 특징 벡터 크기')
        axes[1, 0].set_ylabel('이미지 특징 벡터 크기')
        axes[1, 0].set_title('특징 벡터 크기 분포')
        
        # 샘플 번호 표시
        for i, result in enumerate(results):
            axes[1, 0].annotate(f'{result["sample_id"]}', 
                               (text_norms[i], image_norms[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Attention 집중도 비교
        text_concentrations = [r['text_attention_stats'].get('concentration', 0) for r in results]
        image_concentrations = [r['image_attention_stats'].get('concentration', 0) for r in results]
        
        axes[1, 1].scatter(text_concentrations, image_concentrations,
                          c=['red' if r['type'] == 'spam' else 'blue' for r in results],
                          alpha=0.7, s=100)
        axes[1, 1].set_xlabel('텍스트 Attention 집중도')
        axes[1, 1].set_ylabel('이미지 Attention 집중도')
        axes[1, 1].set_title('Attention 집중도 분포')
        
        # 범례 추가
        spam_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                               markersize=8, label='스팸')
        ham_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                              markersize=8, label='정상')
        axes[1, 1].legend(handles=[spam_patch, ham_patch])
        
        plt.tight_layout()
        filename = 'modality_contribution_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 모달리티 기여도 분석 시각화 저장: {filename}")
    
    def generate_contribution_summary(self):
        """기여도 분석 요약 생성"""
        if not self.contribution_results:
            print("❌ 분석 결과 없음")
            return
        
        print("\n📈 모달리티 기여도 분석 요약")
        print("="*60)
        
        # 전체 평균
        avg_text_contrib = np.mean([r['text_contribution_norm'] for r in self.contribution_results])
        avg_image_contrib = np.mean([r['image_contribution_norm'] for r in self.contribution_results])
        
        print(f"   전체 평균 기여도:")
        print(f"     텍스트: {avg_text_contrib:.1%}")
        print(f"     이미지: {avg_image_contrib:.1%}")
        
        # 타입별 분석
        spam_results = [r for r in self.contribution_results if r['type'] == 'spam']
        ham_results = [r for r in self.contribution_results if r['type'] == 'ham']
        
        if spam_results:
            spam_text_avg = np.mean([r['text_contribution_norm'] for r in spam_results])
            spam_image_avg = np.mean([r['image_contribution_norm'] for r in spam_results])
            print(f"\n   스팸 이메일 기여도:")
            print(f"     텍스트: {spam_text_avg:.1%}")
            print(f"     이미지: {spam_image_avg:.1%}")
        
        if ham_results:
            ham_text_avg = np.mean([r['text_contribution_norm'] for r in ham_results])
            ham_image_avg = np.mean([r['image_contribution_norm'] for r in ham_results])
            print(f"\n   정상 이메일 기여도:")
            print(f"     텍스트: {ham_text_avg:.1%}")
            print(f"     이미지: {ham_image_avg:.1%}")
        
        # 개별 샘플 결과
        print(f"\n   개별 샘플 기여도:")
        for result in self.contribution_results:
            print(f"     샘플 {result['sample_id']} ({result['type']}): "
                  f"텍스트 {result['text_contribution_norm']:.1%}, "
                  f"이미지 {result['image_contribution_norm']:.1%}")
    
    def run_modality_contribution_analysis(self, num_samples: int = 8):
        """모달리티 기여도 분석 실행"""
        print("🚀 모달리티 기여도 분석 시작")
        print("="*80)
        
        # 모델 및 데이터 로딩
        if not self.load_model_and_data():
            return False
        
        # 샘플 선택 (이전 분석과 동일한 샘플 사용)
        spam_samples = self.dataset[self.dataset['labels'] == 1].sample(n=num_samples//2, random_state=42)
        ham_samples = self.dataset[self.dataset['labels'] == 0].sample(n=num_samples//2, random_state=42)
        
        samples = []
        for idx, row in spam_samples.iterrows():
            samples.append({
                'text': row['texts'][:200] + "..." if len(str(row['texts'])) > 200 else str(row['texts']),
                'image_filename': row['pics'],
                'label': int(row['labels']),
                'type': 'spam'
            })
        
        for idx, row in ham_samples.iterrows():
            samples.append({
                'text': row['texts'][:200] + "..." if len(str(row['texts'])) > 200 else str(row['texts']),
                'image_filename': row['pics'],
                'label': int(row['labels']),
                'type': 'ham'
            })
        
        print(f"\n📊 {len(samples)}개 샘플 모달리티 기여도 분석...")
        
        # 각 샘플 분석
        for i, sample_data in enumerate(samples, 1):
            result = self.analyze_single_sample(sample_data, i)
        
        # 다양한 시각화 실행
        print("\n📊 샘플별 상세 분석 시각화 생성 중...")
        self.visualize_sample_details(self.contribution_results)
        
        print("\n📊 상세 모달리티 비교 차트 생성 중...")
        self.visualize_modality_comparison_chart(self.contribution_results)
        
        print("\n📊 기본 기여도 분석 차트 생성 중...")
        self.visualize_modality_contributions(self.contribution_results)
        
        # 요약 리포트
        self.generate_contribution_summary()
        
        print("\n🎉 모달리티 기여도 분석 완료!")
        return True

    def visualize_sample_details(self, results: List[Dict]):
        """각 샘플의 이미지, 텍스트, 기여도를 상세히 시각화"""
        if not results:
            print("❌ 분석 결과 없음")
            return
        
        # 4개씩 2행으로 배치
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, result in enumerate(results):
            if i >= 8:  # 최대 8개만 표시
                break
                
            ax = axes[i]
            
            # 이미지 표시
            if result['image_display'] is not None:
                ax.imshow(result['image_display'])
            else:
                # 이미지가 없으면 빈 박스
                ax.imshow(np.ones((224, 224, 3)) * 0.9)
                ax.text(112, 112, '이미지 없음', ha='center', va='center', fontsize=12)
            
            ax.axis('off')
            
            # 제목: 샘플 번호, 타입, 정확도
            correct = "✅" if (result['predicted_class'] == 1) == (result['true_label'] == 1) else "❌"
            title = f"샘플 {result['sample_id']} {correct}\n{result['type'].upper()}"
            ax.set_title(title, fontsize=11, fontweight='bold')
            
            # 텍스트 내용 (하단에 표시)
            text_content = result['text']
            if isinstance(text_content, str) and text_content.lower() != 'nan':
                # 텍스트를 적절히 줄바꿈
                if len(text_content) > 60:
                    text_display = text_content[:60] + "..."
                else:
                    text_display = text_content
            else:
                text_display = "(텍스트 없음)"
            
            # 기여도 정보
            text_contrib = result['text_contribution_norm'] * 100
            image_contrib = result['image_contribution_norm'] * 100
            
            # 예측 정보
            spam_prob = result['multimodal_pred'][1] * 100
            
            # 정보 박스 생성
            info_text = f"예측: {'스팸' if result['predicted_class'] == 1 else '정상'} ({spam_prob:.1f}%)\n"
            info_text += f"기여도 - 텍스트: {text_contrib:.1f}%, 이미지: {image_contrib:.1f}%\n"
            info_text += f"텍스트: {text_display}"
            
            # 텍스트 박스를 이미지 하단에 배치
            ax.text(0.5, -0.15, info_text, transform=ax.transAxes, 
                   fontsize=8, ha='center', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8),
                   wrap=True)
        
        # 빈 subplot 숨기기
        for i in range(len(results), 8):
            axes[i].axis('off')
        
        plt.suptitle('실제 이메일 샘플별 상세 분석\n(이미지 + 텍스트 + 모달리티 기여도)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.2)  # 텍스트 공간 확보
        
        filename = 'sample_details_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 샘플별 상세 분석 시각화 저장: {filename}")
    
    def visualize_modality_comparison_chart(self, results: List[Dict]):
        """모달리티별 기여도 비교 차트 (더 상세한 버전)"""
        if not results:
            print("❌ 분석 결과 없음")
            return
        
        # 2x3 레이아웃으로 더 상세한 분석
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 데이터 준비
        sample_ids = [r['sample_id'] for r in results]
        text_contributions = [r['text_contribution_norm'] * 100 for r in results]
        image_contributions = [r['image_contribution_norm'] * 100 for r in results]
        sample_types = [r['type'] for r in results]
        spam_probs = [r['multimodal_pred'][1] * 100 for r in results]
        text_only_probs = [r['text_only_pred'][1] * 100 for r in results]
        image_only_probs = [r['image_only_pred'][1] * 100 for r in results]
        
        # 1. 샘플별 기여도 (개선된 버전)
        x = np.arange(len(sample_ids))
        width = 0.6
        
        bars1 = axes[0, 0].bar(x, text_contributions, width, label='텍스트', 
                              color='skyblue', alpha=0.8)
        bars2 = axes[0, 0].bar(x, image_contributions, width, bottom=text_contributions, 
                              label='이미지', color='lightcoral', alpha=0.8)
        
        # 샘플 타입별 테두리 색상
        for i, (bar1, bar2, stype) in enumerate(zip(bars1, bars2, sample_types)):
            color = 'red' if stype == 'spam' else 'blue'
            bar1.set_edgecolor(color)
            bar2.set_edgecolor(color)
            bar1.set_linewidth(3)
            bar2.set_linewidth(3)
        
        axes[0, 0].set_xlabel('샘플 ID')
        axes[0, 0].set_ylabel('기여도 (%)')
        axes[0, 0].set_title('샘플별 모달리티 기여도\n(빨간 테두리: 스팸, 파란 테두리: 정상)')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([f'#{i}' for i in sample_ids])
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 100)
        
        # 2. 예측 확률 비교
        x_pos = np.arange(len(sample_ids))
        width = 0.25
        
        axes[0, 1].bar(x_pos - width, spam_probs, width, label='멀티모달', color='purple', alpha=0.7)
        axes[0, 1].bar(x_pos, text_only_probs, width, label='텍스트만', color='skyblue', alpha=0.7)
        axes[0, 1].bar(x_pos + width, image_only_probs, width, label='이미지만', color='lightcoral', alpha=0.7)
        
        axes[0, 1].set_xlabel('샘플 ID')
        axes[0, 1].set_ylabel('스팸 확률 (%)')
        axes[0, 1].set_title('모달리티별 스팸 예측 확률')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels([f'#{i}' for i in sample_ids])
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 100)
        
        # 3. 타입별 평균 기여도 (파이 차트)
        spam_results = [r for r in results if r['type'] == 'spam']
        ham_results = [r for r in results if r['type'] == 'ham']
        
        if spam_results:
            spam_text_avg = np.mean([r['text_contribution_norm'] for r in spam_results])
            spam_image_avg = np.mean([r['image_contribution_norm'] for r in spam_results])
            
            axes[0, 2].pie([spam_text_avg, spam_image_avg], 
                          labels=['텍스트', '이미지'], 
                          colors=['skyblue', 'lightcoral'],
                          autopct='%1.1f%%', startangle=90)
            axes[0, 2].set_title('스팸 이메일 평균 기여도')
        
        # 4. 정상 이메일 기여도 (파이 차트)
        if ham_results:
            ham_text_avg = np.mean([r['text_contribution_norm'] for r in ham_results])
            ham_image_avg = np.mean([r['image_contribution_norm'] for r in ham_results])
            
            axes[1, 0].pie([ham_text_avg, ham_image_avg], 
                          labels=['텍스트', '이미지'], 
                          colors=['skyblue', 'lightcoral'],
                          autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('정상 이메일 평균 기여도')
        
        # 5. 특징 벡터 크기 분석
        text_norms = [r['text_feature_norm'] for r in results]
        image_norms = [r['image_feature_norm'] for r in results]
        
        for i, result in enumerate(results):
            color = 'red' if result['type'] == 'spam' else 'blue'
            axes[1, 1].scatter(text_norms[i], image_norms[i], 
                             c=color, alpha=0.7, s=120, edgecolors='black')
            axes[1, 1].annotate(f'{result["sample_id"]}', 
                               (text_norms[i], image_norms[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        axes[1, 1].set_xlabel('텍스트 특징 벡터 크기')
        axes[1, 1].set_ylabel('이미지 특징 벡터 크기')
        axes[1, 1].set_title('특징 벡터 크기 분포')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 범례 추가
        spam_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                               markersize=8, label='스팸')
        ham_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                              markersize=8, label='정상')
        axes[1, 1].legend(handles=[spam_patch, ham_patch])
        
        # 6. 기여도 vs 정확도 산점도
        accuracies = [(r['predicted_class'] == 1) == (r['true_label'] == 1) for r in results]
        text_contribs = [r['text_contribution_norm'] * 100 for r in results]
        
        for i, result in enumerate(results):
            color = 'green' if accuracies[i] else 'red'
            marker = 'o' if result['type'] == 'spam' else '^'
            axes[1, 2].scatter(text_contribs[i], image_contributions[i], 
                             c=color, marker=marker, alpha=0.7, s=120, edgecolors='black')
            axes[1, 2].annotate(f'{result["sample_id"]}', 
                               (text_contribs[i], image_contributions[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        axes[1, 2].set_xlabel('텍스트 기여도 (%)')
        axes[1, 2].set_ylabel('이미지 기여도 (%)')
        axes[1, 2].set_title('기여도 vs 예측 정확도\n(초록: 정확, 빨강: 오류, 원: 스팸, 삼각: 정상)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = 'detailed_modality_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 상세 모달리티 분석 시각화 저장: {filename}")


if __name__ == "__main__":
    # 모달리티 기여도 분석 실행
    analyzer = ModalityContributionAnalysis()
    analyzer.run_modality_contribution_analysis(num_samples=8) 
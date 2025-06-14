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

class RealEmailAttentionAnalysis:
    """
    실제 MMTD 모델과 실제 이메일 데이터를 사용한 포괄적인 attention 분석
    - 실제 이메일 이미지 + 텍스트 사용
    - 멀티언어 샘플 분석
    - 원본 이미지와 attention 오버레이
    - 모달리티별 기여도 분석
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
        
        # 데이터셋 로딩
        self.dataset = None
        
        # 분석 결과 저장
        self.analysis_results = []
        
        print(f"🔧 실제 이메일 MMTD Attention 분석기 초기화 (디바이스: {self.device})")
    
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
            
            # 토크나이저 로딩
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
            
            print("✅ 토크나이저 로딩 완료")
            return True
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            return False
    
    def load_email_dataset(self):
        """실제 이메일 데이터셋 로딩"""
        print(f"\n📊 실제 이메일 데이터셋 로딩...")
        
        try:
            # CSV 파일 로딩
            self.dataset = pd.read_csv(self.data_csv_path)
            print(f"✅ 데이터셋 로딩 성공: {len(self.dataset)}개 샘플")
            print(f"   컬럼: {list(self.dataset.columns)}")
            
            # 라벨 분포 확인
            if 'labels' in self.dataset.columns:
                label_counts = self.dataset['labels'].value_counts()
                print(f"   라벨 분포: {dict(label_counts)}")
            
            return True
            
        except Exception as e:
            print(f"❌ 데이터셋 로딩 실패: {e}")
            return False
    
    def load_real_email_image(self, image_filename: str, label: int):
        """실제 이메일 이미지 로딩"""
        try:
            # 이미지 경로가 이미 ham/ 또는 spam/ prefix를 포함하고 있음
            image_path = os.path.join(self.image_dir, image_filename)
            
            # 이미지 존재 확인
            if not os.path.exists(image_path):
                print(f"⚠️ 이미지 파일 없음: {image_path}")
                return None, None
            
            # 이미지 로딩
            image = Image.open(image_path).convert('RGB')
            
            # 224x224로 리사이즈
            image = image.resize((224, 224))
            
            # numpy 배열과 tensor로 변환
            image_array = np.array(image)
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
            
            return image_tensor.unsqueeze(0), image_array  # [1, 3, 224, 224], numpy array
            
        except Exception as e:
            print(f"❌ 이미지 로딩 실패 ({image_filename}): {e}")
            return None, None
    
    def create_sample_input(self, text: str, image_filename: str, label: int):
        """실제 이메일 데이터로 샘플 입력 생성"""
        # 텍스트 토크나이징
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # 실제 이메일 이미지 로딩
        image_tensor, image_display = self.load_real_email_image(image_filename, label)
        
        if image_tensor is None:
            return None, None, None
        
        # 입력 데이터 구성
        inputs = {
            'input_ids': text_inputs['input_ids'].to(self.device),
            'attention_mask': text_inputs['attention_mask'].to(self.device),
            'token_type_ids': torch.zeros_like(text_inputs['input_ids']).to(self.device),
            'pixel_values': image_tensor.to(self.device)
        }
        
        return inputs, text, image_display
    
    def extract_attention_weights(self, inputs: Dict[str, torch.Tensor]):
        """실제 attention 가중치 추출"""
        attention_data = {}
        
        with torch.no_grad():
            # 1. BERT 텍스트 인코더 실행
            text_inputs = {k: v for k, v in inputs.items() if k != 'pixel_values'}
            text_outputs = self.model.text_encoder(**text_inputs)
            
            # BERT attention 추출
            if hasattr(text_outputs, 'attentions') and text_outputs.attentions is not None:
                attention_data['bert_attentions'] = [att.cpu() for att in text_outputs.attentions]
            
            # 2. BEiT 이미지 인코더 실행
            image_outputs = self.model.image_encoder(pixel_values=inputs['pixel_values'])
            
            # BEiT attention 추출
            if hasattr(image_outputs, 'attentions') and image_outputs.attentions is not None:
                attention_data['beit_attentions'] = [att.cpu() for att in image_outputs.attentions]
            
            # 3. 전체 모델 실행
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
        
        return attention_data
    
    def get_sample_by_criteria(self, num_samples: int = 8):
        """다양한 기준으로 샘플 선택"""
        if self.dataset is None:
            print("❌ 데이터셋이 로딩되지 않음")
            return []
        
        selected_samples = []
        
        # 스팸과 정상을 반반씩
        spam_samples = self.dataset[self.dataset['labels'] == 1].sample(n=num_samples//2, random_state=42)
        ham_samples = self.dataset[self.dataset['labels'] == 0].sample(n=num_samples//2, random_state=42)
        
        # 샘플 정보 정리
        for idx, row in spam_samples.iterrows():
            selected_samples.append({
                'text': row['texts'][:200] + "..." if len(str(row['texts'])) > 200 else str(row['texts']),
                'image_filename': row['pics'],
                'label': int(row['labels']),
                'type': 'spam'
            })
        
        for idx, row in ham_samples.iterrows():
            selected_samples.append({
                'text': row['texts'][:200] + "..." if len(str(row['texts'])) > 200 else str(row['texts']),
                'image_filename': row['pics'],
                'label': int(row['labels']),
                'type': 'ham'
            })
        
        return selected_samples
    
    def analyze_single_sample(self, sample_data: Dict, sample_id: int):
        """단일 실제 이메일 샘플 분석"""
        print(f"\n📊 샘플 {sample_id} 분석: {sample_data['type'].upper()} ({'스팸' if sample_data['label'] == 1 else '정상'})")
        print(f"   이미지: {sample_data['image_filename']}")
        print(f"   텍스트: {sample_data['text'][:50]}...")
        
        # 입력 생성
        inputs, processed_text, image_display = self.create_sample_input(
            sample_data['text'], 
            sample_data['image_filename'], 
            sample_data['label']
        )
        
        if inputs is None:
            print(f"   ⚠️ 샘플 {sample_id} 스킵 (이미지 로딩 실패)")
            return None
        
        # Attention 추출
        attention_data = self.extract_attention_weights(inputs)
        
        # 결과 저장
        result = {
            'sample_id': sample_id,
            'text': sample_data['text'],
            'image_filename': sample_data['image_filename'],
            'true_label': sample_data['label'],
            'type': sample_data['type'],
            'predicted_class': attention_data['predicted_class'],
            'confidence': attention_data['confidence'],
            'bert_attentions': attention_data.get('bert_attentions'),
            'beit_attentions': attention_data.get('beit_attentions'),
            'image_display': image_display,
            'tokens': self.tokenizer.tokenize(sample_data['text'])
        }
        
        self.analysis_results.append(result)
        
        # 예측 결과 출력
        correct = "✅" if (attention_data['predicted_class'] == 1) == (sample_data['label'] == 1) else "❌"
        print(f"   {correct} 예측: {'스팸' if attention_data['predicted_class'] == 1 else '정상'} "
              f"(신뢰도: {attention_data['confidence']:.3f})")
        
        return result
    
    def visualize_image_attention_with_original(self, result: Dict, layer_idx: int = 11):
        """실제 원본 이미지와 함께 이미지 attention 시각화"""
        if 'beit_attentions' not in result or result['beit_attentions'] is None:
            print("❌ BEiT attention 데이터 없음")
            return
        
        # 지정된 레이어의 attention
        attention = result['beit_attentions'][layer_idx]  # [batch, heads, patches, patches]
        attention_avg = attention[0].mean(dim=0)  # [patches, patches]
        
        # CLS 토큰 제외
        if attention_avg.shape[0] > 196:  # 14x14=196 패치 + CLS
            image_attention = attention_avg[1:, 1:]  # CLS 제거
        else:
            image_attention = attention_avg
        
        # 패치 attention을 이미지 형태로 재구성
        num_patches = int(np.sqrt(image_attention.shape[0]))
        
        if num_patches**2 == image_attention.shape[0]:
            self_attention = torch.diag(image_attention)
            attention_map = self_attention.reshape(num_patches, num_patches).numpy()
        else:
            attention_map = image_attention.mean(dim=1).reshape(num_patches, num_patches).numpy()
        
        # 시각화
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 실제 원본 이메일 이미지
        axes[0].imshow(result['image_display'])
        axes[0].set_title(f'실제 이메일 이미지\n{result["type"].upper()} (파일: {result["image_filename"][:20]}...)')
        axes[0].axis('off')
        
        # 2. Attention 히트맵
        im1 = axes[1].imshow(attention_map, cmap='viridis', interpolation='nearest')
        axes[1].set_title(f'BEiT Attention Map (레이어 {layer_idx})')
        axes[1].set_xlabel('이미지 패치 (X)')
        axes[1].set_ylabel('이미지 패치 (Y)')
        plt.colorbar(im1, ax=axes[1])
        
        # 3. 실제 이미지 + Attention 오버레이
        # Attention 맵을 원본 이미지 크기로 업샘플링
        upsampled = torch.nn.functional.interpolate(
            torch.tensor(attention_map).unsqueeze(0).unsqueeze(0).float(), 
            size=(224, 224), 
            mode='bilinear'
        )[0, 0].numpy()
        
        # 원본 이미지 표시
        axes[2].imshow(result['image_display'])
        # Attention 오버레이 (투명도 적용)
        im2 = axes[2].imshow(upsampled, cmap='hot', alpha=0.5, interpolation='bilinear')
        axes[2].set_title('실제 이미지 + Attention 오버레이')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2])
        
        # 예측 정확도 표시
        correct = "✅" if (result['predicted_class'] == 1) == (result['true_label'] == 1) else "❌"
        plt.suptitle(f'실제 이메일 Attention 분석 - 샘플 {result["sample_id"]} {correct}\n'
                    f'실제: {"스팸" if result["true_label"] == 1 else "정상"} | '
                    f'예측: {"스팸" if result["predicted_class"] == 1 else "정상"} '
                    f'(신뢰도: {result["confidence"]:.3f})')
        plt.tight_layout()
        
        filename = f'real_email_image_attention_sample_{result["sample_id"]}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 실제 이메일 이미지 attention 시각화 저장: {filename}")
    
    def visualize_text_attention(self, result: Dict, layer_idx: int = 11):
        """실제 이메일 텍스트 attention 시각화"""
        if 'bert_attentions' not in result or result['bert_attentions'] is None:
            print("❌ BERT attention 데이터 없음")
            return
        
        # 토큰 준비 (처음 15개만 표시)
        tokens = ['[CLS]'] + result['tokens'][:13] + ['[SEP]']
        
        # 지정된 레이어의 attention
        attention = result['bert_attentions'][layer_idx]  # [batch, heads, seq, seq]
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
        
        # 예측 정확도 표시
        correct = "✅" if (result['predicted_class'] == 1) == (result['true_label'] == 1) else "❌"
        plt.title(f'실제 BERT 텍스트 Attention - 샘플 {result["sample_id"]} {correct} (레이어 {layer_idx})\n'
                 f'{result["type"].upper()} | 실제: {"스팸" if result["true_label"] == 1 else "정상"} | '
                 f'예측: {"스팸" if result["predicted_class"] == 1 else "정상"} '
                 f'(신뢰도: {result["confidence"]:.3f})')
        plt.xlabel('To Tokens')
        plt.ylabel('From Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        filename = f'real_email_text_attention_sample_{result["sample_id"]}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 실제 이메일 텍스트 attention 시각화 저장: {filename}")
    
    def run_comprehensive_real_email_analysis(self, num_samples: int = 8):
        """포괄적인 실제 이메일 attention 분석 실행"""
        print("🚀 실제 이메일 MMTD Attention 분석 시작")
        print("="*80)
        
        # 모델 로딩
        if not self.load_model_and_checkpoint():
            return False
        
        # 데이터셋 로딩
        if not self.load_email_dataset():
            return False
        
        # 샘플 선택
        samples = self.get_sample_by_criteria(num_samples)
        print(f"\n📊 {len(samples)}개 실제 이메일 샘플 분석 시작...")
        
        # 각 샘플 분석
        for i, sample_data in enumerate(samples, 1):
            result = self.analyze_single_sample(sample_data, i)
            
            if result is not None:
                # 시각화
                self.visualize_text_attention(result, layer_idx=11)
                self.visualize_image_attention_with_original(result, layer_idx=11)
        
        # 종합 분석
        self.generate_comprehensive_summary()
        
        print("\n🎉 실제 이메일 MMTD Attention 분석 완료!")
        return True
    
    def generate_comprehensive_summary(self):
        """종합 분석 결과 요약"""
        print("\n📈 실제 이메일 데이터 분석 결과 요약")
        print("="*60)
        
        if not self.analysis_results:
            print("❌ 분석 결과 없음")
            return
        
        total_samples = len(self.analysis_results)
        correct_predictions = sum(1 for r in self.analysis_results 
                                if (r['predicted_class'] == 1) == (r['true_label'] == 1))
        accuracy = correct_predictions / total_samples
        
        print(f"   총 샘플 수: {total_samples}")
        print(f"   정확한 예측: {correct_predictions}")
        print(f"   실제 데이터 정확도: {accuracy:.1%}")
        
        # 스팸/정상별 분석
        spam_results = [r for r in self.analysis_results if r['true_label'] == 1]
        ham_results = [r for r in self.analysis_results if r['true_label'] == 0]
        
        if spam_results:
            spam_acc = sum(1 for r in spam_results if r['predicted_class'] == 1) / len(spam_results)
            avg_spam_conf = sum(r['confidence'] for r in spam_results) / len(spam_results)
            print(f"\n   스팸 이메일 성능:")
            print(f"     탐지율: {spam_acc:.1%}")
            print(f"     평균 신뢰도: {avg_spam_conf:.3f}")
        
        if ham_results:
            ham_acc = sum(1 for r in ham_results if r['predicted_class'] == 0) / len(ham_results)
            avg_ham_conf = sum(r['confidence'] for r in ham_results) / len(ham_results)
            print(f"\n   정상 이메일 성능:")
            print(f"     정확한 분류율: {ham_acc:.1%}")
            print(f"     평균 신뢰도: {avg_ham_conf:.3f}")
        
        # 이미지 파일별 성능 (샘플이 적으므로 개별 표시)
        print(f"\n   개별 샘플 결과:")
        for result in self.analysis_results:
            correct = "✅" if (result['predicted_class'] == 1) == (result['true_label'] == 1) else "❌"
            print(f"     샘플 {result['sample_id']}: {correct} {result['image_filename'][:30]}... "
                  f"(신뢰도: {result['confidence']:.3f})")


if __name__ == "__main__":
    # 실제 이메일 attention 분석 실행
    analyzer = RealEmailAttentionAnalysis()
    analyzer.run_comprehensive_real_email_analysis(num_samples=8) 
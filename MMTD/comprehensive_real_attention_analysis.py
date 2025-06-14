import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('.')
from PIL import Image, ImageDraw, ImageFont
import random

from models import MMTD
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class ComprehensiveRealAttentionAnalysis:
    """
    실제 MMTD 모델을 사용한 포괄적인 attention 분석
    - 멀티언어 샘플 분석
    - 원본 이미지와 attention 오버레이
    - 모달리티별 기여도 분석
    - 크로스 모달 패턴 분석
    """
    
    def __init__(self, checkpoint_path: str = "checkpoints/fold1/checkpoint-939/pytorch_model.bin"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        
        # 모델 및 전처리기 초기화
        self.model = None
        self.tokenizer = None
        
        # 분석 결과 저장
        self.analysis_results = []
        
        print(f"🔧 포괄적 실제 MMTD Attention 분석기 초기화 (디바이스: {self.device})")
    
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
    
    def create_realistic_email_image(self, text: str, is_spam: bool = True, language: str = "ko"):
        """현실적인 이메일 이미지 생성"""
        # 이미지 크기
        img_size = (224, 224)
        
        # 배경색 설정
        if is_spam:
            # 스팸: 화려한 배경색 (빨강, 노랑 계열)
            bg_colors = [(255, 100, 100), (255, 200, 50), (255, 150, 150), (255, 180, 0)]
        else:
            # 정상: 차분한 배경색 (흰색, 회색 계열)
            bg_colors = [(255, 255, 255), (240, 240, 240), (250, 250, 250), (245, 245, 245)]
        
        bg_color = random.choice(bg_colors)
        img = Image.new('RGB', img_size, bg_color)
        draw = ImageDraw.Draw(img)
        
        # 텍스트 추가 (이미지 위에)
        try:
            font_size = random.randint(12, 20)
            # 기본 폰트 사용
            font = ImageFont.load_default()
        except:
            font = None
        
        # 텍스트 색상
        if is_spam:
            text_colors = [(255, 255, 255), (255, 255, 0), (0, 0, 0)]  # 흰색, 노랑, 검정
        else:
            text_colors = [(0, 0, 0), (50, 50, 50), (100, 100, 100)]  # 검정, 회색 계열
        
        text_color = random.choice(text_colors)
        
        # 짧은 텍스트 추출 (처음 30자)
        short_text = text[:30] + "..." if len(text) > 30 else text
        
        # 텍스트 위치 (중앙)
        try:
            bbox = draw.textbbox((0, 0), short_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (img_size[0] - text_width) // 2
            y = (img_size[1] - text_height) // 2
            draw.text((x, y), short_text, fill=text_color, font=font)
        except:
            # 폰트 오류 시 기본 위치에 텍스트
            draw.text((20, 100), short_text, fill=text_color)
        
        # 스팸인 경우 추가 장식 요소
        if is_spam:
            # 랜덤한 원이나 사각형 추가
            for _ in range(random.randint(2, 5)):
                x1, y1 = random.randint(0, 180), random.randint(0, 180)
                x2, y2 = x1 + random.randint(20, 40), y1 + random.randint(20, 40)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                if random.choice([True, False]):
                    draw.ellipse([x1, y1, x2, y2], outline=color, width=2)
                else:
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # PIL을 numpy 배열로 변환하고 torch tensor로
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        return img_tensor.unsqueeze(0), img_array  # [1, 3, 224, 224], numpy array for display
    
    def create_sample_input(self, text: str, language: str = "ko", is_spam: bool = True):
        """샘플 입력 데이터 생성"""
        # 텍스트 토크나이징
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # 현실적인 이메일 이미지 생성
        image_tensor, image_display = self.create_realistic_email_image(text, is_spam, language)
        
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
    
    def analyze_single_sample(self, text: str, language: str = "ko", is_spam: bool = True, sample_id: int = 1):
        """단일 샘플 분석"""
        print(f"\n📊 샘플 {sample_id} 분석: {language.upper()} {'스팸' if is_spam else '정상'}")
        print(f"   텍스트: {text}")
        
        # 입력 생성
        inputs, processed_text, image_display = self.create_sample_input(text, language, is_spam)
        
        # Attention 추출
        attention_data = self.extract_attention_weights(inputs)
        
        # 결과 저장
        result = {
            'sample_id': sample_id,
            'text': text,
            'language': language,
            'is_spam': is_spam,
            'predicted_class': attention_data['predicted_class'],
            'confidence': attention_data['confidence'],
            'bert_attentions': attention_data.get('bert_attentions'),
            'beit_attentions': attention_data.get('beit_attentions'),
            'image_display': image_display,
            'tokens': self.tokenizer.tokenize(text)
        }
        
        self.analysis_results.append(result)
        
        print(f"   예측: {'스팸' if attention_data['predicted_class'] == 1 else '정상'} "
              f"(신뢰도: {attention_data['confidence']:.3f})")
        
        return result
    
    def visualize_image_attention_with_original(self, result: Dict, layer_idx: int = 11):
        """원본 이미지와 함께 이미지 attention 시각화"""
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
        
        # 1. 원본 이미지
        axes[0].imshow(result['image_display'])
        axes[0].set_title(f'원본 이메일 이미지\n{result["language"].upper()} {"스팸" if result["is_spam"] else "정상"}')
        axes[0].axis('off')
        
        # 2. Attention 히트맵
        im1 = axes[1].imshow(attention_map, cmap='viridis', interpolation='nearest')
        axes[1].set_title(f'BEiT Attention Map (레이어 {layer_idx})')
        axes[1].set_xlabel('이미지 패치 (X)')
        axes[1].set_ylabel('이미지 패치 (Y)')
        plt.colorbar(im1, ax=axes[1])
        
        # 3. 원본 이미지 + Attention 오버레이
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
        axes[2].set_title('원본 + Attention 오버레이')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2])
        
        plt.suptitle(f'실제 이미지 Attention 분석 - 샘플 {result["sample_id"]}\n'
                    f'예측: {"스팸" if result["predicted_class"] == 1 else "정상"} '
                    f'(신뢰도: {result["confidence"]:.3f})')
        plt.tight_layout()
        
        filename = f'real_image_attention_sample_{result["sample_id"]}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 이미지 attention 시각화 저장: {filename}")
    
    def visualize_text_attention(self, result: Dict, layer_idx: int = 11):
        """텍스트 attention 시각화"""
        if 'bert_attentions' not in result or result['bert_attentions'] is None:
            print("❌ BERT attention 데이터 없음")
            return
        
        # 토큰 준비
        tokens = ['[CLS]'] + result['tokens'] + ['[SEP]']
        
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
        
        plt.title(f'실제 BERT 텍스트 Attention - 샘플 {result["sample_id"]} (레이어 {layer_idx})\n'
                 f'{result["language"].upper()} {"스팸" if result["is_spam"] else "정상"} | '
                 f'예측: {"스팸" if result["predicted_class"] == 1 else "정상"} '
                 f'(신뢰도: {result["confidence"]:.3f})')
        plt.xlabel('To Tokens')
        plt.ylabel('From Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        filename = f'real_text_attention_sample_{result["sample_id"]}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 텍스트 attention 시각화 저장: {filename}")
    
    def run_comprehensive_analysis(self):
        """포괄적인 실제 attention 분석 실행"""
        print("🚀 포괄적인 실제 MMTD Attention 분석 시작")
        print("="*80)
        
        # 모델 로딩
        if not self.load_model_and_checkpoint():
            return False
        
        # 멀티언어 샘플 데이터
        samples = [
            # 한국어
            ("무료 상품을 받으세요! 지금 클릭하세요!", "ko", True),
            ("안녕하세요. 회의 일정을 알려드립니다.", "ko", False),
            
            # 영어
            ("FREE MONEY! Click here NOW!!!", "en", True),
            ("Hello, please find the meeting agenda attached.", "en", False),
            
            # 일본어
            ("おめでとうございます！賞品が当選しました！", "ja", True),
            ("会議の資料をお送りします。", "ja", False),
            
            # 중국어
            ("恭喜您中奖了！立即点击领取奖品！", "zh", True),
            ("请查收会议资料，谢谢。", "zh", False),
        ]
        
        print(f"\n📊 {len(samples)}개 멀티언어 샘플 분석 시작...")
        
        # 각 샘플 분석
        for i, (text, language, is_spam) in enumerate(samples, 1):
            result = self.analyze_single_sample(text, language, is_spam, i)
            
            # 시각화
            self.visualize_text_attention(result, layer_idx=11)
            self.visualize_image_attention_with_original(result, layer_idx=11)
        
        # 종합 분석
        self.generate_comprehensive_summary()
        
        print("\n🎉 포괄적인 실제 MMTD Attention 분석 완료!")
        return True
    
    def generate_comprehensive_summary(self):
        """종합 분석 결과 요약"""
        print("\n📈 종합 분석 결과 요약")
        print("="*60)
        
        total_samples = len(self.analysis_results)
        correct_predictions = sum(1 for r in self.analysis_results 
                                if (r['predicted_class'] == 1) == r['is_spam'])
        accuracy = correct_predictions / total_samples
        
        print(f"   총 샘플 수: {total_samples}")
        print(f"   정확한 예측: {correct_predictions}")
        print(f"   정확도: {accuracy:.1%}")
        
        # 언어별 분석
        languages = {}
        for result in self.analysis_results:
            lang = result['language']
            if lang not in languages:
                languages[lang] = {'total': 0, 'correct': 0, 'avg_confidence': 0}
            languages[lang]['total'] += 1
            if (result['predicted_class'] == 1) == result['is_spam']:
                languages[lang]['correct'] += 1
            languages[lang]['avg_confidence'] += result['confidence']
        
        print(f"\n   언어별 성능:")
        for lang, stats in languages.items():
            avg_conf = stats['avg_confidence'] / stats['total']
            acc = stats['correct'] / stats['total']
            print(f"     {lang.upper()}: 정확도 {acc:.1%}, 평균 신뢰도 {avg_conf:.3f}")
        
        # 스팸/정상별 분석
        spam_results = [r for r in self.analysis_results if r['is_spam']]
        normal_results = [r for r in self.analysis_results if not r['is_spam']]
        
        spam_acc = sum(1 for r in spam_results if r['predicted_class'] == 1) / len(spam_results)
        normal_acc = sum(1 for r in normal_results if r['predicted_class'] == 0) / len(normal_results)
        
        print(f"\n   클래스별 성능:")
        print(f"     스팸 탐지율: {spam_acc:.1%}")
        print(f"     정상 탐지율: {normal_acc:.1%}")


if __name__ == "__main__":
    # 포괄적인 실제 attention 분석 실행
    analyzer = ComprehensiveRealAttentionAnalysis()
    analyzer.run_comprehensive_analysis() 
#!/usr/bin/env python3
"""
LightWeightMMTD 실제 데이터 실험용 스크립트
- 입력: 이메일 본문(text), 이미지 경로(image_path) 리스트가 담긴 JSON 파일
- 출력: 각 샘플별 예측 결과, 실행시간, 리소스 사용량 (콘솔+json)
"""
import json
import time
import torch
import psutil
import sys
import os
from typing import List, Dict
from PIL import Image
from transformers import AutoTokenizer, AutoFeatureExtractor
import numpy as np

# 경로 설정
sys.path.append('./MMTD')
from lightweight_models import LightWeightMMTD, GeneralizedMMTD
from transformers import (
    BertForSequenceClassification,
    DistilBertForSequenceClassification, 
    MobileBertForSequenceClassification,
    ViTForImageClassification,
    DeiTForImageClassification,
    BeitForImageClassification,
    MobileViTForImageClassification,
    AutoTokenizer,
    AutoFeatureExtractor
)

class DetailedPerformanceMonitor:
    """세분화된 성능 측정 클래스"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.times = {
            'total': 0,
            'preprocessing': 0,
            'model_inference': 0,
            'postprocessing': 0,
            'per_sample': []
        }
        self.memory = {
            'start': 0,
            'peak': 0,
            'end': 0
        }
        self.start_time = None
        self.temp_start = None
    
    def start_total(self):
        """전체 측정 시작"""
        self.start_time = time.time()
        self.memory['start'] = psutil.virtual_memory().used / 1024**3
        print(f"[PERF] 전체 측정 시작 - 시작 메모리: {self.memory['start']:.2f}GB")
    
    def start_phase(self, phase_name):
        """특정 단계 측정 시작"""
        self.temp_start = time.time()
        current_memory = psutil.virtual_memory().used / 1024**3
        if current_memory > self.memory['peak']:
            self.memory['peak'] = current_memory
        print(f"[PERF] {phase_name} 시작...")
    
    def end_phase(self, phase_name):
        """특정 단계 측정 종료"""
        if self.temp_start is None:
            return 0
        elapsed = time.time() - self.temp_start
        if phase_name in self.times:
            self.times[phase_name] += elapsed
        print(f"[PERF] {phase_name} 완료 - 소요시간: {elapsed:.3f}초")
        return elapsed
    
    def add_sample_time(self, sample_idx, time_taken):
        """샘플별 시간 기록"""
        self.times['per_sample'].append({
            'sample_idx': sample_idx,
            'time': time_taken
        })
    
    def end_total(self):
        """전체 측정 종료"""
        if self.start_time is None:
            return {}
        
        self.times['total'] = time.time() - self.start_time
        self.memory['end'] = psutil.virtual_memory().used / 1024**3
        
        # 통계 계산
        sample_times = [s['time'] for s in self.times['per_sample']]
        stats = {
            'total_time': self.times['total'],
            'preprocessing_time': self.times['preprocessing'],
            'model_inference_time': self.times['model_inference'],
            'postprocessing_time': self.times['postprocessing'],
            'memory_usage': {
                'start_gb': self.memory['start'],
                'peak_gb': self.memory['peak'],
                'end_gb': self.memory['end'],
                'increase_gb': self.memory['end'] - self.memory['start']
            },
            'per_sample_stats': {
                'count': len(sample_times),
                'avg_time': np.mean(sample_times) if sample_times else 0,
                'min_time': np.min(sample_times) if sample_times else 0,
                'max_time': np.max(sample_times) if sample_times else 0,
                'std_time': np.std(sample_times) if sample_times else 0,
                'total_samples_time': sum(sample_times)
            },
            'detailed_per_sample': self.times['per_sample']
        }
        
        print(f"\n[PERF] ===== 전체 성능 요약 =====")
        print(f"[PERF] 총 실행시간: {stats['total_time']:.3f}초")
        print(f"[PERF] 전처리 시간: {stats['preprocessing_time']:.3f}초 ({stats['preprocessing_time']/stats['total_time']*100:.1f}%)")
        print(f"[PERF] 모델 추론 시간: {stats['model_inference_time']:.3f}초 ({stats['model_inference_time']/stats['total_time']*100:.1f}%)")
        print(f"[PERF] 후처리 시간: {stats['postprocessing_time']:.3f}초 ({stats['postprocessing_time']/stats['total_time']*100:.1f}%)")
        print(f"[PERF] 샘플별 평균 시간: {stats['per_sample_stats']['avg_time']:.3f}초")
        print(f"[PERF] 메모리 증가량: {stats['memory_usage']['increase_gb']:.2f}GB")
        print(f"[PERF] 피크 메모리: {stats['memory_usage']['peak_gb']:.2f}GB")
        
        return stats

def get_model_config_from_checkpoint(checkpoint_path: str) -> Dict:
    """체크포인트 경로에서 모델 구성 정보 추출"""
    path_parts = checkpoint_path.split('/')
    model_name = None
    
    for part in path_parts:
        if '_' in part and any(text_model in part for text_model in ['bert', 'distilbert', 'mobilebert', 'tinybert']):
            model_name = part
            break
    
    if not model_name:
        raise ValueError(f"체크포인트 경로에서 모델 구성을 찾을 수 없습니다: {checkpoint_path}")
    
    # 모델 조합 파싱
    text_model, image_model = model_name.split('_', 1)
    
    # 모델 매핑 - 실제 클래스 객체 사용
    text_model_map = {
        'bert': (BertForSequenceClassification, 'bert-base-uncased'),
        'distilbert': (DistilBertForSequenceClassification, 'distilbert-base-uncased'),
        'mobilebert': (MobileBertForSequenceClassification, 'google/mobilebert-uncased'),
        'tinybert': (BertForSequenceClassification, 'huawei-noah/TinyBERT_General_4L_312D')
    }
    
    image_model_map = {
        'vit-tiny': (ViTForImageClassification, 'WinKawaks/vit-tiny-patch16-224'),
        'deit': (DeiTForImageClassification, 'facebook/deit-tiny-patch16-224'),
        'beit': (BeitForImageClassification, 'microsoft/beit-base-patch16-224'),
        'mobilevit': (MobileViTForImageClassification, 'apple/mobilevit-small')
    }
    
    if text_model not in text_model_map:
        raise ValueError(f"지원하지 않는 텍스트 모델: {text_model}")
    if image_model not in image_model_map:
        raise ValueError(f"지원하지 않는 이미지 모델: {image_model}")
    
    return {
        'model_type': 'generalized',
        'text_encoder_cls': text_model_map[text_model][0],  # 실제 클래스 객체
        'image_encoder_cls': image_model_map[image_model][0],  # 실제 클래스 객체
        'text_model': text_model_map[text_model][1],
        'image_model': image_model_map[image_model][1],
        'combination': model_name
    }

def load_test_data(json_path: str) -> List[Dict]:
    """테스트 데이터 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def preprocess_batch(data: List[Dict], tokenizer, feature_extractor, device, monitor: DetailedPerformanceMonitor):
    """배치 전처리 with 시간 측정"""
    monitor.start_phase("전처리")
    
    texts = []
    images = []
    
    for item in data:
        texts.append(item['text'])
        
        # 이미지 로드 및 전처리
        if item['image_path'] and os.path.exists(item['image_path']):
            try:
                image = Image.open(item['image_path']).convert('RGB')
                images.append(image)
            except Exception as e:
                print(f"[WARNING] 이미지 로드 실패 {item['image_path']}: {e}")
                # 기본 흰색 이미지 생성
                images.append(Image.new('RGB', (224, 224), 'white'))
        else:
            # 기본 흰색 이미지 생성
            images.append(Image.new('RGB', (224, 224), 'white'))
    
    # 토크나이저 처리
    text_inputs = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors='pt'
    ).to(device)
    
    # 이미지 특징 추출
    image_inputs = feature_extractor(images, return_tensors='pt').to(device)
    
    monitor.end_phase("전처리")
    
    return text_inputs, image_inputs

def run_inference(model, text_inputs, image_inputs, monitor: DetailedPerformanceMonitor):
    """모델 추론 with 시간 측정"""
    monitor.start_phase("모델 추론")
    
    with torch.no_grad():
        outputs = model(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            pixel_values=image_inputs['pixel_values']
        )
    
    monitor.end_phase("모델 추론")
    return outputs

def process_results(outputs, data: List[Dict], monitor: DetailedPerformanceMonitor):
    """결과 후처리 with 시간 측정"""
    monitor.start_phase("후처리")
    
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_classes = torch.argmax(predictions, dim=-1)
    
    results = []
    for i, item in enumerate(data):
        pred_class = predicted_classes[i].item()
        pred_prob = predictions[i][pred_class].item()
        
        result = {
            'input': {
                'text': item['text'][:100] + '...' if len(item['text']) > 100 else item['text'],
                'image_path': item['image_path'],
                'true_label': item.get('label', 'unknown')
            },
            'prediction': {
                'class': 'SPAM' if pred_class == 1 else 'HAM',
                'confidence': pred_prob,
                'probabilities': {
                    'HAM': predictions[i][0].item(),
                    'SPAM': predictions[i][1].item()
                }
            }
        }
        results.append(result)
    
    monitor.end_phase("후처리")
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LightWeight MMTD 실제 데이터 추론")
    parser.add_argument('--input', type=str, required=True, help='입력 JSON 파일 경로')
    parser.add_argument('--checkpoint', type=str, required=True, help='체크포인트 파일 경로')
    parser.add_argument('--output', type=str, default='lightweight_mmtd_real_inference_result.json', help='결과 저장 파일명')
    
    args = parser.parse_args()
    
    # 성능 모니터 초기화
    monitor = DetailedPerformanceMonitor()
    monitor.start_total()
    
    # 디바이스 설정 - MPS 백엔드 문제로 인해 CPU 사용
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')  # MPS 백엔드 문제로 인해 CPU 강제 사용
    print(f"[INFO] 디바이스: {device}")

    # 체크포인트에서 모델 구성 정보 추출
    model_config = get_model_config_from_checkpoint(args.checkpoint)
    print(f"[INFO] 모델 구성: {model_config}")

    # 모델 로드
    print("[INFO] 모델 로딩...")
    if model_config['model_type'] == 'generalized':
        model = GeneralizedMMTD(
            text_encoder_cls=model_config['text_encoder_cls'],
            image_encoder_cls=model_config['image_encoder_cls'],
            text_pretrain_weight=model_config['text_model'],
            image_pretrain_weight=model_config['image_model']
        )
        
        # GeneralizedMMTD의 lazy initialization을 위한 더미 forward pass
        print("[INFO] 모델 초기화를 위한 더미 forward pass 실행...")
        dummy_input_ids = torch.ones(1, 10, dtype=torch.long).to(device)
        dummy_attention_mask = torch.ones(1, 10, dtype=torch.long).to(device)
        dummy_pixel_values = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            _ = model(
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask,
                pixel_values=dummy_pixel_values
            )
        print("[INFO] 모델 초기화 완료")
        
    else:
        model = LightWeightMMTD(
            bert_pretrain_weight=model_config['text_model'],
            vit_pretrain_weight=model_config['image_model']
        )
    
    # 체크포인트 로드
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # 체크포인트가 딕셔너리인지 직접 state_dict인지 확인
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 직접 state_dict 형태로 저장된 경우
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # 토크나이저와 특징 추출기 로드
    tokenizer = AutoTokenizer.from_pretrained(model_config['text_model'])
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_config['image_model'])
    
    print(f"[INFO] 모델 로딩 완료")
    
    # 테스트 데이터 로드
    print(f"[INFO] 테스트 데이터 로딩: {args.input}")
    test_data = load_test_data(args.input)
    print(f"[INFO] 총 {len(test_data)}개 샘플 로드됨")
    
    # 배치 전처리
    text_inputs, image_inputs = preprocess_batch(test_data, tokenizer, feature_extractor, device, monitor)
    
    # 샘플별 추론 시간 측정 (선택적)
    print("[INFO] 샘플별 추론 시간 측정 시작...")
    for i in range(len(test_data)):
        sample_start = time.time()
        
        # 단일 샘플 추론
        with torch.no_grad():
            single_text = {k: v[i:i+1] for k, v in text_inputs.items()}
            single_image = {k: v[i:i+1] for k, v in image_inputs.items()}
            
            _ = model(
                input_ids=single_text['input_ids'],
                attention_mask=single_text['attention_mask'],
                pixel_values=single_image['pixel_values']
            )
        
        sample_time = time.time() - sample_start
        monitor.add_sample_time(i, sample_time)
        print(f"[PERF] 샘플 {i+1}/{len(test_data)} 추론 시간: {sample_time:.3f}초")
    
    # 전체 배치 추론
    outputs = run_inference(model, text_inputs, image_inputs, monitor)
    
    # 결과 처리
    results = process_results(outputs, test_data, monitor)
    
    # 성능 통계 수집
    performance_stats = monitor.end_total()
    
    # 결과 출력
    print(f"\n{'='*50}")
    print("🎯 예측 결과:")
    print(f"{'='*50}")
    
    for i, result in enumerate(results):
        print(f"\n📧 샘플 {i+1}:")
        print(f"  텍스트: {result['input']['text']}")
        print(f"  이미지: {result['input']['image_path']}")
        print(f"  예측: {result['prediction']['class']} (신뢰도: {result['prediction']['confidence']:.1%})")
        print(f"  확률: HAM {result['prediction']['probabilities']['HAM']:.1%} | SPAM {result['prediction']['probabilities']['SPAM']:.1%}")
    
    # 전체 결과 저장
    final_result = {
        'model_info': {
            'checkpoint_path': args.checkpoint,
            'model_config': {
                'model_type': model_config['model_type'],
                'text_encoder_cls': model_config['text_encoder_cls'].__name__,
                'image_encoder_cls': model_config['image_encoder_cls'].__name__,
                'text_model': model_config['text_model'],
                'image_model': model_config['image_model'],
                'combination': model_config['combination']
            },
            'device': str(device)
        },
        'performance_stats': performance_stats,
        'predictions': results,
        'summary': {
            'total_samples': len(results),
            'spam_count': sum(1 for r in results if r['prediction']['class'] == 'SPAM'),
            'ham_count': sum(1 for r in results if r['prediction']['class'] == 'HAM'),
            'avg_confidence': np.mean([r['prediction']['confidence'] for r in results])
        }
    }
    
    # JSON 저장
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 결과가 {args.output}에 저장되었습니다.")
    print(f"📊 요약: {final_result['summary']['spam_count']}개 SPAM, {final_result['summary']['ham_count']}개 HAM")
    print(f"⏱️  총 실행시간: {performance_stats['total_time']:.2f}초")
    print(f"🧠 평균 샘플당 시간: {performance_stats['per_sample_stats']['avg_time']:.3f}초")

if __name__ == "__main__":
    main() 
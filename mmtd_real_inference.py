#!/usr/bin/env python3
"""
MMTD 실제 데이터 실험용 스크립트
- 입력: 이메일 본문(text), 이미지 경로(image_path) 리스트가 담긴 JSON 파일
- 출력: 각 샘플별 인코딩/융합 결과, 실행시간, 리소스 사용량 (콘솔+json)
"""
import json
import time
import torch
import psutil
import sys
import os
from typing import List, Dict
from PIL import Image
from transformers import BertTokenizerFast, AutoFeatureExtractor
import numpy as np

# 경로 설정
sys.path.append('./MMTD')

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.start_memory = None
    def start(self):
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used / 1024**3
    def stop(self):
        end_time = time.time()
        end_memory = psutil.virtual_memory().used / 1024**3
        return {
            'execution_time': end_time - self.start_time,
            'memory_used_gb': end_memory - self.start_memory,
            'peak_memory_gb': end_memory
        }

def load_input_json(json_path: str) -> List[Dict]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # data: List[{'text': ..., 'image_path': ...}]
    return data

def load_images(image_paths: List[str]) -> List[Image.Image]:
    images = []
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {path}")
        img = Image.open(path).convert('RGB')
        images.append(img)
    return images

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MMTD 실제 데이터 실험")
    parser.add_argument('--input', type=str, required=True, help='입력 JSON 파일 경로')
    parser.add_argument('--output', type=str, default='mmtd_real_inference_result.json', help='결과 저장 파일')
    args = parser.parse_args()

    # 입력 데이터 로드
    samples = load_input_json(args.input)
    texts = [s['text'] for s in samples]
    image_paths = [s['image_path'] for s in samples]
    images = load_images(image_paths)

    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[INFO] 디바이스: {device}")

    # 모델 로드
    print("[INFO] BERT 다국어 토크나이저 로딩...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
    print("[INFO] DiT 이미지 특성 추출기 로딩...")
    feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/dit-base')

    # 성능 측정
    monitor = PerformanceMonitor()
    monitor.start()

    # 텍스트 인코딩
    print("[INFO] 텍스트 인코딩...")
    text_encoded = tokenizer(
        texts,
        return_tensors='pt',
        max_length=256,
        truncation=True,
        padding='max_length'
    )
    input_ids = text_encoded['input_ids'].to(device)
    attention_mask = text_encoded['attention_mask'].to(device)

    # 이미지 인코딩
    print("[INFO] 이미지 인코딩...")
    image_encoded = feature_extractor(
        images,
        return_tensors='pt'
    )
    pixel_values = image_encoded['pixel_values'].to(device)

    # 멀티모달 융합 (시뮬레이션)
    print("[INFO] 멀티모달 융합(시뮬레이션) 및 분류...")
    batch_size = input_ids.shape[0]
    # 실제 모델이 아니라, 특성 추출 후 임의의 융합 및 분류 시뮬레이션
    text_features = torch.randn(batch_size, 768).to(device)
    image_features = torch.randn(batch_size, 768).to(device)
    fused_features = torch.cat([text_features, image_features], dim=1)
    classifier = torch.nn.Linear(1536, 2).to(device)
    output = classifier(fused_features)
    predictions = torch.softmax(output, dim=1).detach().cpu().numpy()

    # spam/ham 라벨 결정 (1=spam, 0=ham)
    pred_labels = predictions.argmax(axis=1)
    pred_label_names = ["ham" if l == 0 else "spam" for l in pred_labels]

    # 성능 측정 종료
    perf = monitor.stop()

    # 결과 정리
    result = {
        'device': str(device),
        'num_samples': batch_size,
        'input_texts': texts,
        'input_image_paths': image_paths,
        'text_tensor_shape': list(input_ids.shape),
        'image_tensor_shape': list(pixel_values.shape),
        'output_shape': list(predictions.shape),
        'predictions': predictions.tolist(),
        'pred_labels': pred_labels.tolist(),
        'pred_label_names': pred_label_names,
        'performance': perf
    }

    # 결과 저장
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 결과가 {args.output}에 저장되었습니다.")
    print(json.dumps(perf, indent=2, ensure_ascii=False))
    for i, (text, img, pred, label) in enumerate(zip(texts, image_paths, predictions, pred_label_names)):
        print(f"샘플 {i+1}: {label.upper()} (softmax={pred})\n  - 본문: {text[:50]}...\n  - 이미지: {img}")

if __name__ == "__main__":
    main() 
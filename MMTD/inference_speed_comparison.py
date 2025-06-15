import os
import torch
import pandas as pd
import numpy as np
import time
import json
from transformers import BertTokenizerFast, DistilBertTokenizerFast, BeitFeatureExtractor, ViTFeatureExtractor
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 모델 임포트
from models import MMTD
from lightweight_models import LightWeightMMTD, UltraLightMMTD


class InferenceDataset(torch.utils.data.Dataset):
    """추론 속도 테스트용 데이터셋"""
    def __init__(self, data_path, data_df, max_samples=100):
        super(InferenceDataset, self).__init__()
        self.data_path = data_path
        # 추론 속도 테스트를 위해 샘플 수 제한
        self.data = data_df.head(max_samples).reset_index(drop=True)

    def __getitem__(self, item):
        text = str(self.data.iloc[item, 0]) if pd.notna(self.data.iloc[item, 0]) else ""
        pic_path = os.path.join(self.data_path, self.data.iloc[item, 1])
        label = self.data.iloc[item, 2]
        
        try:
            pic = Image.open(pic_path).convert('RGB')
        except Exception as e:
            print(f"이미지 로드 실패: {pic_path}, 기본 이미지 사용")
            pic = Image.new('RGB', (224, 224), color='white')
        
        return text, pic, label

    def __len__(self):
        return len(self.data)


class OriginalCollator:
    """기존 MMTD 모델용 콜레이터"""
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
        self.feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/dit-base')

    def __call__(self, batch):
        texts, images, labels = zip(*batch)
        
        # 텍스트 처리
        text_inputs = self.tokenizer(
            list(texts), 
            return_tensors='pt', 
            max_length=256, 
            truncation=True,
            padding='max_length'
        )
        
        # 이미지 처리
        image_inputs = self.feature_extractor(list(images), return_tensors='pt')
        
        # 결합
        inputs = {}
        inputs.update(text_inputs)
        inputs.update(image_inputs)
        inputs['labels'] = torch.LongTensor(labels)
        
        return inputs


class LightweightCollator:
    """경량화 모델용 콜레이터"""
    def __init__(self):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-multilingual-cased')
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-small-patch16-224')

    def __call__(self, batch):
        texts, images, labels = zip(*batch)
        
        # 텍스트 처리 (DistilBERT는 token_type_ids 불필요)
        text_inputs = self.tokenizer(
            list(texts), 
            return_tensors='pt', 
            max_length=256, 
            truncation=True,
            padding='max_length'
        )
        
        # 이미지 처리
        image_inputs = self.feature_extractor(list(images), return_tensors='pt')
        
        # 결합
        inputs = {}
        inputs.update(text_inputs)
        inputs.update(image_inputs)
        inputs['labels'] = torch.LongTensor(labels)
        
        return inputs


def measure_inference_time(model, dataloader, device, num_warmup=10, num_runs=100):
    """모델의 추론 시간을 측정합니다."""
    model.eval()
    model.to(device)
    
    # GPU 워밍업
    print(f"GPU 워밍업 중... ({num_warmup}회)")
    with torch.no_grad():
        warmup_count = 0
        for batch in dataloader:
            if warmup_count >= num_warmup:
                break
            
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            _ = model(**inputs)
            warmup_count += 1
    
    # 실제 추론 시간 측정
    print(f"추론 시간 측정 중... ({num_runs}회)")
    inference_times = []
    
    with torch.no_grad():
        run_count = 0
        for batch in dataloader:
            if run_count >= num_runs:
                break
            
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # GPU 동기화 후 시간 측정
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            outputs = model(**inputs)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            batch_size = inputs['input_ids'].size(0)
            inference_time = (end_time - start_time) * 1000  # ms 단위
            per_sample_time = inference_time / batch_size
            
            inference_times.append(per_sample_time)
            run_count += 1
    
    return np.array(inference_times)


def load_test_data(data_path='DATA/email_data/EDP.csv', pics_path='DATA/email_data/pics', max_samples=100):
    """테스트 데이터를 로드합니다."""
    if not os.path.exists(data_path):
        print(f"데이터 파일을 찾을 수 없습니다: {data_path}")
        return None, None
    
    data_df = pd.read_csv(data_path)
    data_df.fillna('', inplace=True)
    
    # 테스트용 데이터셋 생성
    test_dataset = InferenceDataset(pics_path, data_df, max_samples)
    
    return test_dataset, data_df


def compare_model_sizes(models_dict):
    """모델 크기를 비교합니다."""
    print("\n" + "="*60)
    print("모델 크기 비교")
    print("="*60)
    
    size_comparison = {}
    
    for name, model in models_dict.items():
        if hasattr(model, 'get_model_size'):
            size_info = model.get_model_size()
        else:
            # 기존 MMTD 모델용
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            size_info = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'total_size_mb': total_params * 4 / (1024 * 1024)
            }
        
        size_comparison[name] = size_info
        
        print(f"{name}:")
        print(f"  총 파라미터: {size_info['total_parameters']:,}")
        print(f"  학습 가능 파라미터: {size_info['trainable_parameters']:,}")
        print(f"  모델 크기: {size_info['total_size_mb']:.2f} MB")
        print()
    
    return size_comparison


def plot_inference_comparison(results, save_path='inference_results'):
    """추론 시간 비교 결과를 시각화합니다."""
    os.makedirs(save_path, exist_ok=True)
    
    # 데이터 준비
    model_names = list(results.keys())
    mean_times = [results[name]['mean_time'] for name in model_names]
    std_times = [results[name]['std_time'] for name in model_names]
    
    # 1. 평균 추론 시간 비교
    plt.figure(figsize=(12, 8))
    
    bars = plt.bar(model_names, mean_times, yerr=std_times, capsize=5, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    
    plt.title('모델별 평균 추론 시간 비교', fontsize=16, fontweight='bold')
    plt.xlabel('모델', fontsize=12)
    plt.ylabel('추론 시간 (ms/sample)', fontsize=12)
    plt.xticks(rotation=45)
    
    # 값 표시
    for bar, mean, std in zip(bars, mean_times, std_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                f'{mean:.2f}±{std:.2f}ms', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/inference_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 속도 향상 비교 (기존 모델 대비)
    if 'MMTD (Original)' in results:
        baseline_time = results['MMTD (Original)']['mean_time']
        speedup_ratios = []
        speedup_labels = []
        
        for name in model_names:
            if name != 'MMTD (Original)':
                speedup = baseline_time / results[name]['mean_time']
                speedup_ratios.append(speedup)
                speedup_labels.append(name)
        
        if speedup_ratios:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(speedup_labels, speedup_ratios, 
                          color=['#4ECDC4', '#45B7D1'], alpha=0.8)
            
            plt.title('기존 모델 대비 속도 향상', fontsize=16, fontweight='bold')
            plt.xlabel('경량화 모델', fontsize=12)
            plt.ylabel('속도 향상 배수', fontsize=12)
            plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='기준선 (1x)')
            
            # 값 표시
            for bar, ratio in zip(bars, speedup_ratios):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{ratio:.2f}x', ha='center', va='bottom', fontweight='bold')
            
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{save_path}/speedup_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 3. 추론 시간 분포 박스플롯
    plt.figure(figsize=(12, 8))
    
    data_for_boxplot = []
    labels_for_boxplot = []
    
    for name in model_names:
        data_for_boxplot.append(results[name]['all_times'])
        labels_for_boxplot.append(name)
    
    plt.boxplot(data_for_boxplot, labels=labels_for_boxplot)
    plt.title('모델별 추론 시간 분포', fontsize=16, fontweight='bold')
    plt.xlabel('모델', fontsize=12)
    plt.ylabel('추론 시간 (ms/sample)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/inference_time_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """메인 실험 함수"""
    print("MMTD 모델 추론 속도 비교 실험")
    print("="*60)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 테스트 데이터 로드
    print("\n테스트 데이터 로딩...")
    test_dataset, data_df = load_test_data(max_samples=100)
    
    if test_dataset is None:
        print("테스트 데이터를 로드할 수 없습니다.")
        return
    
    print(f"테스트 샘플 수: {len(test_dataset)}")
    
    # 모델 초기화
    print("\n모델 초기화...")
    models = {
        'MMTD (Original)': MMTD(),
        'LightWeightMMTD': LightWeightMMTD(),
        'UltraLightMMTD': UltraLightMMTD()
    }
    
    # 모델 크기 비교
    size_comparison = compare_model_sizes(models)
    
    # 추론 시간 측정
    results = {}
    batch_size = 8  # 작은 배치 크기로 안정적인 측정
    
    for model_name, model in models.items():
        print(f"\n{model_name} 추론 시간 측정...")
        
        try:
            # 적절한 콜레이터 선택
            if 'Original' in model_name:
                collator = OriginalCollator()
            else:
                collator = LightweightCollator()
            
            # 데이터로더 생성
            dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=0
            )
            
            # 추론 시간 측정
            inference_times = measure_inference_time(
                model, dataloader, device, 
                num_warmup=10, num_runs=50
            )
            
            # 통계 계산
            results[model_name] = {
                'mean_time': np.mean(inference_times),
                'std_time': np.std(inference_times),
                'min_time': np.min(inference_times),
                'max_time': np.max(inference_times),
                'median_time': np.median(inference_times),
                'all_times': inference_times.tolist()
            }
            
            print(f"  평균 추론 시간: {results[model_name]['mean_time']:.2f} ± {results[model_name]['std_time']:.2f} ms/sample")
            print(f"  최소/최대: {results[model_name]['min_time']:.2f} / {results[model_name]['max_time']:.2f} ms")
            
        except Exception as e:
            print(f"  {model_name} 측정 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        print("측정 가능한 모델이 없습니다.")
        return
    
    # 결과 요약
    print("\n" + "="*60)
    print("추론 속도 비교 결과 요약")
    print("="*60)
    
    # 테이블 형태로 출력
    print(f"{'모델':<20} {'평균 시간(ms)':<15} {'표준편차(ms)':<15} {'속도 향상':<10}")
    print("-" * 65)
    
    baseline_time = None
    if 'MMTD (Original)' in results:
        baseline_time = results['MMTD (Original)']['mean_time']
    
    for model_name, result in results.items():
        mean_time = result['mean_time']
        std_time = result['std_time']
        
        if baseline_time and model_name != 'MMTD (Original)':
            speedup = baseline_time / mean_time
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "기준"
        
        print(f"{model_name:<20} {mean_time:<15.2f} {std_time:<15.2f} {speedup_str:<10}")
    
    # 모델 크기와 속도 종합 비교
    print("\n" + "="*60)
    print("모델 크기 vs 추론 속도 종합 비교")
    print("="*60)
    
    for model_name in results.keys():
        if model_name in size_comparison:
            size_mb = size_comparison[model_name]['total_size_mb']
            mean_time = results[model_name]['mean_time']
            
            if baseline_time and model_name != 'MMTD (Original)':
                size_reduction = (size_comparison['MMTD (Original)']['total_size_mb'] - size_mb) / size_comparison['MMTD (Original)']['total_size_mb'] * 100
                speed_improvement = (baseline_time / mean_time - 1) * 100
                print(f"{model_name}:")
                print(f"  모델 크기: {size_mb:.1f} MB ({size_reduction:.1f}% 감소)")
                print(f"  추론 시간: {mean_time:.2f} ms ({speed_improvement:.1f}% 향상)")
                print()
    
    # 결과 저장
    os.makedirs('inference_results', exist_ok=True)
    
    # JSON으로 상세 결과 저장
    combined_results = {
        'inference_times': results,
        'model_sizes': size_comparison,
        'experiment_config': {
            'device': str(device),
            'batch_size': batch_size,
            'num_test_samples': len(test_dataset),
            'num_warmup_runs': 10,
            'num_measurement_runs': 50
        }
    }
    
    with open('inference_results/detailed_results.json', 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    
    # 시각화
    plot_inference_comparison(results)
    
    print(f"\n실험 완료! 결과가 'inference_results/' 폴더에 저장되었습니다.")
    print(f"측정된 모델 수: {len(results)}")


if __name__ == "__main__":
    main() 
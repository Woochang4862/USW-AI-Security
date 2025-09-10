import os
import torch
import pandas as pd
import numpy as np
import time
import json
from transformers import (
    BertTokenizer, BeitImageProcessor,
    BertForSequenceClassification, BeitForImageClassification
)
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 모델 임포트
from models import MMTD

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'AppleGothic']
plt.rcParams['axes.unicode_minus'] = False


class EmailDataset(Dataset):
    """이메일 데이터셋 클래스"""
    def __init__(self, data_path, data_df, max_samples=None):
        super(EmailDataset, self).__init__()
        self.data_path = data_path
        if max_samples:
            self.data = data_df.head(max_samples).reset_index(drop=True)
        else:
            self.data = data_df.reset_index(drop=True)

    def __getitem__(self, item):
        text = str(self.data.iloc[item, 0]) if pd.notna(self.data.iloc[item, 0]) else ""
        pic_path = os.path.join(self.data_path, self.data.iloc[item, 1])
        label = self.data.iloc[item, 2]
        
        try:
            pic = Image.open(pic_path).convert('RGB')
        except Exception as e:
            # 이미지 로드 실패 시 기본 이미지 사용
            pic = Image.new('RGB', (224, 224), color='white')
        
        return text, pic, label

    def __len__(self):
        return len(self.data)


class MMTDCollator:
    """MMTD 모델용 콜레이터"""
    def __init__(self, tokenizer_name="google-bert/bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.image_processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")

    def __call__(self, batch):
        texts, images, labels = zip(*batch)
        
        # 텍스트 토크나이징
        text_inputs = self.tokenizer(
            list(texts),
            return_tensors='pt',
            max_length=256,
            truncation=True,
            padding='max_length'
        )
        
        # 이미지 전처리
        image_inputs = self.image_processor(list(images), return_tensors='pt')
        
        # 입력 결합
        inputs = {}
        inputs.update(text_inputs)
        inputs.update(image_inputs)
        inputs['labels'] = torch.LongTensor(labels)
        
        return inputs


def load_mmtd_model(checkpoint_path, device):
    """MMTD 모델을 로드하고 체크포인트를 적용합니다."""
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return None
    
    try:
        # 체크포인트 먼저 로드하여 구조 확인
        print(f"🔄 체크포인트 로딩 중: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # MMTD 모델 초기화 (체크포인트 구조에 맞춰)
        model = MMTD(
            bert_pretrain_weight=None,  # 사전 훈련된 가중치 사용하지 않음
            beit_pretrain_weight=None,  # 사전 훈련된 가중치 사용하지 않음
            device=device
        )
        
        # 체크포인트에서 모델 구조 정보 추출
        print("📋 체크포인트 구조 분석 중...")
        
        # BERT 토크나이저 vocab 크기 확인
        if 'text_encoder.bert.embeddings.word_embeddings.weight' in checkpoint:
            vocab_size = checkpoint['text_encoder.bert.embeddings.word_embeddings.weight'].shape[0]
            print(f"   BERT vocab 크기: {vocab_size}")
            
            # 적절한 토크나이저로 BERT 재초기화
            if vocab_size > 50000:  # multilingual 모델
                from transformers import BertForSequenceClassification, BertConfig
                bert_config = BertConfig.from_pretrained("google-bert/bert-base-multilingual-cased")
                model.text_encoder = BertForSequenceClassification(bert_config)
                model.text_encoder.config.output_hidden_states = True
                print("   Multilingual BERT 구조로 초기화")
            else:
                from transformers import BertForSequenceClassification, BertConfig
                bert_config = BertConfig.from_pretrained("google-bert/bert-base-uncased")
                model.text_encoder = BertForSequenceClassification(bert_config)
                model.text_encoder.config.output_hidden_states = True
                print("   English BERT 구조로 초기화")
        
        # BEiT 분류기 크기 확인
        if 'image_encoder.classifier.weight' in checkpoint:
            num_classes = checkpoint['image_encoder.classifier.weight'].shape[0]
            print(f"   BEiT 클래스 수: {num_classes}")
            
            if num_classes == 2:  # 이진 분류용으로 훈련됨
                from transformers import BeitForImageClassification, BeitConfig
                beit_config = BeitConfig.from_pretrained("microsoft/beit-base-patch16-224")
                beit_config.num_labels = 2
                model.image_encoder = BeitForImageClassification(beit_config)
                model.image_encoder.config.output_hidden_states = True
                print("   2클래스 BEiT 구조로 초기화")
        
        # 체크포인트 로드 (strict=False로 설정하여 일부 키 불일치 허용)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        
        if missing_keys:
            print(f"   누락된 키: {len(missing_keys)}개")
        if unexpected_keys:
            print(f"   예상치 못한 키: {len(unexpected_keys)}개")
        
        model.to(device)
        model.eval()
        
        print(f"✅ MMTD 모델 로드 성공: {checkpoint_path}")
        return model
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {checkpoint_path}")
        print(f"   오류: {str(e)}")
        return None


def measure_inference_time(model, dataloader, device, num_warmup=10):
    """모델의 추론 시간을 측정합니다."""
    model.eval()
    
    # GPU 워밍업
    print(f"  🔥 GPU 워밍업 중... ({num_warmup}회)")
    with torch.no_grad():
        warmup_count = 0
        for batch in dataloader:
            if warmup_count >= num_warmup:
                break
            
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            try:
                _ = model(**inputs)
                warmup_count += 1
            except Exception as e:
                print(f"    ⚠️ 워밍업 중 오류 (무시됨): {str(e)}")
                warmup_count += 1
                continue
    
    # 실제 추론 시간 측정
    print(f"  ⏱️ 추론 시간 측정 중...")
    inference_times = []
    total_samples = 0
    successful_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="  MMTD 추론")):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            try:
                # GPU 동기화 후 시간 측정
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                outputs = model(**inputs)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                batch_size = inputs['input_ids'].size(0)
                batch_time = (end_time - start_time) * 1000  # ms 단위
                per_sample_time = batch_time / batch_size
                
                inference_times.extend([per_sample_time] * batch_size)
                total_samples += batch_size
                successful_batches += 1
                
            except Exception as e:
                print(f"    ⚠️ 배치 {batch_idx} 처리 중 오류: {str(e)}")
                continue
    
    if not inference_times:
        print(f"    ❌ 측정 가능한 추론 시간이 없습니다.")
        return None
    
    print(f"    ✅ 성공적으로 처리된 샘플: {total_samples}/{len(dataloader.dataset)}")
    print(f"    ✅ 성공적으로 처리된 배치: {successful_batches}/{len(dataloader)}")
    
    return np.array(inference_times)


def get_model_size_info(model):
    """모델 크기 정보를 반환합니다."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'total_size_mb': total_params * 4 / (1024 * 1024)  # float32 기준
    }


def plot_inference_results(inference_times, save_path='mmtd_inference_results'):
    """추론 시간 결과를 시각화합니다."""
    os.makedirs(save_path, exist_ok=True)
    
    # 1. 추론 시간 분포 히스토그램
    plt.figure(figsize=(12, 8))
    plt.hist(inference_times, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(inference_times), color='red', linestyle='--', 
                label=f'평균: {np.mean(inference_times):.3f} ms')
    plt.axvline(np.median(inference_times), color='green', linestyle='--', 
                label=f'중앙값: {np.median(inference_times):.3f} ms')
    
    plt.title('MMTD 모델 추론 시간 분포', fontsize=16, fontweight='bold')
    plt.xlabel('추론 시간 (ms/sample)', fontsize=12)
    plt.ylabel('빈도', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/inference_time_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 추론 시간 박스플롯
    plt.figure(figsize=(10, 6))
    plt.boxplot(inference_times, vert=True)
    plt.title('MMTD 모델 추론 시간 박스플롯', fontsize=16, fontweight='bold')
    plt.ylabel('추론 시간 (ms/sample)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/inference_time_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 시간에 따른 추론 시간 변화 (샘플 순서대로)
    plt.figure(figsize=(14, 8))
    sample_indices = range(len(inference_times))
    plt.plot(sample_indices, inference_times, alpha=0.7, linewidth=0.5)
    plt.axhline(np.mean(inference_times), color='red', linestyle='--', 
                label=f'평균: {np.mean(inference_times):.3f} ms')
    
    plt.title('MMTD 모델 샘플별 추론 시간 변화', fontsize=16, fontweight='bold')
    plt.xlabel('샘플 인덱스', fontsize=12)
    plt.ylabel('추론 시간 (ms/sample)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/inference_time_sequence.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """메인 벤치마크 함수"""
    print("🚀 MMTD 모델 추론 시간 벤치마크")
    print("="*60)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 사용 디바이스: {device}")
    
    # 체크포인트 경로 설정
    checkpoint_path = "checkpoints/fold5/checkpoint-939/pytorch_model.bin"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        print("   다음 경로들을 확인해보세요:")
        possible_paths = [
            "checkpoints/fold5/checkpoint-939/pytorch_model.bin",
            "checkpoints/pytorch_model.bin",
            "best_model.pth",
            "model.pth"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                print(f"   ✅ 발견: {path}")
                checkpoint_path = path
                break
            else:
                print(f"   ❌ 없음: {path}")
        
        if not os.path.exists(checkpoint_path):
            return
    
    # 데이터 로드
    data_path = 'DATA/email_data/EDP.csv'
    pics_path = 'DATA/email_data/pics'
    
    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        return
    
    print(f"📊 데이터 로딩: {data_path}")
    data_df = pd.read_csv(data_path)
    data_df.fillna('', inplace=True)
    
    # 전체 데이터셋 사용
    test_dataset = EmailDataset(pics_path, data_df)
    print(f"📈 전체 데이터셋 크기: {len(test_dataset)} 샘플")
    
    # 모델 로드
    print(f"\n🎯 MMTD 모델 로딩")
    print(f"📁 체크포인트: {checkpoint_path}")
    
    model = load_mmtd_model(checkpoint_path, device)
    if model is None:
        return
    
    # 모델 크기 정보
    size_info = get_model_size_info(model)
    print(f"📏 모델 파라미터: {size_info['total_parameters']:,}")
    print(f"💾 모델 크기: {size_info['total_size_mb']:.2f} MB")
    
    # 적절한 토크나이저 결정
    tokenizer_name = "google-bert/bert-base-uncased"
    if hasattr(model.text_encoder, 'config') and hasattr(model.text_encoder.config, 'vocab_size'):
        if model.text_encoder.config.vocab_size > 50000:
            tokenizer_name = "google-bert/bert-base-multilingual-cased"
            print(f"🔤 Multilingual 토크나이저 사용: {tokenizer_name}")
        else:
            print(f"🔤 English 토크나이저 사용: {tokenizer_name}")
    
    # 콜레이터 및 데이터로더 생성
    collator = MMTDCollator(tokenizer_name)
    batch_size = 8  # 안정적인 추론을 위한 작은 배치 크기
    
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )
    
    print(f"⚙️ 배치 크기: {batch_size}")
    print(f"📦 총 배치 수: {len(dataloader)}")
    
    # 추론 시간 측정
    print(f"\n{'='*60}")
    print(f"⏱️ 추론 시간 측정 시작")
    print(f"{'='*60}")
    
    inference_times = measure_inference_time(
        model, dataloader, device, num_warmup=5
    )
    
    if inference_times is None:
        print(f"❌ 추론 시간 측정에 실패했습니다.")
        return
    
    # 결과 분석
    print(f"\n{'='*60}")
    print(f"📊 추론 시간 벤치마크 결과")
    print(f"{'='*60}")
    
    results = {
        'inference_times': inference_times.tolist(),
        'mean_time': np.mean(inference_times),
        'std_time': np.std(inference_times),
        'min_time': np.min(inference_times),
        'max_time': np.max(inference_times),
        'median_time': np.median(inference_times),
        'q25_time': np.percentile(inference_times, 25),
        'q75_time': np.percentile(inference_times, 75),
        'model_size_mb': size_info['total_size_mb'],
        'total_parameters': size_info['total_parameters'],
        'samples_processed': len(inference_times),
        'total_time_seconds': np.sum(inference_times) / 1000,
        'throughput_samples_per_second': len(inference_times) / (np.sum(inference_times) / 1000)
    }
    
    print(f"📈 통계 요약:")
    print(f"   평균 추론 시간: {results['mean_time']:.3f} ± {results['std_time']:.3f} ms/sample")
    print(f"   중앙값: {results['median_time']:.3f} ms/sample")
    print(f"   최소/최대: {results['min_time']:.3f} / {results['max_time']:.3f} ms")
    print(f"   25%/75% 분위수: {results['q25_time']:.3f} / {results['q75_time']:.3f} ms")
    print(f"   처리된 샘플 수: {results['samples_processed']:,}")
    print(f"   총 처리 시간: {results['total_time_seconds']:.2f} 초")
    print(f"   처리량: {results['throughput_samples_per_second']:.2f} samples/sec")
    
    # 성능 분석
    print(f"\n📊 성능 분석:")
    if results['mean_time'] < 1.0:
        performance_level = "🚀 매우 빠름"
    elif results['mean_time'] < 2.0:
        performance_level = "⚡ 빠름"
    elif results['mean_time'] < 5.0:
        performance_level = "👍 보통"
    elif results['mean_time'] < 10.0:
        performance_level = "⚠️ 느림"
    else:
        performance_level = "🐌 매우 느림"
    
    print(f"   성능 등급: {performance_level}")
    print(f"   변동 계수: {(results['std_time']/results['mean_time']*100):.2f}% {'(안정적)' if results['std_time']/results['mean_time'] < 0.1 else '(불안정)'}")
    
    # 결과 저장
    save_path = 'mmtd_inference_results'
    os.makedirs(save_path, exist_ok=True)
    
    # JSON으로 상세 결과 저장
    experiment_info = {
        'device': str(device),
        'batch_size': batch_size,
        'total_samples': len(test_dataset),
        'num_warmup_runs': 5,
        'checkpoint_path': checkpoint_path,
        'model_type': 'MMTD',
        'text_encoder': 'google-bert/bert-base-uncased',
        'image_encoder': 'microsoft/beit-base-patch16-224'
    }
    
    full_results = {
        'results': results,
        'experiment_config': experiment_info
    }
    
    with open(f'{save_path}/mmtd_inference_results.json', 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    
    # CSV로 요약 결과 저장
    summary_data = [{
        'Model': 'MMTD (BERT+BEiT)',
        'Checkpoint': checkpoint_path,
        'Mean_Inference_Time_ms': results['mean_time'],
        'Std_Inference_Time_ms': results['std_time'],
        'Min_Inference_Time_ms': results['min_time'],
        'Max_Inference_Time_ms': results['max_time'],
        'Median_Inference_Time_ms': results['median_time'],
        'Q25_Inference_Time_ms': results['q25_time'],
        'Q75_Inference_Time_ms': results['q75_time'],
        'Model_Size_MB': results['model_size_mb'],
        'Total_Parameters': results['total_parameters'],
        'Samples_Processed': results['samples_processed'],
        'Total_Time_Seconds': results['total_time_seconds'],
        'Throughput_Samples_Per_Second': results['throughput_samples_per_second']
    }]
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{save_path}/mmtd_benchmark_summary.csv', index=False)
    
    # 시각화
    plot_inference_results(inference_times, save_path)
    
    print(f"\n💾 결과 저장 완료:")
    print(f"   📁 폴더: {save_path}/")
    print(f"   📄 상세 결과: mmtd_inference_results.json")
    print(f"   📊 요약 CSV: mmtd_benchmark_summary.csv")
    print(f"   📈 시각화: *.png 파일들")
    
    print(f"\n🎉 MMTD 모델 벤치마크 완료!")
    print(f"   평균 추론 시간: {results['mean_time']:.3f} ms/sample")
    print(f"   처리량: {results['throughput_samples_per_second']:.2f} samples/sec")
    
    # 메모리 정리
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main() 
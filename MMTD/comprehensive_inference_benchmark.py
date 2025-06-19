import os
import torch
import pandas as pd
import numpy as np
import time
import json
from transformers import (
    DistilBertTokenizerFast, ViTFeatureExtractor, 
    DistilBertForSequenceClassification, AutoTokenizer, 
    AutoFeatureExtractor, DeiTForImageClassification, 
    ViTForImageClassification, AutoModelForSequenceClassification, 
    MobileBertForSequenceClassification, MobileViTForImageClassification, 
    MobileBertTokenizer, MobileViTImageProcessor, AutoImageProcessor, 
    AutoModelForImageClassification
)
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 모델 임포트
from lightweight_models import GeneralizedMMTD
from models import PretrainedMMTD, HybridMMTD, HybridMMTDTextTrainable

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


class DynamicCollator:
    """동적 콜레이터 - 모델에 따라 적절한 토크나이저와 프로세서 사용"""
    def __init__(self, text_model_name, image_model_name):
        # 텍스트 토크나이저 분기
        if "mobilebert" in text_model_name:
            self.tokenizer = MobileBertTokenizer.from_pretrained(text_model_name)
        elif "tinybert" in text_model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        elif "bert" in text_model_name and "distil" not in text_model_name:
            # BERT 기본 모델
            self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        else:
            # DistilBERT 등 기타
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(text_model_name)
            
        # 이미지 프로세서 분기
        if "mobilevit" in image_model_name:
            self.feature_extractor = MobileViTImageProcessor.from_pretrained(image_model_name)
        elif "deit" in image_model_name:
            self.feature_extractor = AutoImageProcessor.from_pretrained(image_model_name)
        elif "beit" in image_model_name:
            self.feature_extractor = AutoImageProcessor.from_pretrained(image_model_name)
        elif "vit-tiny" in image_model_name:
            self.feature_extractor = AutoImageProcessor.from_pretrained(image_model_name)
        else:
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(image_model_name)

    def __call__(self, batch):
        texts, images, labels = zip(*batch)
        text_inputs = self.tokenizer(
            list(texts),
            return_tensors='pt',
            max_length=256,
            truncation=True,
            padding='max_length'
        )
        image_inputs = self.feature_extractor(list(images), return_tensors='pt')
        inputs = {}
        inputs.update(text_inputs)
        inputs.update(image_inputs)
        inputs['labels'] = torch.LongTensor(labels)
        return inputs


def get_experiment_configs():
    """실험 설정을 반환합니다."""
    return {
        # === 기본 경량화 모델들 ===
        "mobilebert_mobilevit": {
            "model_class": GeneralizedMMTD,
            "text_encoder_cls": MobileBertForSequenceClassification,
            "image_encoder_cls": MobileViTForImageClassification,
            "text_encoder_name": "google/mobilebert-uncased",
            "image_encoder_name": "apple/mobilevit-small",
            "checkpoint_path": "outputs/mobilebert_mobilevit/best_model.pth",
        },
        "mobilebert_deit": {
            "model_class": GeneralizedMMTD,
            "text_encoder_cls": MobileBertForSequenceClassification,
            "image_encoder_cls": AutoModelForImageClassification,
            "text_encoder_name": "google/mobilebert-uncased",
            "image_encoder_name": "facebook/deit-base-patch16-224",
            "checkpoint_path": "outputs/mobilebert_deit/best_model.pth",
        },
        "distilbert_mobilevit": {
            "model_class": GeneralizedMMTD,
            "text_encoder_cls": DistilBertForSequenceClassification,
            "image_encoder_cls": MobileViTForImageClassification,
            "text_encoder_name": "distilbert-base-multilingual-cased",
            "image_encoder_name": "apple/mobilevit-small",
            "checkpoint_path": "outputs/distilbert_mobilevit/best_model.pth",
        },
        "distilbert_deit": {
            "model_class": GeneralizedMMTD,
            "text_encoder_cls": DistilBertForSequenceClassification,
            "image_encoder_cls": AutoModelForImageClassification,
            "text_encoder_name": "distilbert-base-multilingual-cased",
            "image_encoder_name": "facebook/deit-base-patch16-224",
            "checkpoint_path": "outputs/distilbert_deit/best_model.pth",
        },
        "tinybert_vit-tiny": {
            "model_class": GeneralizedMMTD,
            "text_encoder_cls": AutoModelForSequenceClassification,
            "image_encoder_cls": AutoModelForImageClassification,
            "text_encoder_name": "huawei-noah/TinyBERT_General_4L_312D",
            "image_encoder_name": "WinKawaks/vit-tiny-patch16-224",
            "checkpoint_path": "outputs/tinybert_vit-tiny/best_model.pth",
        },
        "tinybert_deit": {
            "model_class": GeneralizedMMTD,
            "text_encoder_cls": AutoModelForSequenceClassification,
            "image_encoder_cls": AutoModelForImageClassification,
            "text_encoder_name": "huawei-noah/TinyBERT_General_4L_312D",
            "image_encoder_name": "facebook/deit-base-patch16-224",
            "checkpoint_path": "outputs/tinybert_deit/best_model.pth",
        },
        "tinybert_mobilevit": {
            "model_class": GeneralizedMMTD,
            "text_encoder_cls": AutoModelForSequenceClassification,
            "image_encoder_cls": MobileViTForImageClassification,
            "text_encoder_name": "huawei-noah/TinyBERT_General_4L_312D",
            "image_encoder_name": "apple/mobilevit-small",
            "checkpoint_path": "outputs/tinybert_mobilevit/best_model.pth",
        },
        "distilbert_vit-tiny": {
            "model_class": GeneralizedMMTD,
            "text_encoder_cls": DistilBertForSequenceClassification,
            "image_encoder_cls": AutoModelForImageClassification,
            "text_encoder_name": "distilbert-base-multilingual-cased",
            "image_encoder_name": "WinKawaks/vit-tiny-patch16-224",
            "checkpoint_path": "outputs/distilbert_vit-tiny/best_model.pth",
        },
        "mobilebert_vit-tiny": {
            "model_class": GeneralizedMMTD,
            "text_encoder_cls": MobileBertForSequenceClassification,
            "image_encoder_cls": AutoModelForImageClassification,
            "text_encoder_name": "google/mobilebert-uncased",
            "image_encoder_name": "WinKawaks/vit-tiny-patch16-224",
            "checkpoint_path": "outputs/mobilebert_vit-tiny/best_model.pth",
        },
        
        # === 하이브리드 모델들 (BERT 기반) ===
        "bert_deit": {
            "model_class": HybridMMTD,
            "image_encoder_cls": AutoModelForImageClassification,
            "image_encoder_name": "facebook/deit-base-patch16-224",
            "text_encoder_name": "google-bert/bert-base-uncased",
            "checkpoint_path": "outputs/bert_deit/best_model.pth",
            "pretrained_checkpoint": "checkpoints/fold5/checkpoint-939/pytorch_model.bin",
        },
        "bert_mobilevit": {
            "model_class": HybridMMTD,
            "image_encoder_cls": MobileViTForImageClassification,
            "image_encoder_name": "apple/mobilevit-small",
            "text_encoder_name": "google-bert/bert-base-uncased",
            "checkpoint_path": "outputs/bert_mobilevit/best_model.pth",
            "pretrained_checkpoint": "checkpoints/fold5/checkpoint-939/pytorch_model.bin",
        },
        "bert_vit-tiny": {
            "model_class": HybridMMTD,
            "image_encoder_cls": AutoModelForImageClassification,
            "image_encoder_name": "WinKawaks/vit-tiny-patch16-224",
            "text_encoder_name": "google-bert/bert-base-uncased",
            "checkpoint_path": "outputs/bert_vit-tiny/best_model.pth",
            "pretrained_checkpoint": "checkpoints/fold5/checkpoint-939/pytorch_model.bin",
        },
        
        # === 하이브리드 모델들 (BEiT 기반) ===
        "mobilebert_beit": {
            "model_class": HybridMMTDTextTrainable,
            "text_encoder_cls": MobileBertForSequenceClassification,
            "text_encoder_name": "google/mobilebert-uncased",
            "image_encoder_name": "microsoft/beit-base-patch16-224",
            "checkpoint_path": "outputs/mobilebert_beit/best_model.pth",
            "pretrained_checkpoint": "checkpoints/fold5/checkpoint-939/pytorch_model.bin",
        },
        "distilbert_beit": {
            "model_class": HybridMMTDTextTrainable,
            "text_encoder_cls": DistilBertForSequenceClassification,
            "text_encoder_name": "distilbert-base-multilingual-cased",
            "image_encoder_name": "microsoft/beit-base-patch16-224",
            "checkpoint_path": "outputs/distilbert_beit/best_model.pth",
            "pretrained_checkpoint": "checkpoints/fold5/checkpoint-939/pytorch_model.bin",
        },
        "tinybert_beit": {
            "model_class": HybridMMTDTextTrainable,
            "text_encoder_cls": AutoModelForSequenceClassification,
            "text_encoder_name": "huawei-noah/TinyBERT_General_4L_312D",
            "image_encoder_name": "microsoft/beit-base-patch16-224",
            "checkpoint_path": "outputs/tinybert_beit/best_model.pth",
            "pretrained_checkpoint": "checkpoints/fold5/checkpoint-939/pytorch_model.bin",
        },
    }


def load_model_with_checkpoint(config, device):
    """설정에 따라 모델을 로드하고 체크포인트를 적용합니다."""
    
    if not os.path.exists(config["checkpoint_path"]):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {config['checkpoint_path']}")
        return None
    
    try:
        if config["model_class"] == GeneralizedMMTD:
            # 일반 경량화 모델
            model = config["model_class"](
                text_encoder_cls=config["text_encoder_cls"],
                image_encoder_cls=config["image_encoder_cls"],
                text_pretrain_weight=config["text_encoder_name"],
                image_pretrain_weight=config["image_encoder_name"]
            )
        elif config["model_class"] == HybridMMTD:
            # BERT 기반 하이브리드 모델
            model = config["model_class"](
                pretrained_checkpoint_path=config["pretrained_checkpoint"],
                image_encoder_cls=config["image_encoder_cls"],
                image_pretrain_weight=config["image_encoder_name"],
                device=device
            )
        elif config["model_class"] == HybridMMTDTextTrainable:
            # BEiT 기반 하이브리드 모델
            model = config["model_class"](
                pretrained_checkpoint_path=config["pretrained_checkpoint"],
                text_encoder_cls=config["text_encoder_cls"],
                text_pretrain_weight=config["text_encoder_name"],
                device=device
            )
        else:
            print(f"❌ 지원하지 않는 모델 클래스: {config['model_class']}")
            return None
        
        # 체크포인트 로드
        checkpoint = torch.load(config["checkpoint_path"], map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        model.to(device)
        model.eval()
        
        print(f"✅ 모델 로드 성공: {config['checkpoint_path']}")
        return model
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {config['checkpoint_path']}")
        print(f"   오류: {str(e)}")
        return None


def measure_inference_time(model, dataloader, device, model_name, num_warmup=10):
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
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"  {model_name}")):
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
    if hasattr(model, 'get_model_size'):
        return model.get_model_size()
    else:
        # 기본 크기 계산
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_size_mb': total_params * 4 / (1024 * 1024)
        }


def plot_comprehensive_results(results, save_path='inference_benchmark_results'):
    """종합적인 결과를 시각화합니다."""
    os.makedirs(save_path, exist_ok=True)
    
    # 데이터 준비
    model_names = []
    mean_times = []
    std_times = []
    model_sizes = []
    text_encoders = []
    image_encoders = []
    
    for model_name, result in results.items():
        if result['inference_times'] is not None:
            model_names.append(model_name)
            mean_times.append(result['mean_time'])
            std_times.append(result['std_time'])
            model_sizes.append(result['model_size_mb'])
            
            # 텍스트/이미지 인코더 분류
            if 'bert' in model_name and 'distil' not in model_name and 'mobile' not in model_name and 'tiny' not in model_name:
                text_enc = 'BERT'
            elif 'distilbert' in model_name:
                text_enc = 'DistilBERT'
            elif 'mobilebert' in model_name:
                text_enc = 'MobileBERT'
            elif 'tinybert' in model_name:
                text_enc = 'TinyBERT'
            else:
                text_enc = 'Other'
            text_encoders.append(text_enc)
            
            if 'deit' in model_name:
                img_enc = 'DeiT'
            elif 'mobilevit' in model_name:
                img_enc = 'MobileViT'
            elif 'beit' in model_name:
                img_enc = 'BEiT'
            elif 'vit-tiny' in model_name:
                img_enc = 'ViT-Tiny'
            else:
                img_enc = 'Other'
            image_encoders.append(img_enc)
    
    # 1. 추론 시간 비교 (바 차트)
    plt.figure(figsize=(16, 10))
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    bars = plt.bar(range(len(model_names)), mean_times, yerr=std_times, 
                   capsize=5, color=colors, alpha=0.8)
    
    plt.title('모델별 평균 추론 시간 비교\n(전체 데이터셋 기준)', fontsize=16, fontweight='bold')
    plt.xlabel('모델', fontsize=12)
    plt.ylabel('추론 시간 (ms/sample)', fontsize=12)
    plt.xticks(range(len(model_names)), [name.replace('_', '\n') for name in model_names], 
               rotation=45, ha='right')
    
    # 값 표시
    for i, (bar, mean, std) in enumerate(zip(bars, mean_times, std_times)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + max(mean_times)*0.01,
                f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/inference_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 모델 크기 vs 추론 시간 scatter plot
    plt.figure(figsize=(14, 10))
    
    # 텍스트 인코더별 색상 분류
    unique_text_encoders = list(set(text_encoders))
    colors_dict = {
        'BERT': '#1f77b4',
        'DistilBERT': '#ff7f0e',
        'MobileBERT': '#d62728',
        'TinyBERT': '#2ca02c'
    }
    
    for text_enc in unique_text_encoders:
        indices = [i for i, enc in enumerate(text_encoders) if enc == text_enc]
        x_vals = [model_sizes[i] for i in indices]
        y_vals = [mean_times[i] for i in indices]
        plt.scatter(x_vals, y_vals, c=colors_dict.get(text_enc, '#7f7f7f'), 
                   label=text_enc, s=100, alpha=0.7, edgecolors='black')
        
        # 모델명 라벨 추가
        for i in indices:
            plt.annotate(model_names[i].replace('_', '\n'), 
                        (model_sizes[i], mean_times[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.xlabel('모델 크기 (MB)', fontsize=12, fontweight='bold')
    plt.ylabel('평균 추론 시간 (ms/sample)', fontsize=12, fontweight='bold')
    plt.title('모델 크기 vs 추론 시간\n(전체 데이터셋 추론 성능)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_path}/size_vs_inference_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 텍스트 인코더별 성능 비교
    plt.figure(figsize=(12, 8))
    text_enc_data = {}
    for i, text_enc in enumerate(text_encoders):
        if text_enc not in text_enc_data:
            text_enc_data[text_enc] = []
        text_enc_data[text_enc].append(mean_times[i])
    
    box_data = [text_enc_data[enc] for enc in sorted(text_enc_data.keys())]
    box_labels = sorted(text_enc_data.keys())
    
    plt.boxplot(box_data, labels=box_labels)
    plt.title('텍스트 인코더별 추론 시간 분포', fontsize=14, fontweight='bold')
    plt.xlabel('텍스트 인코더', fontsize=12)
    plt.ylabel('추론 시간 (ms/sample)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/text_encoder_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 이미지 인코더별 성능 비교
    plt.figure(figsize=(12, 8))
    img_enc_data = {}
    for i, img_enc in enumerate(image_encoders):
        if img_enc not in img_enc_data:
            img_enc_data[img_enc] = []
        img_enc_data[img_enc].append(mean_times[i])
    
    box_data = [img_enc_data[enc] for enc in sorted(img_enc_data.keys())]
    box_labels = sorted(img_enc_data.keys())
    
    plt.boxplot(box_data, labels=box_labels)
    plt.title('이미지 인코더별 추론 시간 분포', fontsize=14, fontweight='bold')
    plt.xlabel('이미지 인코더', fontsize=12)
    plt.ylabel('추론 시간 (ms/sample)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/image_encoder_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """메인 벤치마크 함수"""
    print("🚀 MMTD 모델 종합 추론 시간 벤치마크")
    print("="*80)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 사용 디바이스: {device}")
    
    # 데이터 로드
    data_path = 'DATA/email_data/EDP.csv'
    pics_path = 'DATA/email_data/pics'
    
    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        return
    
    print(f"📊 데이터 로딩: {data_path}")
    data_df = pd.read_csv(data_path)
    data_df.fillna('', inplace=True)
    
    # 전체 데이터셋 사용 (추론 시간 측정용)
    test_dataset = EmailDataset(pics_path, data_df)
    print(f"📈 전체 데이터셋 크기: {len(test_dataset)} 샘플")
    
    # 실험 설정 가져오기
    experiment_configs = get_experiment_configs()
    
    # 사용 가능한 모델 필터링
    available_models = {}
    print(f"\n🔍 사용 가능한 모델 체크포인트 확인:")
    print("-" * 60)
    
    for exp_name, config in experiment_configs.items():
        if os.path.exists(config["checkpoint_path"]):
            available_models[exp_name] = config
            print(f"✅ {exp_name}: {config['checkpoint_path']}")
        else:
            print(f"❌ {exp_name}: {config['checkpoint_path']} (파일 없음)")
    
    if not available_models:
        print(f"\n❌ 사용 가능한 모델 체크포인트가 없습니다.")
        return
    
    print(f"\n📋 벤치마크 대상 모델: {len(available_models)}개")
    
    # 결과 저장용
    results = {}
    batch_size = 8  # 안정적인 추론을 위한 작은 배치 크기
    
    # 각 모델에 대해 추론 시간 측정
    for exp_name, config in available_models.items():
        print(f"\n{'='*60}")
        print(f"🎯 모델 테스트: {exp_name}")
        print(f"{'='*60}")
        
        try:
            # 모델 로드
            model = load_model_with_checkpoint(config, device)
            if model is None:
                results[exp_name] = {
                    'inference_times': None,
                    'error': 'Model loading failed'
                }
                continue
            
            # 모델 크기 정보
            size_info = get_model_size_info(model)
            print(f"📏 모델 파라미터: {size_info['total_parameters']:,}")
            print(f"💾 모델 크기: {size_info['total_size_mb']:.2f} MB")
            
            # 콜레이터 생성
            collator = DynamicCollator(
                config["text_encoder_name"], 
                config["image_encoder_name"]
            )
            
            # 데이터로더 생성
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
            inference_times = measure_inference_time(
                model, dataloader, device, exp_name, num_warmup=5
            )
            
            if inference_times is not None:
                # 통계 계산
                results[exp_name] = {
                    'inference_times': inference_times.tolist(),
                    'mean_time': np.mean(inference_times),
                    'std_time': np.std(inference_times),
                    'min_time': np.min(inference_times),
                    'max_time': np.max(inference_times),
                    'median_time': np.median(inference_times),
                    'model_size_mb': size_info['total_size_mb'],
                    'total_parameters': size_info['total_parameters'],
                    'samples_processed': len(inference_times)
                }
                
                print(f"📊 결과:")
                print(f"   평균 추론 시간: {results[exp_name]['mean_time']:.3f} ± {results[exp_name]['std_time']:.3f} ms/sample")
                print(f"   최소/최대: {results[exp_name]['min_time']:.3f} / {results[exp_name]['max_time']:.3f} ms")
                print(f"   중앙값: {results[exp_name]['median_time']:.3f} ms")
                print(f"   처리된 샘플: {results[exp_name]['samples_processed']}")
            else:
                results[exp_name] = {
                    'inference_times': None,
                    'error': 'Inference measurement failed'
                }
            
            # 메모리 정리
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"❌ {exp_name} 테스트 중 오류 발생: {str(e)}")
            results[exp_name] = {
                'inference_times': None,
                'error': str(e)
            }
            continue
    
    # 결과 요약 및 출력
    print(f"\n{'='*80}")
    print(f"📊 종합 벤치마크 결과 요약")
    print(f"{'='*80}")
    
    # 성공적으로 측정된 모델들만 필터링
    successful_results = {k: v for k, v in results.items() if v['inference_times'] is not None}
    
    if not successful_results:
        print(f"❌ 성공적으로 측정된 모델이 없습니다.")
        return
    
    # 테이블 형태로 결과 출력
    print(f"{'모델':<25} {'평균시간(ms)':<12} {'표준편차':<10} {'크기(MB)':<10} {'샘플수':<8}")
    print("-" * 75)
    
    # 평균 시간 기준으로 정렬
    sorted_results = sorted(successful_results.items(), key=lambda x: x[1]['mean_time'])
    
    for model_name, result in sorted_results:
        print(f"{model_name:<25} {result['mean_time']:<12.3f} {result['std_time']:<10.3f} "
              f"{result['model_size_mb']:<10.1f} {result['samples_processed']:<8}")
    
    # 추가 통계
    print(f"\n📈 추가 통계:")
    print(f"   가장 빠른 모델: {sorted_results[0][0]} ({sorted_results[0][1]['mean_time']:.3f} ms)")
    print(f"   가장 느린 모델: {sorted_results[-1][0]} ({sorted_results[-1][1]['mean_time']:.3f} ms)")
    
    # 효율성 분석 (성능/크기 비율)
    efficiency_scores = []
    for model_name, result in successful_results.items():
        # 낮은 추론 시간과 작은 모델 크기가 좋으므로 역수 사용
        efficiency = 1000 / (result['mean_time'] * result['model_size_mb'])
        efficiency_scores.append((model_name, efficiency))
    
    efficiency_scores.sort(key=lambda x: x[1], reverse=True)
    print(f"   가장 효율적인 모델: {efficiency_scores[0][0]} (효율성: {efficiency_scores[0][1]:.6f})")
    
    # 결과 저장
    save_path = 'inference_benchmark_results'
    os.makedirs(save_path, exist_ok=True)
    
    # JSON으로 상세 결과 저장
    with open(f'{save_path}/comprehensive_benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'experiment_config': {
                'device': str(device),
                'batch_size': batch_size,
                'total_samples': len(test_dataset),
                'num_warmup_runs': 5,
                'successful_models': len(successful_results),
                'total_models_tested': len(available_models)
            }
        }, f, indent=2, ensure_ascii=False)
    
    # CSV로 요약 결과 저장
    summary_data = []
    for model_name, result in successful_results.items():
        summary_data.append({
            'Model': model_name,
            'Mean_Inference_Time_ms': result['mean_time'],
            'Std_Inference_Time_ms': result['std_time'],
            'Min_Inference_Time_ms': result['min_time'],
            'Max_Inference_Time_ms': result['max_time'],
            'Median_Inference_Time_ms': result['median_time'],
            'Model_Size_MB': result['model_size_mb'],
            'Total_Parameters': result['total_parameters'],
            'Samples_Processed': result['samples_processed']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{save_path}/benchmark_summary.csv', index=False)
    
    # 시각화
    plot_comprehensive_results(successful_results, save_path)
    
    print(f"\n💾 결과 저장 완료:")
    print(f"   📁 폴더: {save_path}/")
    print(f"   📄 상세 결과: comprehensive_benchmark_results.json")
    print(f"   📊 요약 CSV: benchmark_summary.csv")
    print(f"   📈 시각화: *.png 파일들")
    print(f"\n🎉 벤치마크 완료! 총 {len(successful_results)}/{len(available_models)} 모델 성공")


if __name__ == "__main__":
    main() 
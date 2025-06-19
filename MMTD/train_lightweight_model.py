import os
import torch
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizerFast, ViTFeatureExtractor, get_linear_schedule_with_warmup, DistilBertForSequenceClassification, AutoTokenizer, AutoFeatureExtractor, DeiTForImageClassification, ViTForImageClassification, AutoModelForSequenceClassification, MobileBertForSequenceClassification, MobileViTForImageClassification, MobileBertTokenizer, MobileViTImageProcessor, AutoImageProcessor, AutoModelForImageClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from PIL import Image
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'

# 경량화 모델 임포트
from lightweight_models import LightWeightMMTD, UltraLightMMTD, GeneralizedMMTD
from utils import MobileBertMobileViTCollator
from models import PretrainedMMTD, HybridMMTD, HybridMMTDTextTrainable

# TinyBERT는 transformers에서 'huawei-noah/TinyBERT_General_4L_312D' 등으로 사용
from transformers import DeiTForImageClassification, ViTForImageClassification


class EmailDataset(Dataset):
    """이메일 데이터셋 클래스"""
    def __init__(self, data_path, data_df):
        super(EmailDataset, self).__init__()
        self.data_path = data_path
        self.data = data_df.reset_index(drop=True)

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


class DynamicCollator:
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


class DataSplitter:
    """데이터 분할 클래스"""
    def __init__(self, csv_path, k_fold=5):
        self.data = pd.read_csv(csv_path)
        self.data.fillna('', inplace=True)
        self.k_fold = k_fold
        
        # 데이터를 k개 fold로 분할
        fold_size = len(self.data) // k_fold
        self.folds = []
        
        for i in range(k_fold - 1):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size
            self.folds.append(self.data.iloc[start_idx:end_idx])
        
        # 마지막 fold는 나머지 모든 데이터
        self.folds.append(self.data.iloc[(k_fold - 1) * fold_size:])
    
    def get_fold_data(self, test_fold_idx):
        """특정 fold를 테스트 데이터로, 나머지를 훈련 데이터로 반환"""
        test_data = self.folds[test_fold_idx]
        train_data = pd.concat([self.folds[i] for i in range(self.k_fold) if i != test_fold_idx], 
                              ignore_index=True)
        return train_data, test_data


def evaluate_model(model, dataloader, device):
    """모델 평가"""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            outputs = model(**inputs)
            loss = outputs.loss
            logits = outputs.logits
            
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(inputs['labels'].cpu().numpy())
            total_loss += loss.item()
    
    accuracy = accuracy_score(all_labels, all_predictions)
    avg_loss = total_loss / len(dataloader)
    
    return accuracy, avg_loss, all_predictions, all_labels


def train_model(model, train_dataloader, val_dataloader, device, 
                num_epochs=3, learning_rate=2e-5, save_path='lightweight_checkpoints'):
    """모델 훈련"""
    model.to(device)
    
    # 옵티마이저 설정
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # 스케줄러 설정
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 훈련 기록
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    val_losses = []
    
    best_val_accuracy = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        # 훈련 단계
        model.train()
        total_train_loss = 0
        train_predictions = []
        train_labels = []
        
        train_pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
        for batch in train_pbar:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            outputs = model(**inputs)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
            # 훈련 정확도 계산을 위한 예측값 수집
            predictions = torch.argmax(outputs.logits, dim=-1)
            train_predictions.extend(predictions.cpu().numpy())
            train_labels.extend(inputs['labels'].cpu().numpy())
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # 검증 단계
        val_accuracy, val_loss, _, _ = evaluate_model(model, val_dataloader, device)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        
        # 최고 성능 모델 저장
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            print(f"새로운 최고 성능 모델 저장! (Accuracy: {val_accuracy:.4f})")
    
    return {
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'best_val_accuracy': best_val_accuracy
    }


def plot_training_history(history, save_path='lightweight_checkpoints'):
    """훈련 과정 시각화"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss 그래프
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy 그래프
    ax2.plot(epochs, history['val_accuracy'], 'g-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 실험 config 정의 (기존 경량화 모델 + 사전 훈련된 모델 조합)
experiment_configs = {
    # MobileBert + MobileViT
    "mobilebert_mobilevit": {
        "model_class": GeneralizedMMTD,
        "collator_class": lambda: DynamicCollator("google/mobilebert-uncased", "apple/mobilevit-small"),
        "text_encoder_cls": MobileBertForSequenceClassification,
        "image_encoder_cls": MobileViTForImageClassification,
        "text_encoder_name": "google/mobilebert-uncased",
        "image_encoder_name": "apple/mobilevit-small",
        "checkpoint_path": "outputs/mobilebert_mobilevit/best_model.pth",
        "batch_size": 32,
    },
    # MobileBert + DeiT
    "mobilebert_deit": {
        "model_class": GeneralizedMMTD,
        "collator_class": lambda: DynamicCollator("google/mobilebert-uncased", "facebook/deit-base-patch16-224"),
        "text_encoder_cls": MobileBertForSequenceClassification,
        "image_encoder_cls": AutoModelForImageClassification,
        "text_encoder_name": "google/mobilebert-uncased",
        "image_encoder_name": "facebook/deit-base-patch16-224",
        "checkpoint_path": "outputs/mobilebert_deit/best_model.pth",
        "batch_size": 32,
    },
    # DistilBERT + MobileViT
    "distilbert_mobilevit": {
        "model_class": GeneralizedMMTD,
        "collator_class": lambda: DynamicCollator("distilbert-base-multilingual-cased", "apple/mobilevit-small"),
        "text_encoder_cls": DistilBertForSequenceClassification,
        "image_encoder_cls": MobileViTForImageClassification,
        "text_encoder_name": "distilbert-base-multilingual-cased",
        "image_encoder_name": "apple/mobilevit-small",
        "checkpoint_path": "outputs/distilbert_mobilevit/best_model.pth",
        "batch_size": 32,
    },
    # DistilBERT + DeiT
    "distilbert_deit": {
        "model_class": GeneralizedMMTD,
        "collator_class": lambda: DynamicCollator("distilbert-base-multilingual-cased", "facebook/deit-base-patch16-224"),
        "text_encoder_cls": DistilBertForSequenceClassification,
        "image_encoder_cls": AutoModelForImageClassification,
        "text_encoder_name": "distilbert-base-multilingual-cased",
        "image_encoder_name": "facebook/deit-base-patch16-224",
        "checkpoint_path": "outputs/distilbert_deit/best_model.pth",
        "batch_size": 32,
    },
    # TinyBERT + ViT-Tiny
    "tinybert_vit-tiny": {
        "model_class": GeneralizedMMTD,
        "collator_class": lambda: DynamicCollator("huawei-noah/TinyBERT_General_4L_312D", "WinKawaks/vit-tiny-patch16-224"),
        "text_encoder_cls": AutoModelForSequenceClassification,
        "image_encoder_cls": AutoModelForImageClassification,
        "text_encoder_name": "huawei-noah/TinyBERT_General_4L_312D",
        "image_encoder_name": "WinKawaks/vit-tiny-patch16-224",
        "checkpoint_path": "outputs/tinybert_vit-tiny/best_model.pth",
        "batch_size": 32,
    },
    
    # === 사전 훈련된 BERT+BEIT 기반 조합들 ===
    # BERT + BEIT (사전 훈련된 모델, 추론만 가능)
    "bert_beit_pretrained": {
        "model_class": PretrainedMMTD,
        "collator_class": lambda: DynamicCollator("google-bert/bert-base-uncased", "microsoft/beit-base-patch16-224"),
        "checkpoint_path": "outputs/bert_beit_pretrained/best_model.pth",
        "batch_size": 32,
        "pretrained_checkpoint": "checkpoints/fold5/checkpoint-939/pytorch_model.bin",
        "is_pretrained": True,  # 이미 훈련된 모델임을 표시
    },
    
    # BERT + DeiT (사전 훈련된 BERT + 새로운 DeiT)
    "bert_deit": {
        "model_class": HybridMMTD,
        "collator_class": lambda: DynamicCollator("google-bert/bert-base-uncased", "facebook/deit-base-patch16-224"),
        "image_encoder_cls": AutoModelForImageClassification,
        "image_encoder_name": "facebook/deit-base-patch16-224",
        "checkpoint_path": "outputs/bert_deit/best_model.pth",
        "batch_size": 32,
        "pretrained_checkpoint": "checkpoints/fold5/checkpoint-939/pytorch_model.bin",
    },
    
    # BERT + MobileViT (사전 훈련된 BERT + 새로운 MobileViT)
    "bert_mobilevit": {
        "model_class": HybridMMTD,
        "collator_class": lambda: DynamicCollator("google-bert/bert-base-uncased", "apple/mobilevit-small"),
        "image_encoder_cls": MobileViTForImageClassification,
        "image_encoder_name": "apple/mobilevit-small",
        "checkpoint_path": "outputs/bert_mobilevit/best_model.pth",
        "batch_size": 32,
        "pretrained_checkpoint": "checkpoints/fold5/checkpoint-939/pytorch_model.bin",
    },
    
    # BERT + ViT-Tiny (사전 훈련된 BERT + 새로운 ViT-Tiny)
    "bert_vit-tiny": {
        "model_class": HybridMMTD,
        "collator_class": lambda: DynamicCollator("google-bert/bert-base-uncased", "WinKawaks/vit-tiny-patch16-224"),
        "image_encoder_cls": AutoModelForImageClassification,
        "image_encoder_name": "WinKawaks/vit-tiny-patch16-224",
        "checkpoint_path": "outputs/bert_vit-tiny/best_model.pth",
        "batch_size": 32,
        "pretrained_checkpoint": "checkpoints/fold5/checkpoint-939/pytorch_model.bin",
    },
    
    # === BEiT 고정 + 텍스트 인코더 학습 조합들 ===
    # MobileBERT + BEiT (사전 훈련된 BEiT + 새로운 MobileBERT)
    "mobilebert_beit": {
        "model_class": HybridMMTDTextTrainable,
        "collator_class": lambda: DynamicCollator("google/mobilebert-uncased", "microsoft/beit-base-patch16-224"),
        "text_encoder_cls": MobileBertForSequenceClassification,
        "text_encoder_name": "google/mobilebert-uncased",
        "checkpoint_path": "outputs/mobilebert_beit/best_model.pth",
        "batch_size": 32,
        "pretrained_checkpoint": "checkpoints/fold5/checkpoint-939/pytorch_model.bin",
    },
    
    # DistilBERT + BEiT (사전 훈련된 BEiT + 새로운 DistilBERT)
    "distilbert_beit": {
        "model_class": HybridMMTDTextTrainable,
        "collator_class": lambda: DynamicCollator("distilbert-base-multilingual-cased", "microsoft/beit-base-patch16-224"),
        "text_encoder_cls": DistilBertForSequenceClassification,
        "text_encoder_name": "distilbert-base-multilingual-cased",
        "checkpoint_path": "outputs/distilbert_beit/best_model.pth",
        "batch_size": 32,
        "pretrained_checkpoint": "checkpoints/fold5/checkpoint-939/pytorch_model.bin",
    },
    
    # TinyBERT + BEiT (사전 훈련된 BEiT + 새로운 TinyBERT)
    "tinybert_beit": {
        "model_class": HybridMMTDTextTrainable,
        "collator_class": lambda: DynamicCollator("huawei-noah/TinyBERT_General_4L_312D", "microsoft/beit-base-patch16-224"),
        "text_encoder_cls": AutoModelForSequenceClassification,
        "text_encoder_name": "huawei-noah/TinyBERT_General_4L_312D",
        "checkpoint_path": "outputs/tinybert_beit/best_model.pth",
        "batch_size": 32,
        "pretrained_checkpoint": "checkpoints/fold5/checkpoint-939/pytorch_model.bin",
    },
}


def train_single_fold(fold_num, experiment=None, model_type='lightweight', 
                     data_path='DATA/email_data/EDP.csv', 
                     pics_path='DATA/email_data/pics',
                     num_epochs=10, batch_size=16, learning_rate=2e-5):
    """단일 fold에 대해 모델을 훈련합니다."""
    import time
    from datetime import datetime
    
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"🚀 실험: {experiment} | Fold {fold_num} 훈련 시작")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 사용 디바이스: {device}")
    
    # 데이터 준비
    splitter = DataSplitter(data_path, k_fold=5)
    train_data, val_data = splitter.get_fold_data(fold_num - 1)
    print(f"📊 데이터 분할 - 훈련: {len(train_data)}, 검증: {len(val_data)}")
    
    train_dataset = EmailDataset(pics_path, train_data)
    val_dataset = EmailDataset(pics_path, val_data)

    # config 기반 분기
    if experiment is not None and experiment in experiment_configs:
        config = experiment_configs[experiment]
        collator = config["collator_class"]()
        batch_size = config.get("batch_size", batch_size)
        
        print(f"🔧 모델 구성: {config['model_class'].__name__}")
        
        # 모델 생성 분기
        if config["model_class"] == PretrainedMMTD:
            # 사전 훈련된 BERT+BEIT 모델
            model = config["model_class"](
                checkpoint_path=config["pretrained_checkpoint"],
                device=device
            )
            # 사전 훈련된 모델은 학습하지 않음
            if config.get("is_pretrained", False):
                print("⚠️  사전 훈련된 모델입니다. 평가만 수행합니다.")
                # 평가만 수행하고 리턴하는 로직을 여기에 추가할 수 있음
        elif config["model_class"] == HybridMMTD:
            # 하이브리드 모델 (사전 훈련된 BERT + 새로운 이미지 인코더)
            model = config["model_class"](
                pretrained_checkpoint_path=config["pretrained_checkpoint"],
                image_encoder_cls=config["image_encoder_cls"],
                image_pretrain_weight=config["image_encoder_name"],
                device=device
            )
        elif config["model_class"] == HybridMMTDTextTrainable:
            # 하이브리드 모델 (사전 훈련된 BEiT + 새로운 텍스트 인코더)
            model = config["model_class"](
                pretrained_checkpoint_path=config["pretrained_checkpoint"],
                text_encoder_cls=config["text_encoder_cls"],
                text_pretrain_weight=config["text_encoder_name"],
                device=device
            )
        else:
            # GeneralizedMMTD 모델
            model = config["model_class"](
                text_encoder_cls=config["text_encoder_cls"],
                image_encoder_cls=config["image_encoder_cls"],
                text_pretrain_weight=config["text_encoder_name"],
                image_pretrain_weight=config["image_encoder_name"]
            )
        
        model_name = f"{experiment}"
        save_path = os.path.dirname(config["checkpoint_path"])
        checkpoint_path = config["checkpoint_path"]
    else:
        raise ValueError(f"❌ 유효하지 않은 실험: {experiment}")

    # 모델 정보 출력
    model_size_mb = 0
    if hasattr(model, 'get_model_size'):
        size_info = model.get_model_size()
        model_size_mb = size_info['total_size_mb']
        print(f"📏 모델 파라미터: {size_info['total_parameters']:,}")
        print(f"💾 모델 크기: {model_size_mb:.2f} MB")
    
    # 데이터로더 생성
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )
    
    print(f"⚙️  하이퍼파라미터: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
    print(f"💿 저장 경로: {save_path}")
    
    # 훈련 시작
    os.makedirs(save_path, exist_ok=True)
    print(f"\n🎯 훈련 시작...")
    
    history = train_model(
        model, train_dataloader, val_dataloader, device,
        num_epochs=num_epochs, learning_rate=learning_rate, save_path=save_path
    )
    
    # 훈련 히스토리 시각화
    plot_training_history(history, save_path)
    
    # 모델 저장
    torch.save(model.state_dict(), checkpoint_path)
    print(f"💾 체크포인트 저장: {checkpoint_path}")
    
    # 최종 평가
    print(f"\n📊 최종 평가 중...")
    model.load_state_dict(torch.load(checkpoint_path))
    final_accuracy, final_loss, predictions, labels = evaluate_model(model, val_dataloader, device)
    
    # 메트릭 계산
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    
    # 상세 분류 리포트
    report = classification_report(labels, predictions, target_names=['ham', 'spam'], output_dict=True)
    
    # 훈련 시간 계산
    training_time = int(time.time() - start_time)
    
    # 결과를 기존 JSON 형식에 맞춰 생성
    results = {
        'model_name': experiment.replace('_', ' ').title().replace(' ', ' + '),
        'fold': fold_num,
        'final_accuracy': round(final_accuracy, 4),
        'final_loss': round(final_loss, 4),
        'best_accuracy': round(max(history['val_accuracy']), 4),
        'best_loss': round(min(history['val_loss']), 4),
        'epochs_trained': num_epochs,
        'training_time_seconds': training_time,
        'model_size_mb': model_size_mb,
        'parameters': {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'weight_decay': 0.01,
            'warmup_steps': int(0.1 * len(train_dataloader) * num_epochs)
        },
        'history': {
            'train_loss': [round(loss, 4) for loss in history['train_loss']],
            'train_accuracy': [round(acc, 4) for acc in history['train_accuracy']],
            'val_loss': [round(loss, 4) for loss in history['val_loss']],
            'val_accuracy': [round(acc, 4) for acc in history['val_accuracy']]
        },
        'metrics': {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4)
        },
        'classification_report': report,
        'timestamp': datetime.now().isoformat()
    }
    
    # JSON으로 결과 저장
    results_file = os.path.join(save_path, f'fold_{fold_num}_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"✅ Fold {fold_num} 훈련 완료!")
    print(f"{'='*60}")
    print(f"🎯 최고 검증 정확도: {history['best_val_accuracy']:.4f}")
    print(f"🎯 최종 테스트 정확도: {final_accuracy:.4f}")
    print(f"📉 최종 테스트 손실: {final_loss:.4f}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🎯 Recall: {recall:.4f}")
    print(f"🎯 F1-Score: {f1:.4f}")
    print(f"🏷️  Ham F1-Score: {report['ham']['f1-score']:.4f}")
    print(f"🏷️  Spam F1-Score: {report['spam']['f1-score']:.4f}")
    print(f"⏱️  훈련 시간: {training_time}초")
    print(f"💾 결과 저장: {results_file}")
    print(f"{'='*60}")
    
    return results


def main():
    """메인 훈련 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='🤖 MMTD 모델 훈련 스크립트')
    parser.add_argument('--model_type', type=str, default='lightweight', 
                       choices=['lightweight', 'ultralight'],
                       help='훈련할 모델 타입')
    parser.add_argument('--fold', type=int, default=None,
                       help='특정 fold만 훈련 (1-5), None이면 모든 fold 훈련')
    parser.add_argument('--epochs', type=int, default=10,
                       help='훈련 에포크 수')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='학습률')
    parser.add_argument('--experiment', type=str, default=None, 
                       help='실험 config 이름 (예: mobilebert_mobilevit)')
    
    args = parser.parse_args()
    
    # 실험 이름 검증
    if args.experiment and args.experiment not in experiment_configs:
        print(f"❌ 유효하지 않은 실험 이름: {args.experiment}")
        print(f"📋 사용 가능한 실험들:")
        for exp_name in experiment_configs.keys():
            print(f"   - {exp_name}")
        return
    
    print(f"\n🚀 MMTD 모델 훈련 시작")
    print(f"📋 실험: {args.experiment}")
    print(f"🎯 모델 타입: {args.model_type}")
    print(f"📊 에포크: {args.epochs}")
    print(f"📦 배치 크기: {args.batch_size}")
    print(f"📈 학습률: {args.learning_rate}")
    
    if args.fold is not None:
        print(f"🔢 단일 Fold: {args.fold}")
        if args.fold < 1 or args.fold > 5:
            print(f"❌ 유효하지 않은 fold 번호: {args.fold} (1-5 사이여야 함)")
            return
            
        result = train_single_fold(
            args.fold, experiment=args.experiment, model_type=args.model_type,
            num_epochs=args.epochs, batch_size=args.batch_size, 
            learning_rate=args.learning_rate
        )
        
        if result:
            print(f"\n🎉 단일 Fold {args.fold} 훈련 성공적으로 완료!")
        else:
            print(f"\n❌ Fold {args.fold} 훈련 실패")
    else:
        print(f"🔢 전체 Folds: 1-5")
        all_results = []
        
        for fold in range(1, 6):
            try:
                result = train_single_fold(
                    fold, experiment=args.experiment, model_type=args.model_type,
                    num_epochs=args.epochs, batch_size=args.batch_size, 
                    learning_rate=args.learning_rate
                )
                if result:
                    all_results.append(result)
                    
                # 메모리 정리
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"❌ Fold {fold} 훈련 중 오류 발생: {str(e)}")
                continue
        
        # 전체 결과 요약
        if all_results:
            print(f"\n{'='*80}")
            print(f"📊 전체 실험 결과 요약 ({args.experiment})")
            print(f"{'='*80}")
            
            accuracies = [r['best_accuracy'] for r in all_results]
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            
            print(f"📈 평균 검증 정확도: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
            print(f"🎯 최고 검증 정확도: {max(accuracies):.4f}")
            print(f"📉 최저 검증 정확도: {min(accuracies):.4f}")
            
            print(f"\n{'Fold':<6} {'Val Accuracy':<15} {'Test Accuracy':<15} {'Ham F1':<10} {'Spam F1':<10}")
            print("-" * 70)
            
            for result in all_results:
                fold = result['fold']
                val_acc = result['best_accuracy']
                test_acc = result['final_accuracy']
                ham_f1 = result['classification_report']['ham']['f1-score']
                spam_f1 = result['classification_report']['spam']['f1-score']
                
                print(f"{fold:<6} {val_acc:<15.4f} {test_acc:<15.4f} {ham_f1:<10.4f} {spam_f1:<10.4f}")
            
            print(f"\n🎉 전체 {len(all_results)}/5 Folds 훈련 완료!")
        else:
            print(f"\n❌ 훈련 가능한 fold가 없습니다.")


if __name__ == "__main__":
    main() 
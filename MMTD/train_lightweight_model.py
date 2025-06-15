import os
import torch
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizerFast, ViTFeatureExtractor, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 경량화 모델 임포트
from lightweight_models import LightWeightMMTD, UltraLightMMTD


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


class LightweightCollator:
    """경량화 모델용 데이터 콜레이터"""
    def __init__(self):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-multilingual-cased')
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

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
    val_accuracies = []
    val_losses = []
    
    best_val_accuracy = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        # 훈련 단계
        model.train()
        total_train_loss = 0
        
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
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # 검증 단계
        val_accuracy, val_loss, _, _ = evaluate_model(model, val_dataloader, device)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        
        # 최고 성능 모델 저장
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            print(f"새로운 최고 성능 모델 저장! (Accuracy: {val_accuracy:.4f})")
    
    return {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'val_losses': val_losses,
        'best_val_accuracy': best_val_accuracy
    }


def plot_training_history(history, save_path='lightweight_checkpoints'):
    """훈련 과정 시각화"""
    epochs = range(1, len(history['train_losses']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss 그래프
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy 그래프
    ax2.plot(epochs, history['val_accuracies'], 'g-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


def train_single_fold(fold_num, model_type='lightweight', 
                     data_path='DATA/email_data/EDP.csv', 
                     pics_path='DATA/email_data/pics',
                     num_epochs=3, batch_size=16, learning_rate=2e-5):
    """단일 fold에 대해 모델을 훈련합니다."""
    print(f"\n{'='*20} Fold {fold_num} 훈련 {'='*20}")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 데이터 분할
    splitter = DataSplitter(data_path, k_fold=5)
    train_data, val_data = splitter.get_fold_data(fold_num - 1)
    
    print(f"훈련 데이터: {len(train_data)}, 검증 데이터: {len(val_data)}")
    
    # 데이터셋 생성
    train_dataset = EmailDataset(pics_path, train_data)
    val_dataset = EmailDataset(pics_path, val_data)
    
    # 콜레이터 생성
    collator = LightweightCollator()
    
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
    
    # 모델 초기화
    if model_type == 'lightweight':
        model = LightWeightMMTD()
        model_name = 'LightWeightMMTD'
    elif model_type == 'ultralight':
        model = UltraLightMMTD()
        model_name = 'UltraLightMMTD'
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    print(f"모델: {model_name}")
    
    # 모델 크기 정보
    if hasattr(model, 'get_model_size'):
        size_info = model.get_model_size()
        print(f"모델 파라미터: {size_info['total_parameters']:,}")
        print(f"모델 크기: {size_info['total_size_mb']:.2f} MB")
    
    # 저장 경로 설정
    save_path = f'lightweight_checkpoints/{model_name.lower()}_fold{fold_num}'
    
    try:
        # 모델 훈련
        history = train_model(
            model, train_dataloader, val_dataloader, device,
            num_epochs=num_epochs, learning_rate=learning_rate, save_path=save_path
        )
        
        # 훈련 과정 시각화
        plot_training_history(history, save_path)
        
        # 최종 평가
        model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pth')))
        final_accuracy, final_loss, predictions, labels = evaluate_model(model, val_dataloader, device)
        
        # 상세 분류 리포트
        report = classification_report(labels, predictions, target_names=['ham', 'spam'], output_dict=True)
        
        # 결과 저장
        results = {
            'fold': fold_num,
            'model_type': model_type,
            'model_name': model_name,
            'final_accuracy': final_accuracy,
            'final_loss': final_loss,
            'best_val_accuracy': history['best_val_accuracy'],
            'classification_report': report,
            'training_history': history,
            'hyperparameters': {
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }
        }
        
        # JSON으로 결과 저장
        with open(os.path.join(save_path, 'training_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nFold {fold_num} 훈련 완료!")
        print(f"최고 검증 정확도: {history['best_val_accuracy']:.4f}")
        print(f"최종 테스트 정확도: {final_accuracy:.4f}")
        print(f"결과 저장 위치: {save_path}")
        
        return results
        
    except Exception as e:
        print(f"Fold {fold_num} 훈련 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def train_all_folds(model_type='lightweight', num_epochs=3, batch_size=16, learning_rate=2e-5):
    """모든 fold에 대해 모델을 훈련합니다."""
    print(f"경량화 모델 ({model_type}) 전체 fold 훈련 시작")
    print("="*60)
    
    all_results = []
    
    for fold in range(1, 6):
        result = train_single_fold(
            fold, model_type=model_type,
            num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate
        )
        
        if result:
            all_results.append(result)
            
            # 메모리 정리
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if not all_results:
        print("훈련 가능한 fold가 없습니다.")
        return
    
    # 전체 결과 요약
    print("\n" + "="*60)
    print("전체 훈련 결과 요약")
    print("="*60)
    
    accuracies = [r['best_val_accuracy'] for r in all_results]
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    print(f"평균 검증 정확도: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"최고 검증 정확도: {max(accuracies):.4f}")
    print(f"최저 검증 정확도: {min(accuracies):.4f}")
    
    # Fold별 상세 결과
    print(f"\n{'Fold':<6} {'Val Accuracy':<15} {'Test Accuracy':<15} {'Ham F1':<10} {'Spam F1':<10}")
    print("-" * 60)
    
    for result in all_results:
        fold = result['fold']
        val_acc = result['best_val_accuracy']
        test_acc = result['final_accuracy']
        ham_f1 = result['classification_report']['ham']['f1-score']
        spam_f1 = result['classification_report']['spam']['f1-score']
        
        print(f"{fold:<6} {val_acc:<15.4f} {test_acc:<15.4f} {ham_f1:<10.4f} {spam_f1:<10.4f}")
    
    # 전체 결과 저장
    summary_results = {
        'model_type': model_type,
        'individual_results': all_results,
        'summary_statistics': {
            'mean_val_accuracy': mean_accuracy,
            'std_val_accuracy': std_accuracy,
            'min_val_accuracy': min(accuracies),
            'max_val_accuracy': max(accuracies)
        }
    }
    
    summary_path = f'lightweight_checkpoints/{model_type}_summary'
    os.makedirs(summary_path, exist_ok=True)
    
    with open(os.path.join(summary_path, 'all_folds_results.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n전체 훈련 완료! 요약 결과가 '{summary_path}'에 저장되었습니다.")


def main():
    """메인 훈련 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='경량화 MMTD 모델 훈련')
    parser.add_argument('--model_type', type=str, default='lightweight', 
                       choices=['lightweight', 'ultralight'],
                       help='훈련할 모델 타입')
    parser.add_argument('--fold', type=int, default=None,
                       help='특정 fold만 훈련 (1-5), None이면 모든 fold 훈련')
    parser.add_argument('--epochs', type=int, default=3,
                       help='훈련 에포크 수')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='학습률')
    
    args = parser.parse_args()
    
    if args.fold is not None:
        # 특정 fold만 훈련
        train_single_fold(
            args.fold, model_type=args.model_type,
            num_epochs=args.epochs, batch_size=args.batch_size, 
            learning_rate=args.learning_rate
        )
    else:
        # 모든 fold 훈련
        train_all_folds(
            model_type=args.model_type,
            num_epochs=args.epochs, batch_size=args.batch_size, 
            learning_rate=args.learning_rate
        )


if __name__ == "__main__":
    main() 
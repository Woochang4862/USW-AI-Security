import os
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizerFast, BeitFeatureExtractor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 실제 MMTD 모델 클래스 (models.py에서 가져옴)
from models import MMTD

# 간단한 데이터셋 클래스
class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, data_df):
        super(EvalDataset, self).__init__()
        self.data_path = data_path
        self.data = data_df.reset_index(drop=True)

    def __getitem__(self, item):
        text = str(self.data.iloc[item, 0]) if pd.notna(self.data.iloc[item, 0]) else ""
        pic_path = os.path.join(self.data_path, self.data.iloc[item, 1])
        label = self.data.iloc[item, 2]
        try:
            pic = Image.open(pic_path).convert('RGB')
        except Exception as e:
            print(f"이미지 로드 실패: {pic_path}, 오류: {e}")
            pic = Image.new('RGB', (224, 224), color='white')
        return text, pic, label

    def __len__(self):
        return len(self.data)

# 콜레이터 클래스
class EvalCollator:
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

# 데이터 분할 클래스
class DataSplitter:
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

def load_model_checkpoint(checkpoint_path, device='cpu'):
    """체크포인트에서 MMTD 모델을 로드합니다."""
    print(f"체크포인트 로딩: {checkpoint_path}")
    
    # 모델 초기화
    model = MMTD()
    
    # 체크포인트 로드
    model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("모델 로딩 완료")
    return model

def evaluate_model(model, dataloader, device='cpu'):
    """모델을 평가하고 예측 결과를 반환합니다."""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    print("모델 평가 중...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 배치를 디바이스로 이동
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # 모델 추론
            outputs = model(**inputs)
            logits = outputs.logits
            
            # 예측 및 확률 계산
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(inputs['labels'].cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels)

def calculate_metrics(y_true, y_pred, y_prob):
    """평가 메트릭을 계산합니다."""
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['ham', 'spam'], output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # AUC 계산
    try:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    except:
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision_ham': report['ham']['precision'],
        'recall_ham': report['ham']['recall'],
        'f1_ham': report['ham']['f1-score'],
        'precision_spam': report['spam']['precision'],
        'recall_spam': report['spam']['recall'],
        'f1_spam': report['spam']['f1-score'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'auc': auc,
        'confusion_matrix': cm.tolist()
    }

def evaluate_single_fold(fold_num, data_path='DATA/email_data/EDP.csv', pics_path='DATA/email_data/pics'):
    """단일 fold에 대해 모델을 평가합니다."""
    print(f"\n{'='*20} Fold {fold_num} 평가 {'='*20}")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 데이터 분할
    splitter = DataSplitter(data_path, k_fold=5)
    train_data, test_data = splitter.get_fold_data(fold_num - 1)  # fold_num은 1부터 시작
    
    print(f"훈련 데이터: {len(train_data)}, 테스트 데이터: {len(test_data)}")
    
    # 체크포인트 경로
    checkpoint_path = f'checkpoints/fold{fold_num}/checkpoint-939'
    if not os.path.exists(checkpoint_path):
        print(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        return None
    
    try:
        # 모델 로드
        model = load_model_checkpoint(checkpoint_path, device)
        
        # 테스트 데이터셋 생성
        test_dataset = EvalDataset(pics_path, test_data)
        collator = EvalCollator()
        
        # 데이터로더 생성 (배치 크기를 작게 설정)
        dataloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=8,  # 메모리 사용량을 줄이기 위해 작은 배치 크기 사용
            shuffle=False, 
            collate_fn=collator,
            num_workers=0
        )
        
        # 모델 평가
        y_pred, y_prob, y_true = evaluate_model(model, dataloader, device)
        
        # 메트릭 계산
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        metrics['fold'] = fold_num
        
        # 결과 출력
        print(f"\nFold {fold_num} 결과:")
        print(f"  정확도: {metrics['accuracy']:.4f}")
        print(f"  매크로 F1: {metrics['macro_f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Ham - Precision: {metrics['precision_ham']:.4f}, Recall: {metrics['recall_ham']:.4f}, F1: {metrics['f1_ham']:.4f}")
        print(f"  Spam - Precision: {metrics['precision_spam']:.4f}, Recall: {metrics['recall_spam']:.4f}, F1: {metrics['f1_spam']:.4f}")
        
        # 혼동 행렬 시각화
        plot_confusion_matrix(np.array(metrics['confusion_matrix']), fold_num)
        
        return metrics
        
    except Exception as e:
        print(f"Fold {fold_num} 평가 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def plot_confusion_matrix(cm, fold_num):
    """혼동 행렬을 시각화합니다."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.title(f'Confusion Matrix - Fold {fold_num}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    os.makedirs('evaluation_results', exist_ok=True)
    plt.savefig(f'evaluation_results/confusion_matrix_fold{fold_num}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_summary(results):
    """성능 요약을 시각화합니다."""
    if not results:
        return
        
    folds = [r['fold'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['macro_f1'] for r in results]
    auc_scores = [r['auc'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 정확도
    axes[0, 0].bar(folds, accuracies, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Accuracy by Fold')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(folds[i], v + 0.01, f'{v:.3f}', ha='center')
    
    # F1 점수
    axes[0, 1].bar(folds, f1_scores, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Macro F1 Score by Fold')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_ylim(0, 1)
    for i, v in enumerate(f1_scores):
        axes[0, 1].text(folds[i], v + 0.01, f'{v:.3f}', ha='center')
    
    # AUC 점수
    axes[1, 0].bar(folds, auc_scores, color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('AUC Score by Fold')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('AUC Score')
    axes[1, 0].set_ylim(0, 1)
    for i, v in enumerate(auc_scores):
        axes[1, 0].text(folds[i], v + 0.01, f'{v:.3f}', ha='center')
    
    # 전체 메트릭 비교
    metrics = ['Accuracy', 'Macro F1', 'AUC']
    means = [np.mean(accuracies), np.mean(f1_scores), np.mean(auc_scores)]
    stds = [np.std(accuracies), np.std(f1_scores), np.std(auc_scores)]
    
    bars = axes[1, 1].bar(metrics, means, yerr=stds, capsize=5, 
                         color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.7)
    axes[1, 1].set_title('Average Performance Metrics')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1)
    
    # 평균값 표시
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                        f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('evaluation_results/performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """메인 평가 함수"""
    print("MMTD 모델 성능 평가 시작")
    print("=" * 60)
    
    # 결과 저장 디렉토리 생성
    os.makedirs('evaluation_results', exist_ok=True)
    
    all_results = []
    
    # 각 fold 평가 (메모리 절약을 위해 하나씩 처리)
    for fold in range(1, 6):
        print(f"\n처리 중: Fold {fold}/5")
        try:
            result = evaluate_single_fold(fold)
            if result:
                all_results.append(result)
                
                # 메모리 정리
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        except Exception as e:
            print(f"Fold {fold} 평가 실패: {str(e)}")
            continue
    
    if not all_results:
        print("평가 가능한 fold가 없습니다.")
        return
    
    # 전체 결과 요약
    print("\n" + "=" * 60)
    print("전체 성능 요약")
    print("=" * 60)
    
    # 통계 계산
    metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'auc']
    summary = {}
    
    for metric in metrics:
        values = [r[metric] for r in all_results]
        summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # 결과 출력
    for metric, stats in summary.items():
        print(f"{metric.upper()}:")
        print(f"  평균: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  범위: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print()
    
    # 상세 결과 테이블
    print("Fold별 상세 결과:")
    print("-" * 80)
    print(f"{'Fold':<6} {'Accuracy':<10} {'Macro F1':<10} {'AUC':<10} {'Ham F1':<10} {'Spam F1':<10}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['fold']:<6} {result['accuracy']:<10.4f} {result['macro_f1']:<10.4f} "
              f"{result['auc']:<10.4f} {result['f1_ham']:<10.4f} {result['f1_spam']:<10.4f}")
    
    # 결과 저장
    with open('evaluation_results/detailed_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'individual_results': all_results,
            'summary_statistics': summary
        }, f, indent=2, ensure_ascii=False)
    
    # 시각화
    plot_performance_summary(all_results)
    
    print(f"\n평가 완료! 결과가 'evaluation_results/' 폴더에 저장되었습니다.")
    print(f"총 {len(all_results)}개 fold 평가 완료")

if __name__ == "__main__":
    main() 
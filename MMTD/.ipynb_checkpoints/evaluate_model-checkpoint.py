import os
import torch
import pandas as pd
import numpy as np
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from Email_dataset import EDPDataset, EDPCollator
from models import MMTD
from utils import SplitData, EvalMetrics
import json

def load_model_from_checkpoint(checkpoint_path):
    """체크포인트에서 모델을 로드합니다."""
    print(f"체크포인트에서 모델 로딩: {checkpoint_path}")
    
    # 모델 초기화 (사전 훈련된 가중치 없이)
    model = MMTD()
    
    # 체크포인트에서 모델 가중치 로드
    model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print("모델 가중치 로딩 완료")
    else:
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    return model

def evaluate_single_fold(fold_num, data_path='DATA/email_data/EDP.csv'):
    """단일 fold에 대해 모델을 평가합니다."""
    print(f"\n=== Fold {fold_num} 평가 시작 ===")
    
    # 데이터 로드
    split_data = SplitData(data_path, 5)
    
    # 해당 fold의 테스트 데이터 가져오기
    for i in range(fold_num):
        train_df, test_df = split_data()
    
    print(f"테스트 데이터 크기: {len(test_df)}")
    
    # 데이터셋 생성
    test_dataset = EDPDataset('DATA/email_data/pics', test_df)
    
    # 체크포인트 경로
    checkpoint_path = f'checkpoints/fold{fold_num}/checkpoint-939'
    
    if not os.path.exists(checkpoint_path):
        print(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        return None
    
    # 모델 로드
    model = load_model_from_checkpoint(checkpoint_path)
    model.eval()
    
    # 평가 설정
    args = TrainingArguments(
        output_dir='./temp_eval',
        per_device_eval_batch_size=32,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=test_dataset,
        data_collator=EDPCollator(),
    )
    
    # 평가 실행
    print("모델 평가 중...")
    eval_results = trainer.evaluate()
    
    # 예측 수행
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    
    # 상세 메트릭 계산
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['ham', 'spam'], output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC AUC 계산 (확률 점수 사용)
    y_prob = torch.softmax(torch.from_numpy(predictions.predictions), dim=1).numpy()
    try:
        auc_score = roc_auc_score(y_true, y_prob[:, 1])
    except:
        auc_score = 0.0
    
    results = {
        'fold': fold_num,
        'accuracy': accuracy,
        'precision_ham': report['ham']['precision'],
        'recall_ham': report['ham']['recall'],
        'f1_ham': report['ham']['f1-score'],
        'precision_spam': report['spam']['precision'],
        'recall_spam': report['spam']['recall'],
        'f1_spam': report['spam']['f1-score'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'auc_score': auc_score,
        'confusion_matrix': cm.tolist(),
        'eval_loss': eval_results.get('eval_loss', 0.0)
    }
    
    print(f"Fold {fold_num} 결과:")
    print(f"  정확도: {accuracy:.4f}")
    print(f"  매크로 F1: {report['macro avg']['f1-score']:.4f}")
    print(f"  AUC: {auc_score:.4f}")
    print(f"  평가 손실: {eval_results.get('eval_loss', 0.0):.4f}")
    
    return results

def plot_confusion_matrix(cm, fold_num, save_path='evaluation_results'):
    """혼동 행렬을 시각화합니다."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.title(f'Confusion Matrix - Fold {fold_num}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/confusion_matrix_fold{fold_num}.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_all_folds():
    """모든 fold에 대해 평가를 수행합니다."""
    print("MMTD 모델 성능 평가 시작")
    print("=" * 50)
    
    all_results = []
    
    # 각 fold 평가
    for fold in range(1, 6):
        try:
            results = evaluate_single_fold(fold)
            if results:
                all_results.append(results)
                
                # 혼동 행렬 시각화
                cm = np.array(results['confusion_matrix'])
                plot_confusion_matrix(cm, fold)
                
        except Exception as e:
            print(f"Fold {fold} 평가 중 오류 발생: {str(e)}")
            continue
    
    if not all_results:
        print("평가할 수 있는 fold가 없습니다.")
        return
    
    # 전체 결과 요약
    print("\n" + "=" * 50)
    print("전체 결과 요약")
    print("=" * 50)
    
    metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'auc_score', 'eval_loss']
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
    print(f"{'Fold':<6} {'Accuracy':<10} {'Macro F1':<10} {'AUC':<10} {'Loss':<10}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['fold']:<6} {result['accuracy']:<10.4f} {result['macro_f1']:<10.4f} "
              f"{result['auc_score']:<10.4f} {result['eval_loss']:<10.4f}")
    
    # 결과를 JSON 파일로 저장
    os.makedirs('evaluation_results', exist_ok=True)
    with open('evaluation_results/evaluation_summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'individual_results': all_results,
            'summary_statistics': summary
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n결과가 'evaluation_results/' 폴더에 저장되었습니다.")
    
    # 성능 시각화
    plot_performance_summary(all_results)
    
    return all_results, summary

def plot_performance_summary(results):
    """성능 요약을 시각화합니다."""
    folds = [r['fold'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['macro_f1'] for r in results]
    auc_scores = [r['auc_score'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 정확도
    axes[0, 0].bar(folds, accuracies, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Accuracy by Fold')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    
    # F1 점수
    axes[0, 1].bar(folds, f1_scores, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Macro F1 Score by Fold')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_ylim(0, 1)
    
    # AUC 점수
    axes[1, 0].bar(folds, auc_scores, color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('AUC Score by Fold')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('AUC Score')
    axes[1, 0].set_ylim(0, 1)
    
    # 전체 메트릭 비교
    metrics = ['Accuracy', 'Macro F1', 'AUC']
    means = [np.mean(accuracies), np.mean(f1_scores), np.mean(auc_scores)]
    stds = [np.std(accuracies), np.std(f1_scores), np.std(auc_scores)]
    
    axes[1, 1].bar(metrics, means, yerr=stds, capsize=5, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.7)
    axes[1, 1].set_title('Average Performance Metrics')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('evaluation_results/performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 작업 디렉토리를 MMTD로 변경
    os.chdir('MMTD')
    
    # 평가 실행
    results, summary = evaluate_all_folds() 
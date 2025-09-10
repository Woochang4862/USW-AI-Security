import os
import json
import pandas as pd
from pathlib import Path
import argparse

def extract_fold_data(json_data):
    """JSON 데이터에서 필요한 정보를 추출합니다."""
    row_data = {
        'model_name': json_data.get('model_name', ''),
        'fold': json_data.get('fold', ''),
        'final_accuracy': json_data.get('final_accuracy', ''),
        'final_loss': json_data.get('final_loss', ''),
        'best_accuracy': json_data.get('best_accuracy', ''),
        'best_loss': json_data.get('best_loss', ''),
        'epochs_trained': json_data.get('epochs_trained', ''),
        'training_time_seconds': json_data.get('training_time_seconds', ''),
        'model_size_mb': json_data.get('model_size_mb', ''),
        'timestamp': json_data.get('timestamp', ''),
    }
    
    # Parameters 정보 추가
    params = json_data.get('parameters', {})
    row_data.update({
        'learning_rate': params.get('learning_rate', ''),
        'batch_size': params.get('batch_size', ''),
        'weight_decay': params.get('weight_decay', ''),
        'warmup_steps': params.get('warmup_steps', ''),
    })
    
    # Metrics 정보 추가
    metrics = json_data.get('metrics', {})
    row_data.update({
        'accuracy': metrics.get('accuracy', ''),
        'precision': metrics.get('precision', ''),
        'recall': metrics.get('recall', ''),
        'f1_score': metrics.get('f1_score', ''),
    })
    
    # Classification report 정보 추가
    clf_report = json_data.get('classification_report', {})
    
    # Ham 클래스 정보
    ham_data = clf_report.get('ham', {})
    row_data.update({
        'ham_precision': ham_data.get('precision', ''),
        'ham_recall': ham_data.get('recall', ''),
        'ham_f1_score': ham_data.get('f1-score', ''),
        'ham_support': ham_data.get('support', ''),
    })
    
    # Spam 클래스 정보
    spam_data = clf_report.get('spam', {})
    row_data.update({
        'spam_precision': spam_data.get('precision', ''),
        'spam_recall': spam_data.get('recall', ''),
        'spam_f1_score': spam_data.get('f1-score', ''),
        'spam_support': spam_data.get('support', ''),
    })
    
    # Macro average 정보
    macro_avg = clf_report.get('macro avg', {})
    row_data.update({
        'macro_avg_precision': macro_avg.get('precision', ''),
        'macro_avg_recall': macro_avg.get('recall', ''),
        'macro_avg_f1_score': macro_avg.get('f1-score', ''),
        'macro_avg_support': macro_avg.get('support', ''),
    })
    
    # Weighted average 정보
    weighted_avg = clf_report.get('weighted avg', {})
    row_data.update({
        'weighted_avg_precision': weighted_avg.get('precision', ''),
        'weighted_avg_recall': weighted_avg.get('recall', ''),
        'weighted_avg_f1_score': weighted_avg.get('f1-score', ''),
        'weighted_avg_support': weighted_avg.get('support', ''),
    })
    
    # History 정보 추가 (마지막 epoch 값들)
    history = json_data.get('history', {})
    train_loss = history.get('train_loss', [])
    train_accuracy = history.get('train_accuracy', [])
    val_loss = history.get('val_loss', [])
    val_accuracy = history.get('val_accuracy', [])
    
    row_data.update({
        'final_train_loss': train_loss[-1] if train_loss else '',
        'final_train_accuracy': train_accuracy[-1] if train_accuracy else '',
        'final_val_loss': val_loss[-1] if val_loss else '',
        'final_val_accuracy': val_accuracy[-1] if val_accuracy else '',
    })
    
    return row_data

def process_outputs_folder(outputs_path, output_csv_path):
    """outputs 폴더의 모든 모델 폴더를 처리하여 통합 CSV를 생성합니다."""
    all_data = []
    
    # outputs 폴더 내의 모든 하위 폴더 확인
    for model_folder in os.listdir(outputs_path):
        model_path = os.path.join(outputs_path, model_folder)
        
        # 폴더인지 확인
        if not os.path.isdir(model_path):
            continue
            
        print(f"처리 중: {model_folder}")
        
        # 각 모델 폴더 내의 fold_*_results.json 파일들 찾기
        for file_name in os.listdir(model_path):
            if file_name.startswith('fold_') and file_name.endswith('_results.json'):
                file_path = os.path.join(model_path, file_name)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # 데이터 추출
                    row_data = extract_fold_data(json_data)
                    all_data.append(row_data)
                    
                except Exception as e:
                    print(f"오류 발생 - 파일: {file_path}, 오류: {e}")
                    continue
    
    # DataFrame 생성 및 CSV 저장
    if all_data:
        df = pd.DataFrame(all_data)
        
        # 정렬 (model_name, fold 순으로)
        df = df.sort_values(['model_name', 'fold'])
        
        # CSV 저장
        df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"\n통합 CSV 파일이 생성되었습니다: {output_csv_path}")
        print(f"총 {len(all_data)}개의 레코드가 포함되었습니다.")
        
        # 요약 정보 출력
        print(f"\n포함된 모델들:")
        unique_models = df['model_name'].unique()
        for model in unique_models:
            model_count = len(df[df['model_name'] == model])
            print(f"  - {model}: {model_count}개 fold")
            
    else:
        print("처리할 데이터가 없습니다.")

def main():
    parser = argparse.ArgumentParser(description='outputs 폴더의 모든 fold 결과를 통합한 CSV 파일을 생성합니다.')
    parser.add_argument('--outputs_path', type=str, default='outputs', 
                       help='outputs 폴더 경로 (기본값: outputs)')
    parser.add_argument('--output_csv', type=str, default='unified_results.csv',
                       help='출력 CSV 파일명 (기본값: unified_results.csv)')
    
    args = parser.parse_args()
    
    # 경로 확인
    outputs_path = Path(args.outputs_path)
    if not outputs_path.exists():
        print(f"오류: {outputs_path} 폴더가 존재하지 않습니다.")
        return
    
    # 출력 파일 경로 설정
    output_csv_path = Path(args.output_csv)
    
    print(f"출력 경로: {outputs_path}")
    print(f"CSV 저장 경로: {output_csv_path}")
    print("-" * 50)
    
    # 처리 실행
    process_outputs_folder(outputs_path, output_csv_path)

if __name__ == "__main__":
    main()
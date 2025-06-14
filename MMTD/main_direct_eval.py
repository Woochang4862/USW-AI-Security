from Email_dataset import EDPDataset, EDPCollator
from models import MMTD
from utils import SplitData
import wandb
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
fold = 5
split_data = SplitData('DATA/email_data/EDP.csv', fold)

# 기존 훈련된 MMTD 체크포인트 경로
mmtd_checkpoint_path = './checkpoints'
mmtd_folds = sorted([f for f in os.listdir(mmtd_checkpoint_path) if f.startswith('fold')])

def evaluate_model(model, dataloader, device):
    """직접 모델 평가 함수"""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 배치 데이터를 디바이스로 이동
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # 모델 추론
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )
            
            # 예측값과 실제값 수집
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 손실값 누적
            if outputs.loss is not None:
                total_loss += outputs.loss.item()
            num_batches += 1
    
    # 메트릭 계산
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': avg_loss
    }

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    for i in range(fold):
        print(f"\n{'='*50}")
        print(f"Processing Fold {i + 1}")
        print(f"{'='*50}")
        
        wandb.init(project='MMTD-DirectEval')
        wandb.run.name = f'MMTD-direct-eval-fold-{i + 1}'
        
        # 데이터 준비
        train_df, test_df = split_data()
        train_dataset = EDPDataset('DATA/email_data/pics', train_df)
        test_dataset = EDPDataset('DATA/email_data/pics', test_df)
        
        # 데이터로더 생성
        collator = EDPCollator()
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collator)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collator)
        
        # 체크포인트 로드
        fold_name = mmtd_folds[i]
        checkpoint_dir = os.path.join(mmtd_checkpoint_path, fold_name)
        checkpoint_names = os.listdir(checkpoint_dir)
        
        if checkpoint_names:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_names[0], 'pytorch_model.bin')
            print(f"Loading checkpoint: {checkpoint_path}")
            
            # 모델 생성 및 체크포인트 로드
            model = MMTD()
            model = model.to(device)
            
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # 호환 가능한 키만 로드
                model_state_dict = model.state_dict()
                compatible_state_dict = {}
                
                for key, value in checkpoint.items():
                    if key in model_state_dict and model_state_dict[key].shape == value.shape:
                        compatible_state_dict[key] = value
                
                model.load_state_dict(compatible_state_dict, strict=False)
                print(f"Loaded {len(compatible_state_dict)} compatible parameters")
                
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Using random initialization")
        else:
            print("No checkpoint found, using random initialization")
            model = MMTD()
            model = model.to(device)
        
        # 훈련 데이터 평가
        print("\nEvaluating on training data...")
        train_metrics = evaluate_model(model, train_dataloader, device)
        print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Train F1: {train_metrics['f1']:.4f}")
        
        # 테스트 데이터 평가
        print("\nEvaluating on test data...")
        test_metrics = evaluate_model(model, test_dataloader, device)
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        
        # WandB 로깅
        wandb.log({
            'fold': i + 1,
            'train_accuracy': train_metrics['accuracy'],
            'train_loss': train_metrics['loss'],
            'train_f1': train_metrics['f1'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'test_accuracy': test_metrics['accuracy'],
            'test_loss': test_metrics['loss'],
            'test_f1': test_metrics['f1'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall']
        })
        
        # 결과 저장
        os.makedirs('./output/MMTD_direct_eval', exist_ok=True)
        results = {
            'fold': i + 1,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        
        import json
        with open(f'./output/MMTD_direct_eval/fold_{i+1}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        wandb.finish()
        print(f"Completed fold {i + 1}")

print("\n" + "="*50)
print("All folds completed!")
print("="*50) 
from Email_dataset import EDPDataset, EDPCollator
from models import MMTD
from transformers import Trainer, TrainingArguments
from torch.optim import AdamW, lr_scheduler
from utils import metrics, SplitData, save_config, EvalMetrics
import wandb
import os
import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
fold = 5
split_data = SplitData('DATA/email_data/EDP.csv', fold)

# 기존 훈련된 MMTD 체크포인트 경로
mmtd_checkpoint_path = './checkpoints'
mmtd_folds = os.listdir(mmtd_checkpoint_path)
mmtd_checkpoints = list()
for fold_dir in mmtd_folds:
    if fold_dir.startswith('fold'):
        checkpoint_dir = os.path.join(mmtd_checkpoint_path, fold_dir)
        if os.path.isdir(checkpoint_dir):
            checkpoints_in_fold = os.listdir(checkpoint_dir)
            mmtd_checkpoints.append(checkpoints_in_fold[0] if checkpoints_in_fold else None)

if __name__ == '__main__':
    for i in range(fold):
        wandb.init(project='MMTD-Evaluation')
        wandb.run.name = 'MMTD-eval-fold-' + str(i + 1)
        train_df, test_df = split_data()
        train_dataset = EDPDataset('DATA/email_data/pics', train_df)
        test_dataset = EDPDataset('DATA/email_data/pics', test_df)

        # 기존 훈련된 MMTD 모델 로드
        fold_name = f'fold{i + 1}'
        if mmtd_checkpoints[i] is not None:
            mmtd_checkpoint = os.path.join(mmtd_checkpoint_path, fold_name, mmtd_checkpoints[i])
            print(f"Loading MMTD checkpoint from: {mmtd_checkpoint}")
            
            # 새로운 MMTD 모델 생성 (사전 훈련된 가중치 없이)
            model = MMTD()
            
            # 훈련된 체크포인트 로드
            checkpoint = torch.load(os.path.join(mmtd_checkpoint, 'pytorch_model.bin'), map_location='cpu')
            model.load_state_dict(checkpoint)
            
            print(f"Successfully loaded checkpoint for fold {i + 1}")
        else:
            print(f"No checkpoint found for fold {i + 1}, creating new model")
            model = MMTD()

        # 모든 파라미터를 평가 모드로 설정 (학습하지 않음)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # 빈 파라미터 리스트 (평가만 수행)
        filtered_parameters = []

        # 옵티마이저는 빈 파라미터로 생성 (실제로는 사용되지 않음)
        optimizer = AdamW([torch.tensor(0.0, requires_grad=True)], lr=5e-4)

        args = TrainingArguments(
            output_dir='./output/MMTD_eval/results/fold' + str(i + 1),
            logging_dir='./output/MMTD_eval/log',
            logging_strategy='no',  # 평가만 수행하므로 로깅 최소화
            learning_rate=5e-4,
            per_device_train_batch_size=40,
            per_device_eval_batch_size=40,
            num_train_epochs=1,  # 평가만 수행
            weight_decay=0.0,
            save_strategy="no",  # 저장하지 않음
            evaluation_strategy="no",  # 수동으로 평가
            load_best_model_at_end=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            run_name=wandb.run.name,
            auto_find_batch_size=False,
            overwrite_output_dir=True,
            remove_unused_columns=False,
            report_to=["wandb"],
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            optimizers=(optimizer, None),
            data_collator=EDPCollator(),
            compute_metrics=metrics,
        )

        # 훈련 데이터에 대한 평가
        print(f"Evaluating on training data for fold {i + 1}...")
        train_acc = trainer.evaluate(eval_dataset=train_dataset)
        train_result = {'train_acc': train_acc['eval_acc'], 'train_loss': train_acc['eval_loss']}
        wandb.log(train_result)
        print(f"Train accuracy: {train_acc['eval_acc']:.4f}, Train loss: {train_acc['eval_loss']:.4f}")

        # 테스트 데이터에 대한 평가
        print(f"Evaluating on test data for fold {i + 1}...")
        trainer.compute_metrics = EvalMetrics('./output/MMTD_eval/results', wandb.run.name, True)
        test_acc = trainer.evaluate(eval_dataset=test_dataset)
        test_result = {'test_acc': test_acc['eval_acc'], 'test_loss': test_acc['eval_loss']}
        wandb.log(test_result)
        print(f"Test accuracy: {test_acc['eval_acc']:.4f}, Test loss: {test_acc['eval_loss']:.4f}")

        wandb.config = args.to_dict()
        
        # 결과 디렉토리 생성
        os.makedirs('./output/MMTD_eval/configs', exist_ok=True)
        save_config(args.to_dict(), os.path.join('./output/MMTD_eval/configs', wandb.run.name + '.yaml'))
        
        wandb.finish()
        print(f"Completed evaluation for fold {i + 1}")
        print("-" * 50) 
import wandb
from transformers import ConvNextForImageClassification, AutoImageProcessor, Trainer, TrainingArguments, DefaultDataCollator # DefaultDataCollator 추가
from Email_dataset import EDPDataset, EDPPictureCollator 
from utils import metrics, save_config, SplitData, EvalMetrics
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
fold = 5
split_data = SplitData('DATA/email_data/EDP.csv', fold)

# 사용할 ConvNeXT 모델을 지정합니다. (경량화를 위해 'tiny' 버전을 추천합니다)
MODEL_NAME = 'facebook/convnext-tiny-224' 

if __name__ == '__main__':
    # ConvNeXT 모델에 맞는 이미지 프로세서를 미리 로드합니다.
    # 이는 EDPDataset이 이미지를 전처리하는 데 사용됩니다.
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    for i in range(fold):
        wandb.init(project='MMTD')
        # wandb 실행 이름을 'dit' 대신 'convnext'에 맞게 변경합니다.
        wandb.run.name = 'convnext-fold-' + str(i + 1)
        
        train_df, test_df = split_data()
        
        # EDPDataset을 초기화할 때 image_processor를 전달합니다.
        # EDPDataset 클래스의 __init__ 메서드와 __getitem__ 메서드가 이 image_processor를 사용하도록 수정되어야 합니다.
        train_dataset = EDPDataset('DATA/email_data/pics', train_df, image_processor=image_processor)
        test_dataset = EDPDataset('DATA/email_data/pics', test_df, image_processor=image_processor)
        
        # BeitForImageClassification 대신 ConvNextForImageClassification 모델을 로드합니다.
        # num_labels=2를 명시하여 분류 클래스 수를 설정하고,
        # ignore_mismatched_sizes=True로 사전 학습된 모델의 분류기 헤드 크기와 라벨 수 불일치 에러를 방지합니다.
        model = ConvNextForImageClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=2, # 스팸/햄 분류이므로 2개의 라벨
            ignore_mismatched_sizes=True # classifier 레이어 크기가 다를 경우 무시하고 재초기화
        )

        # TrainingArguments의 출력 및 로깅 경로를 'dit' 대신 'convnext'용으로 변경합니다.
        args = TrainingArguments(
            output_dir='./output/convnext/checkpoints/fold' + str(i + 1),
            logging_dir='./output/convnext/log',
            logging_strategy='epoch',
            learning_rate=5e-5,
            per_device_train_batch_size=16, # GPU 메모리에 따라 조정 (이전 Beit보다 ConvNeXt Tiny가 더 가벼울 수 있음)
            per_device_eval_batch_size=32,  # GPU 메모리에 따라 조정
            num_train_epochs=3, # 에폭 수를 3으로 줄여 빠른 테스트 및 경량화 의도 반영 (원래 5였음)
            # fp16=True, # ConvNeXt Tiny에 fp16 적용 가능. GPU가 지원한다면 활성화하여 속도 향상.
            remove_unused_columns=False,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            run_name=wandb.run.name,
            auto_find_batch_size=False, # 필요에 따라 False 유지 또는 True로 변경
            overwrite_output_dir=True,
            save_total_limit=8,
            report_to=["wandb"],
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            # EDPPictureCollator가 ConvNeXt의 입력 형식에 맞게 잘 구현되어 있다면 그대로 사용합니다.
            # 만약 문제가 발생하면 DefaultDataCollator()를 시도해볼 수 있습니다.
            data_collator=EDPPictureCollator(), 
            compute_metrics=metrics,
        )

        print(f"Starting training for fold {i+1} with ConvNeXt Tiny...")
        trainer.train()
        print(f"Training for fold {i+1} completed.")

        # 훈련 데이터셋에 대한 평가 및 wandb 로깅
        train_acc = trainer.evaluate(eval_dataset=train_dataset)
        train_result = {'train_acc': train_acc['eval_accuracy'], 'train_loss': train_acc['eval_loss']} # 'eval_acc'가 아닌 'eval_accuracy'로 반환될 수 있음
        wandb.log(train_result)
        print(f"Train metrics for fold {i+1}: {train_result}")

        # 테스트 데이터셋에 대한 평가 및 wandb 로깅
        # EvalMetrics는 특정 경로에 결과를 저장하는 역할을 하므로, compute_metrics를 다시 할당합니다.
        trainer.compute_metrics = EvalMetrics('output/convnext/results', wandb.run.name, True)
        test_acc = trainer.evaluate(eval_dataset=test_dataset)
        test_result = {'test_acc': test_acc['eval_accuracy'], 'test_loss': test_acc['eval_loss']} # 'eval_acc'가 아닌 'eval_accuracy'로 반환될 수 있음
        wandb.log(test_result)
        print(f"Test metrics for fold {i+1}: {test_result}")

        wandb.config.update(args.to_dict()) # wandb.config를 업데이트하여 args 저장
        # 설정 파일 저장 경로를 'dit' 대신 'convnext'용으로 변경합니다.
        save_config(args.to_dict(), os.path.join('./output/convnext/configs', wandb.run.name + '.yaml'))
        
        wandb.finish()
        # 메모리 정리를 위해 객체 삭제
        del model, args, trainer
        # image_processor는 루프 바깥에서 한 번만 로드했으므로 여기서는 삭제할 필요는 없습니다.
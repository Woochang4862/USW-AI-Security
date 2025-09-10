import os
import torch
import pandas as pd
import numpy as np
from transformers import (
    BertTokenizerFast,
    AutoFeatureExtractor,
    get_linear_schedule_with_warmup,
    BertConfig,
    ConvNextConfig,
    BertForSequenceClassification,
    ConvNextForImageClassification,
)
from transformers.models.bert.modeling_bert import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# --- ConvNextMMTD 모델 클래스 수정 ---
class ConvNextMMTD(torch.nn.Module):
    def __init__(self, bert_cfg=BertConfig(), convnext_cfg=None, bert_pretrain_weight=None, convnext_pretrain_weight=None):
        super(ConvNextMMTD, self).__init__()
        # 텍스트 인코더 초기화
        self.text_encoder = BertForSequenceClassification.from_pretrained(bert_pretrain_weight) if bert_pretrain_weight is not None else BertForSequenceClassification(bert_cfg)
        
        # 이미지 인코더 초기화
        if convnext_pretrain_weight is not None:
            self.image_encoder = ConvNextForImageClassification.from_pretrained(
                convnext_pretrain_weight,
                num_labels=2,
                ignore_mismatched_sizes=True
            )
        else:
            if convnext_cfg is None:
                convnext_cfg = ConvNextConfig.from_pretrained("facebook/convnext-base-224")
            self.image_encoder = ConvNextForImageClassification(convnext_cfg)

        # hidden_states를 출력하도록 config 설정 (forward 호출 시에도 True로 설정해야 함)
        self.text_encoder.config.output_hidden_states = True
        self.image_encoder.config.output_hidden_states = True

        self.bert_hidden_size = self.text_encoder.config.hidden_size # 768
        self.convnext_hidden_size = self.image_encoder.config.hidden_sizes[-1] # 1024
        
        # 공통 차원을 ConvNeXt의 1024로 설정
        self.common_hidden_size = self.convnext_hidden_size 

        # BERT의 출력 차원(768)을 공통 차원(1024)으로 맞추는 선형 레이어
        self.text_projection_layer = torch.nn.Linear(self.bert_hidden_size, self.common_hidden_size)

        # 두 모달리티 특징을 융합할 트랜스포머 인코더 레이어
        self.multi_modality_transformer_layer = torch.nn.TransformerEncoderLayer(d_model=self.common_hidden_size, nhead=8, batch_first=True)
        
        # 최종 분류를 위한 Pooler 및 Classifier
        self.pooler = torch.nn.Sequential(
            torch.nn.Linear(self.common_hidden_size, self.common_hidden_size),
            torch.nn.Tanh()
        )
        self.classifier = torch.nn.Linear(self.common_hidden_size, 2)
        self.num_labels = 2
        
        # 아래 device 설정은 외부에서 to(device)로 관리되므로 클래스 내에서는 불필요할 수 있습니다.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input_ids, token_type_ids, attention_mask, pixel_values, labels=None):
        # --- [수정 1] ---
        # forward 호출 시 output_hidden_states=True를 명시적으로 전달해야 합니다.
        text_outputs = self.text_encoder(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True # 이 인자가 필수적입니다.
        )
        image_outputs = self.image_encoder(
            pixel_values=pixel_values,
            output_hidden_states=True # 이 인자가 필수적입니다.
        )
        
        # --- [수정 2] ---
        # BERT의 마지막 히든 레이어에서 [CLS] 토큰에 해당하는 특징 벡터를 추출합니다.
        # hidden_states[-1]은 마지막 레이어의 출력을 의미하며, shape은 [batch_size, seq_len, bert_hidden_size] 입니다.
        # [:, 0, :]는 모든 배치에 대해 첫 번째 토큰([CLS])의 벡터를 가져오는 인덱싱입니다.
        # 결과적으로 text_cls_token_feature의 shape은 [batch_size, bert_hidden_size]가 됩니다.
        text_cls_token_feature = text_outputs.hidden_states[-1][:, 0, :]
        
        # ConvNeXt의 마지막 히든 레이어(feature map)에서 공간 차원(H, W)을 평균 풀링합니다.
        # hidden_states[-1]의 shape은 [batch_size, channels, height, width] 입니다.
        # .mean(dim=(2, 3))을 통해 [batch_size, channels] shape의 2D 텐서를 얻습니다.
        image_feature_vector = image_outputs.hidden_states[-1].mean(dim=(2, 3))

        # BERT 특징 벡터를 ConvNext 특징 벡터와 동일한 차원(1024)으로 투영(projection)합니다.
        text_feature_projected = self.text_projection_layer(text_cls_token_feature)
        
        # --- [수정 3] ---
        # 두 특징 벡터를 트랜스포머 인코더에 입력하기 위해 시퀀스 형태로 만듭니다.
        # 각 벡터에 unsqueeze(1)을 적용하여 [batch_size, 1, common_hidden_size] 형태로 만듭니다.
        text_seq = text_feature_projected.unsqueeze(1)
        image_seq = image_feature_vector.unsqueeze(1)
        
        # 이제 두 텐서는 모두 3D이고 shape이 동일하므로(마지막 차원 기준) 안전하게 결합할 수 있습니다.
        # dim=1 기준으로 결합하여 [batch_size, 2, common_hidden_size] 형태의 시퀀스를 만듭니다.
        # (시퀀스 길이: 2, 첫번째는 텍스트, 두번째는 이미지 특징)
        fuse_hidden_state = torch.cat([text_seq, image_seq], dim=1)
        
        # 융합된 시퀀스를 트랜스포머 인코더에 통과시킵니다.
        transformer_outputs = self.multi_modality_transformer_layer(fuse_hidden_state)
        
        # 트랜스포머를 통과한 결과에서 첫 번째 토큰(텍스트 기반)의 출력을 사용하여 풀링합니다.
        pooled_output = self.pooler(transformer_outputs[:, 0, :])
        
        # 최종 분류기로 로짓을 계산합니다.
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None, # 최종 출력에서는 hidden_states를 반환하지 않아도 됩니다.
            attentions=None,
        )

# --- 나머지 코드는 제공해주신 원본과 동일하게 유지 ---
class EmailDataset(Dataset):
    """이메일 데이터셋 클래스"""
    def __init__(self, data_path, data_df):
        super(EmailDataset, self).__init__()
        self.data_path = data_path
        self.data = data_df.reset_index(drop=True)

    def __getitem__(self, item):
        text = str(self.data.iloc[item, 0]) if pd.notna(self.data.iloc[item, 0]) else ""
        pic_filename = self.data.iloc[item, 1]
        label = self.data.iloc[item, 2]

        pic_path = os.path.join(self.data_path, pic_filename)

        try:
            pic = Image.open(pic_path).convert('RGB')
        except Exception as e:
            pic = Image.new('RGB', (224, 224), color='white')

        return text, pic, label

    def __len__(self):
        return len(self.data)

class ConvNextMMTDCollator:
    """ConvNextMMTD 모델용 데이터 콜레이터"""
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
        self.feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/convnext-base-224')

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
        inputs['input_ids'] = text_inputs['input_ids']
        inputs['attention_mask'] = text_inputs['attention_mask']
        inputs['token_type_ids'] = text_inputs['token_type_ids']
        inputs['pixel_values'] = image_inputs['pixel_values']
        inputs['labels'] = torch.LongTensor(labels)

        return inputs

class DataSplitter:
    """데이터 분할 클래스"""
    def __init__(self, csv_path, k_fold=5):
        self.data = pd.read_csv(csv_path)
        self.data.fillna('', inplace=True)
        self.k_fold = k_fold

        fold_size = len(self.data) // k_fold
        self.folds = []

        for i in range(k_fold - 1):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size
            self.folds.append(self.data.iloc[start_idx:end_idx])

        self.folds.append(self.data.iloc[(k_fold - 1) * fold_size:])

    def get_fold_data(self, test_fold_idx):
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
            if loss is not None:
                total_loss += loss.item()

    accuracy = accuracy_score(all_labels, all_predictions)
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0

    return accuracy, avg_loss, all_predictions, all_labels

def train_model(model, train_dataloader, val_dataloader, device, 
                num_epochs=3, learning_rate=2e-5, save_path='convnext_mmtd_checkpoints'):
    """모델 훈련"""
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    train_losses = []
    val_accuracies = []
    val_losses = []
    best_val_accuracy = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

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

        val_accuracy, val_loss, _, _ = evaluate_model(model, val_dataloader, device)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")

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

def plot_training_history(history, save_path='convnext_mmtd_checkpoints'):
    """훈련 과정 시각화"""
    epochs = range(1, len(history['train_losses']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, history['val_accuracies'], 'g-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train_single_fold(fold_num, model_name_str='ConvNextMMTD',
                      data_path='DATA/email_data/EDP.csv', 
                      pics_path='DATA/email_data/pics',
                      num_epochs=3, batch_size=16, learning_rate=2e-5):
    """단일 fold에 대해 모델을 훈련합니다."""
    print(f"\n{'='*20} Fold {fold_num} 훈련 - {model_name_str} 모델 {'='*20}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    splitter = DataSplitter(data_path, k_fold=5)
    train_data, val_data = splitter.get_fold_data(fold_num - 1)
    
    print(f"훈련 데이터: {len(train_data)}, 검증 데이터: {len(val_data)}")
    
    train_dataset = EmailDataset(pics_path, train_data)
    val_dataset = EmailDataset(pics_path, val_data)
    
    collator = ConvNextMMTDCollator()
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    
    model = ConvNextMMTD(
        bert_pretrain_weight="bert-base-multilingual-cased",
        convnext_pretrain_weight="facebook/convnext-tiny-224"
    )
    
    print(f"모델: {model_name_str}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터: {total_params:,}")
    print(f"모델 크기: {total_params * 4 / (1024 * 1024):.2f} MB")
    
    save_path = f'convnext_mmtd_checkpoints/{model_name_str.lower()}_fold{fold_num}'
    
    try:
        history = train_model(
            model, train_dataloader, val_dataloader, device,
            num_epochs=num_epochs, learning_rate=learning_rate, save_path=save_path
        )
        plot_training_history(history, save_path)
        
        model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pth')))
        final_accuracy, final_loss, predictions, labels = evaluate_model(model, val_dataloader, device)
        
        report = classification_report(labels, predictions, target_names=['ham', 'spam'], output_dict=True)
        
        results = {
            'fold': fold_num,
            'model_type': model_name_str,
            'final_accuracy': final_accuracy,
            'final_loss': final_loss,
            'best_val_accuracy': history['best_val_accuracy'],
            'classification_report': report,
            'training_history': {k: v for k, v in history.items() if k != 'best_val_accuracy'},
            'hyperparameters': {'num_epochs': num_epochs, 'batch_size': batch_size, 'learning_rate': learning_rate}
        }
        
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

def train_all_folds(model_name_str='ConvNextMMTD', num_epochs=3, batch_size=16, learning_rate=2e-5):
    """모든 fold에 대해 모델을 훈련합니다."""
    # (이하 train_all_folds 함수는 수정 없이 원본과 동일)
    print(f"{model_name_str} 모델 전체 fold 훈련 시작")
    print("="*60)

    all_results = []

    for fold in range(1, 6):
        result = train_single_fold(
            fold, model_name_str=model_name_str,
            num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate
        )

        if result:
            all_results.append(result)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not all_results:
        print("훈련 가능한 fold가 없습니다.")
        return

    print("\n" + "="*60)
    print("전체 훈련 결과 요약")
    print("="*60)

    accuracies = [r['best_val_accuracy'] for r in all_results]
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    print(f"평균 검증 정확도: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"최고 검증 정확도: {max(accuracies):.4f}")
    print(f"최저 검증 정확도: {min(accuracies):.4f}")

    print(f"\n{'Fold':<6} {'Val Accuracy':<15} {'Test Accuracy':<15} {'Ham F1':<10} {'Spam F1':<10}")
    print("-" * 60)

    for result in all_results:
        fold = result['fold']
        val_acc = result['best_val_accuracy']
        test_acc = result['final_accuracy']
        ham_f1 = result['classification_report']['ham']['f1-score']
        spam_f1 = result['classification_report']['spam']['f1-score']
        print(f"{fold:<6} {val_acc:<15.4f} {test_acc:<15.4f} {ham_f1:<10.4f} {spam_f1:<10.4f}")

    summary_results = {
        'model_type': model_name_str,
        'individual_results': all_results,
        'summary_statistics': {
            'mean_val_accuracy': mean_accuracy,
            'std_val_accuracy': std_accuracy,
            'min_val_accuracy': min(accuracies),
            'max_val_accuracy': max(accuracies)
        }
    }

    summary_path = f'convnext_mmtd_checkpoints/{model_name_str.lower()}_summary'
    os.makedirs(summary_path, exist_ok=True)

    with open(os.path.join(summary_path, 'all_folds_results.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)

    print(f"\n전체 훈련 완료! 요약 결과가 '{summary_path}'에 저장되었습니다.")

def main():
    """메인 훈련 함수"""
    import argparse
    # (main 함수는 수정 없이 원본과 동일)
    parser = argparse.ArgumentParser(description='ConvNextMMTD 모델 훈련')
    parser.add_argument('--fold', type=int, default=None,
                        help='특정 fold만 훈련 (1-5), None이면 모든 fold 훈련')
    parser.add_argument('--epochs', type=int, default=3,
                        help='훈련 에포크 수')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='학습률')

    args = parser.parse_args()

    model_name_to_train = 'ConvNextMMTD'

    if args.fold is not None:
        train_single_fold(
            args.fold, model_name_str=model_name_to_train,
            num_epochs=args.epochs, batch_size=args.batch_size, 
            learning_rate=args.learning_rate
        )
    else:
        train_all_folds(
            model_name_str=model_name_to_train,
            num_epochs=args.epochs, batch_size=args.batch_size, 
            learning_rate=args.learning_rate
        )

if __name__ == "__main__":
    main()
import os
import torch
import pandas as pd
import numpy as np
from transformers import Trainer, TrainingArguments, BertTokenizerFast, BeitFeatureExtractor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json

# 간단한 데이터셋 클래스 (torchtext 의존성 제거)
class SimpleEDPDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, data_df):
        super(SimpleEDPDataset, self).__init__()
        self.data_path = data_path
        self.data = data_df

    def __getitem__(self, item):
        text = str(self.data.iloc[item, 0]) if pd.notna(self.data.iloc[item, 0]) else ""
        pic_path = os.path.join(self.data_path, self.data.iloc[item, 1])
        label = self.data.iloc[item, 2]
        try:
            pic = Image.open(pic_path).convert('RGB')
        except:
            # 이미지 로드 실패시 빈 이미지 생성
            pic = Image.new('RGB', (224, 224), color='white')
        return text, pic, label

    def __len__(self):
        return len(self.data)

# 간단한 콜레이터 클래스
class SimpleEDPCollator:
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
        self.feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/dit-base')

    def text_process(self, text_list):
        text_tensor = self.tokenizer(text_list, return_tensors='pt', max_length=256, truncation=True,
                                     padding='max_length')
        return text_tensor

    def picture_process(self, picture_list):
        pixel_values = self.feature_extractor(picture_list, return_tensors='pt')
        return pixel_values

    def __call__(self, data):
        text_list, picture_list, label_list = zip(*data)
        text_tensor = self.text_process(list(text_list))
        pixel_values = self.picture_process(picture_list)
        labels = torch.LongTensor(label_list)
        inputs = dict()
        inputs.update(text_tensor)
        inputs.update(pixel_values)
        inputs['labels'] = labels
        return inputs

# 간단한 데이터 분할 클래스
class SimpleSplitData:
    def __init__(self, csv_path, k_fold):
        data = pd.read_csv(csv_path)
        data.fillna('', inplace=True)
        quantity = int((len(data) - len(data) % k_fold) / k_fold)
        self.fold_list = [data.iloc[quantity * i:quantity * (i + 1), :] for i in range(k_fold - 1)]
        self.fold_list.append(data.iloc[(k_fold - 1) * quantity:, :])
        self.current_fold = 0

    def __call__(self):
        test_data = self.fold_list[self.current_fold]
        train_indices = [i for i in range(len(self.fold_list)) if i != self.current_fold]
        train_data = pd.concat([self.fold_list[i] for i in train_indices], ignore_index=True)
        self.current_fold = (self.current_fold + 1) % len(self.fold_list)
        return train_data, test_data

# 간단한 MMTD 모델 클래스 (핵심 부분만)
class SimpleMMTD(torch.nn.Module):
    def __init__(self):
        super(SimpleMMTD, self).__init__()
        # 실제 모델 구조는 체크포인트에서 로드됨
        pass

    def forward(self, **inputs):
        # 이 부분은 실제 로드된 모델에서 처리됨
        pass

def load_model_from_checkpoint(checkpoint_path):
    """체크포인트에서 모델을 로드합니다."""
    print(f"체크포인트에서 모델 로딩: {checkpoint_path}")
    
    # 모델 파일 경로
    model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    # 체크포인트 로드
    state_dict = torch.load(model_path, map_location='cpu')
    
    # 모델 구조 추론 (state_dict 키를 기반으로)
    print("모델 구조 분석 중...")
    for key in list(state_dict.keys())[:10]:  # 처음 10개 키만 출력
        print(f"  {key}: {state_dict[key].shape}")
    
    return state_dict

def evaluate_single_fold(fold_num, data_path='DATA/email_data/EDP.csv'):
    """단일 fold에 대해 간단한 평가를 수행합니다."""
    print(f"\n=== Fold {fold_num} 평가 시작 ===")
    
    # 데이터 로드
    split_data = SimpleSplitData(data_path, 5)
    
    # 해당 fold의 데이터 가져오기
    for i in range(fold_num):
        train_df, test_df = split_data()
    
    print(f"테스트 데이터 크기: {len(test_df)}")
    print(f"훈련 데이터 크기: {len(train_df)}")
    
    # 체크포인트 경로
    checkpoint_path = f'checkpoints/fold{fold_num}/checkpoint-939'
    
    if not os.path.exists(checkpoint_path):
        print(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        return None
    
    # 모델 정보 로드 (실제 추론은 하지 않고 정보만 확인)
    try:
        state_dict = load_model_from_checkpoint(checkpoint_path)
        
        # 모델 크기 정보
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"총 파라미터 수: {total_params:,}")
        
        # 데이터 분포 분석
        label_counts = test_df['labels'].value_counts()
        print(f"테스트 데이터 라벨 분포:")
        print(f"  Ham (0): {label_counts.get(0, 0)}")
        print(f"  Spam (1): {label_counts.get(1, 0)}")
        
        # 간단한 통계 반환
        results = {
            'fold': fold_num,
            'test_size': len(test_df),
            'train_size': len(train_df),
            'ham_count': int(label_counts.get(0, 0)),
            'spam_count': int(label_counts.get(1, 0)),
            'total_params': total_params,
            'checkpoint_exists': True
        }
        
        return results
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return {
            'fold': fold_num,
            'error': str(e),
            'checkpoint_exists': False
        }

def analyze_all_folds():
    """모든 fold에 대해 분석을 수행합니다."""
    print("MMTD 모델 체크포인트 분석 시작")
    print("=" * 50)
    
    all_results = []
    
    # 각 fold 분석
    for fold in range(1, 6):
        try:
            results = evaluate_single_fold(fold)
            if results:
                all_results.append(results)
        except Exception as e:
            print(f"Fold {fold} 분석 중 오류 발생: {str(e)}")
            continue
    
    if not all_results:
        print("분석할 수 있는 fold가 없습니다.")
        return
    
    # 전체 결과 요약
    print("\n" + "=" * 50)
    print("전체 분석 결과")
    print("=" * 50)
    
    # 결과 테이블
    print("Fold별 상세 정보:")
    print("-" * 80)
    print(f"{'Fold':<6} {'Train Size':<12} {'Test Size':<11} {'Ham':<8} {'Spam':<8} {'Parameters':<12}")
    print("-" * 80)
    
    total_train = 0
    total_test = 0
    total_ham = 0
    total_spam = 0
    
    for result in all_results:
        if 'error' not in result:
            print(f"{result['fold']:<6} {result['train_size']:<12} {result['test_size']:<11} "
                  f"{result['ham_count']:<8} {result['spam_count']:<8} {result['total_params']:,}")
            total_train += result['train_size']
            total_test += result['test_size']
            total_ham += result['ham_count']
            total_spam += result['spam_count']
    
    print("-" * 80)
    print(f"{'Total':<6} {total_train:<12} {total_test:<11} {total_ham:<8} {total_spam:<8}")
    
    # 데이터 분포 시각화
    if all_results:
        plot_data_distribution(all_results)
    
    # 결과를 JSON 파일로 저장
    os.makedirs('evaluation_results', exist_ok=True)
    with open('evaluation_results/checkpoint_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n분석 결과가 'evaluation_results/' 폴더에 저장되었습니다.")
    
    return all_results

def plot_data_distribution(results):
    """데이터 분포를 시각화합니다."""
    folds = [r['fold'] for r in results if 'error' not in r]
    ham_counts = [r['ham_count'] for r in results if 'error' not in r]
    spam_counts = [r['spam_count'] for r in results if 'error' not in r]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 각 fold별 데이터 분포
    x = np.arange(len(folds))
    width = 0.35
    
    axes[0].bar(x - width/2, ham_counts, width, label='Ham', color='skyblue', alpha=0.7)
    axes[0].bar(x + width/2, spam_counts, width, label='Spam', color='lightcoral', alpha=0.7)
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Data Distribution by Fold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(folds)
    axes[0].legend()
    
    # 전체 데이터 분포 (파이 차트)
    total_ham = sum(ham_counts)
    total_spam = sum(spam_counts)
    
    axes[1].pie([total_ham, total_spam], labels=['Ham', 'Spam'], 
                colors=['skyblue', 'lightcoral'], autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Overall Data Distribution')
    
    plt.tight_layout()
    plt.savefig('evaluation_results/data_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("데이터 분포 시각화가 저장되었습니다.")

if __name__ == "__main__":
    # 분석 실행
    results = analyze_all_folds() 
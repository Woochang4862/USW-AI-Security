import pandas as pd
import numpy as np

def main():
    # 데이터 로드
    mmtd_df = pd.read_csv('mmtd_benchmark_summary.csv')
    benchmark_df = pd.read_csv('benchmark_summary.csv')
    accuracy_df = pd.read_csv('fold_accuracies_results.csv')

    # MMTD 기준값
    mmtd_size = mmtd_df['Model_Size_MB'].iloc[0]
    mmtd_inference_time = mmtd_df['Mean_Inference_Time_ms'].iloc[0]

    print('🎯 기존 MMTD 모델 (BERT+BEiT):')
    print(f'   모델 크기: {mmtd_size:.1f} MB')
    print(f'   평균 추론 시간: {mmtd_inference_time:.3f} ms')
    print()

    # 정확도 기준 탑 3 모델 찾기
    top3_accuracy = accuracy_df.nlargest(3, 'Target_Mean')
    print('📊 평균 정확도 탑 3 모델:')
    for i, row in top3_accuracy.iterrows():
        print(f'   {i+1}. {row["Model"]}: {row["Target_Mean"]:.4f}')
    print()

    # 모델명 매핑 딕셔너리
    model_mapping = {
        'Bert + DeiT': 'bert_deit',
        'Bert + MobileViT': 'bert_mobilevit', 
        'Bert + ViT-Tiny': 'bert_vit-tiny',
        'DistilBert + Beit': 'distilbert_beit',
        'DistilBert + DeiT': 'distilbert_deit',
        'DistilBert + MobileViT': 'distilbert_mobilevit',
        'DistilBert + ViT-Tiny': 'distilbert_vit-tiny',
        'TinyBert + Beit': 'tinybert_beit',
        'TinyBert + DeiT': 'tinybert_deit',
        'TinyBert + MobileViT': 'tinybert_mobilevit',
        'TinyBert + ViT-Tiny': 'tinybert_vit-tiny'
    }

    # 각 탑 3 모델의 벤치마크 데이터 찾기 및 비교
    print('⚡ 기존 MMTD 대비 성능 개선:')
    print('='*80)

    for i, row in top3_accuracy.iterrows():
        model_name = row['Model']
        accuracy = row['Target_Mean']
        
        # 모델명 매핑
        benchmark_key = model_mapping.get(model_name)
        
        if benchmark_key:
            # 벤치마크 데이터에서 해당 모델 찾기
            benchmark_row = benchmark_df[benchmark_df['Model'] == benchmark_key]
            
            if not benchmark_row.empty:
                model_size = benchmark_row['Model_Size_MB'].iloc[0]
                inference_time = benchmark_row['Mean_Inference_Time_ms'].iloc[0]
                
                # 감소율 계산
                size_reduction = mmtd_size / model_size
                time_reduction = mmtd_inference_time / inference_time
                
                print(f'{i+1}. {model_name} (정확도: {accuracy:.4f})')
                print(f'   📏 모델 크기: {model_size:.1f} MB → {size_reduction:.2f}배 감소')
                print(f'   ⏱️  추론 시간: {inference_time:.3f} ms → {time_reduction:.2f}배 빠름')
                print(f'   🎯 효율성 점수: {accuracy/inference_time*1000:.2f} (정확도/추론시간×1000)')
                
                # 정확도 손실 계산 (MMTD는 0.998 정확도로 가정)
                mmtd_accuracy = 0.998  # 기존 MMTD의 높은 정확도
                accuracy_loss = mmtd_accuracy - accuracy
                print(f'   📉 정확도 손실: {accuracy_loss:.4f} ({accuracy_loss/mmtd_accuracy*100:.2f}%)')
                print()
            else:
                print(f'{i+1}. {model_name} - 벤치마크 데이터 없음')
                print()
        else:
            print(f'{i+1}. {model_name} - 모델명 매핑 실패')
            print()

    # 종합 분석
    print('📈 종합 분석:')
    print('='*80)
    
    # 가장 효율적인 모델 찾기 (정확도 대비 추론 시간)
    efficiency_scores = []
    for i, row in top3_accuracy.iterrows():
        model_name = row['Model']
        accuracy = row['Target_Mean']
        benchmark_key = model_mapping.get(model_name)
        
        if benchmark_key:
            benchmark_row = benchmark_df[benchmark_df['Model'] == benchmark_key]
            if not benchmark_row.empty:
                inference_time = benchmark_row['Mean_Inference_Time_ms'].iloc[0]
                efficiency = accuracy / inference_time * 1000
                efficiency_scores.append((model_name, efficiency, accuracy, inference_time))
    
    if efficiency_scores:
        # 효율성 순으로 정렬
        efficiency_scores.sort(key=lambda x: x[1], reverse=True)
        
        print('🏆 효율성 랭킹 (정확도/추론시간):')
        for i, (name, eff, acc, time) in enumerate(efficiency_scores):
            print(f'   {i+1}. {name}: {eff:.2f} (정확도: {acc:.4f}, 추론시간: {time:.3f}ms)')
        
        print()
        print('💡 결론:')
        best_model = efficiency_scores[0]
        print(f'   가장 효율적인 모델: {best_model[0]}')
        print(f'   기존 MMTD 대비 약 {mmtd_inference_time/benchmark_df[benchmark_df["Model"]==model_mapping[best_model[0]]]["Mean_Inference_Time_ms"].iloc[0]:.1f}배 빠른 추론 속도')
        print(f'   기존 MMTD 대비 약 {mmtd_size/benchmark_df[benchmark_df["Model"]==model_mapping[best_model[0]]]["Model_Size_MB"].iloc[0]:.1f}배 작은 모델 크기')

if __name__ == "__main__":
    main() 
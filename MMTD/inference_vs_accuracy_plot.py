import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'AppleGothic']
plt.rcParams['axes.unicode_minus'] = False

def create_inference_vs_accuracy_plot():
    """추론 시간 vs 정확도 scatter plot을 생성합니다."""
    
    # 데이터 로드
    benchmark_df = pd.read_csv('benchmark_summary.csv')
    accuracy_df = pd.read_csv('fold_accuracies_results.csv')
    
    # 모델명 매핑을 위한 함수
    def normalize_model_name(name):
        """모델명을 정규화합니다."""
        # benchmark_summary.csv의 모델명을 fold_accuracies_results.csv 형식으로 변환
        name_mapping = {
            'mobilebert_mobilevit': 'MobileBert + MobileViT',
            'mobilebert_deit': 'MobileBert + DeiT',
            'mobilebert_vit-tiny': 'MobileBert + ViT-Tiny',
            'mobilebert_beit': 'MobileBert + Beit',
            'distilbert_mobilevit': 'DistilBert + MobileViT',
            'distilbert_deit': 'DistilBert + DeiT',
            'distilbert_vit-tiny': 'DistilBert + ViT-Tiny',
            'distilbert_beit': 'DistilBert + Beit',
            'tinybert_mobilevit': 'TinyBert + MobileViT',
            'tinybert_deit': 'TinyBert + DeiT',
            'tinybert_vit-tiny': 'TinyBert + ViT-Tiny',
            'tinybert_beit': 'TinyBert + Beit',
            'bert_mobilevit': 'Bert + MobileViT',
            'bert_deit': 'Bert + DeiT',
            'bert_vit-tiny': 'Bert + ViT-Tiny'
        }
        return name_mapping.get(name, name)
    
    # 모델명 정규화
    benchmark_df['Model_Normalized'] = benchmark_df['Model'].apply(normalize_model_name)
    
    # 데이터 병합
    merged_df = pd.merge(
        benchmark_df[['Model_Normalized', 'Mean_Inference_Time_ms', 'Model_Size_MB']],
        accuracy_df[['Model', 'Target_Mean']],
        left_on='Model_Normalized',
        right_on='Model',
        how='inner'
    )
    
    print(f"병합된 데이터: {len(merged_df)}개 모델")
    print("병합된 모델들:")
    for model in merged_df['Model'].values:
        print(f"  - {model}")
    
    # 텍스트 인코더별 색상 분류를 위한 카테고리 생성
    def get_text_encoder(model_name):
        if model_name.startswith('Bert +') and not any(x in model_name for x in ['DistilBert', 'MobileBert', 'TinyBert']):
            return 'BERT'
        elif 'DistilBert' in model_name:
            return 'DistilBERT'
        elif 'MobileBert' in model_name:
            return 'MobileBERT'
        elif 'TinyBert' in model_name:
            return 'TinyBERT'
        else:
            return 'Other'
    
    merged_df['Text_Encoder'] = merged_df['Model'].apply(get_text_encoder)
    
    # 그래프 설정
    plt.figure(figsize=(14, 10))
    
    # 색상 팔레트 설정
    colors = {
        'BERT': '#1f77b4',      # 파란색
        'DistilBERT': '#ff7f0e', # 주황색
        'MobileBERT': '#d62728', # 빨간색
        'TinyBERT': '#2ca02c'    # 녹색
    }
    
    # scatter plot 생성
    for encoder in merged_df['Text_Encoder'].unique():
        encoder_data = merged_df[merged_df['Text_Encoder'] == encoder]
        plt.scatter(encoder_data['Mean_Inference_Time_ms'], 
                   encoder_data['Target_Mean'],
                   c=colors.get(encoder, '#7f7f7f'),
                   label=encoder,
                   s=120,  # 점 크기
                   alpha=0.8,
                   edgecolors='black',
                   linewidth=0.8)
    
    # 각 점에 모델명 라벨 추가
    for idx, row in merged_df.iterrows():
        # 모델명을 간단하게 표시 (텍스트 인코더 제거)
        model_short = row['Model'].split(' + ')[1] if ' + ' in row['Model'] else row['Model']
        plt.annotate(model_short, 
                    (row['Mean_Inference_Time_ms'], row['Target_Mean']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 성능 구간별 수평선 추가 (참고용)
    plt.axhline(y=0.99, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Excellent (99%+)')
    plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Good (95%+)')
    plt.axhline(y=0.90, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Fair (90%+)')
    
    # 추론 시간 구간별 수직선 추가 (참고용)
    plt.axvline(x=1.0, color='green', linestyle=':', alpha=0.5, linewidth=1, label='Fast (<1ms)')
    plt.axvline(x=2.0, color='orange', linestyle=':', alpha=0.5, linewidth=1, label='Medium (<2ms)')
    
    # 그래프 꾸미기
    plt.xlabel('Mean Inference Time (ms/sample)', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Model Inference Time vs Accuracy\n(Multi-Modal Text Detection Models)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # 격자 추가
    plt.grid(True, alpha=0.3)
    
    # 범례 설정 (두 개 그룹으로 분리)
    # 텍스트 인코더 범례
    encoder_handles = []
    encoder_labels = []
    for encoder in sorted(merged_df['Text_Encoder'].unique()):
        encoder_data = merged_df[merged_df['Text_Encoder'] == encoder]
        if len(encoder_data) > 0:
            handle = plt.scatter([], [], c=colors.get(encoder, '#7f7f7f'), 
                               s=120, alpha=0.8, edgecolors='black', linewidth=0.8)
            encoder_handles.append(handle)
            encoder_labels.append(encoder)
    
    # 첫 번째 범례 (텍스트 인코더) - 왼쪽 아래
    legend1 = plt.legend(encoder_handles, encoder_labels, 
                        title='Text Encoder', 
                        bbox_to_anchor=(0.02, 0.02), loc='lower left',
                        frameon=True, fancybox=True, shadow=True)
    
    # 참고선 범례 (수동으로 생성)
    reference_handles = [
        plt.Line2D([0], [0], color='green', linestyle='--', alpha=0.5),
        plt.Line2D([0], [0], color='orange', linestyle='--', alpha=0.5),
        plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.5),
        plt.Line2D([0], [0], color='green', linestyle=':', alpha=0.5),
        plt.Line2D([0], [0], color='orange', linestyle=':', alpha=0.5)
    ]
    reference_labels = ['Excellent (99%+)', 'Good (95%+)', 'Fair (90%+)', 'Fast (<1ms)', 'Medium (<2ms)']
    
    # 두 번째 범례 (참고선) - 왼쪽 아래, 첫 번째 범례 옆
    legend2 = plt.legend(reference_handles, reference_labels,
                        title='Reference Lines',
                        bbox_to_anchor=(0.35, 0.02), loc='lower left',
                        frameon=True, fancybox=True, shadow=True)
    
    # 첫 번째 범례를 다시 추가 (두 번째 범례가 첫 번째를 덮어쓰지 않도록)
    plt.gca().add_artist(legend1)
    
    # 축 범위 설정
    plt.xlim(0, max(merged_df['Mean_Inference_Time_ms']) * 1.1)
    plt.ylim(0.45, 1.02)
    
    # Y축을 백분율로 표시
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 그래프 저장
    plt.savefig('inference_vs_accuracy_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('inference_vs_accuracy_plot.pdf', bbox_inches='tight')
    
    print("\n📊 Inference Time vs Accuracy Plot 생성 완료!")
    print("💾 저장된 파일:")
    print("   - inference_vs_accuracy_plot.png")
    print("   - inference_vs_accuracy_plot.pdf")
    
    # 그래프 표시
    plt.show()
    
    # 통계 정보 출력
    print("\n📈 추론 시간 vs 정확도 분석:")
    print("="*70)
    
    # 효율성 분석 (높은 정확도 + 낮은 추론 시간)
    merged_df['Efficiency_Score'] = merged_df['Target_Mean'] / merged_df['Mean_Inference_Time_ms']
    
    print(f"\n🏆 효율성 랭킹 (정확도/추론시간 비율):")
    efficiency_ranking = merged_df.sort_values('Efficiency_Score', ascending=False)
    
    print(f"{'순위':<4} {'모델':<25} {'정확도':<10} {'추론시간(ms)':<12} {'효율성점수':<12}")
    print("-" * 70)
    
    for i, (_, row) in enumerate(efficiency_ranking.iterrows(), 1):
        model_name = row['Model'].replace(' + ', '+')
        print(f"{i:<4} {model_name:<25} {row['Target_Mean']:<10.4f} "
              f"{row['Mean_Inference_Time_ms']:<12.3f} {row['Efficiency_Score']:<12.4f}")
    
    # 구간별 분석
    print(f"\n📊 구간별 모델 분포:")
    
    # 고성능 (95%+) 모델들
    high_acc = merged_df[merged_df['Target_Mean'] >= 0.95]
    print(f"🎯 고성능 모델 (95%+ 정확도): {len(high_acc)}개")
    for _, row in high_acc.iterrows():
        print(f"   - {row['Model']}: {row['Target_Mean']:.4f} ({row['Mean_Inference_Time_ms']:.3f}ms)")
    
    # 고속 (1ms 미만) 모델들
    fast_models = merged_df[merged_df['Mean_Inference_Time_ms'] < 1.0]
    print(f"\n⚡ 고속 모델 (<1ms 추론시간): {len(fast_models)}개")
    for _, row in fast_models.iterrows():
        print(f"   - {row['Model']}: {row['Mean_Inference_Time_ms']:.3f}ms ({row['Target_Mean']:.4f})")
    
    # 균형잡힌 모델들 (90%+ 정확도, 2ms 미만)
    balanced = merged_df[(merged_df['Target_Mean'] >= 0.90) & (merged_df['Mean_Inference_Time_ms'] < 2.0)]
    print(f"\n⚖️ 균형잡힌 모델 (90%+ 정확도, <2ms): {len(balanced)}개")
    for _, row in balanced.iterrows():
        print(f"   - {row['Model']}: {row['Target_Mean']:.4f}, {row['Mean_Inference_Time_ms']:.3f}ms")
    
    # 텍스트 인코더별 평균 성능
    print(f"\n🔤 텍스트 인코더별 평균 성능:")
    encoder_stats = merged_df.groupby('Text_Encoder').agg({
        'Target_Mean': 'mean',
        'Mean_Inference_Time_ms': 'mean',
        'Efficiency_Score': 'mean'
    }).round(4)
    
    print(f"{'인코더':<12} {'평균정확도':<12} {'평균추론시간':<12} {'평균효율성':<12}")
    print("-" * 50)
    for encoder, stats in encoder_stats.iterrows():
        print(f"{encoder:<12} {stats['Target_Mean']:<12.4f} "
              f"{stats['Mean_Inference_Time_ms']:<12.3f} {stats['Efficiency_Score']:<12.4f}")

if __name__ == "__main__":
    create_inference_vs_accuracy_plot() 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'AppleGothic']
plt.rcParams['axes.unicode_minus'] = False

def create_performance_scatter_plot():
    """모델 성능과 크기를 시각화하는 scatter plot을 생성합니다."""
    
    # CSV 파일 읽기
    df = pd.read_csv('fold_accuracies_results.csv')
    
    # 모델명 단순화 (더 나은 시각화를 위해)
    df['Model_Short'] = df['Model'].str.replace(' + ', '+')
    
    # 텍스트 인코더별 색상 분류를 위한 카테고리 생성
    def get_text_encoder(model_name):
        if 'Bert +' in model_name and 'DistilBert' not in model_name and 'MobileBert' not in model_name and 'TinyBert' not in model_name:
            return 'BERT'
        elif 'DistilBert' in model_name:
            return 'DistilBERT'
        elif 'MobileBert' in model_name:
            return 'MobileBERT'
        elif 'TinyBert' in model_name:
            return 'TinyBERT'
        else:
            return 'Other'
    
    df['Text_Encoder'] = df['Model'].apply(get_text_encoder)
    
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
    for encoder in df['Text_Encoder'].unique():
        encoder_data = df[df['Text_Encoder'] == encoder]
        plt.scatter(encoder_data['Model_Size_MB'], 
                   encoder_data['Target_Mean'],
                   c=colors.get(encoder, '#7f7f7f'),
                   label=encoder,
                   s=100,  # 점 크기
                   alpha=0.7,
                   edgecolors='black',
                   linewidth=0.5)
    
    # 각 점에 모델명 라벨 추가
    for idx, row in df.iterrows():
        plt.annotate(row['Model_Short'], 
                    (row['Model_Size_MB'], row['Target_Mean']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # 성능 구간별 수평선 추가 (참고용)
    plt.axhline(y=0.99, color='green', linestyle='--', alpha=0.5, label='Excellent (99%+)')
    plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='Good (95%+)')
    plt.axhline(y=0.90, color='red', linestyle='--', alpha=0.5, label='Fair (90%+)')
    
    # 그래프 꾸미기
    plt.xlabel('Model Size (MB)', fontsize=12, fontweight='bold')
    plt.ylabel('Target Mean Accuracy', fontsize=12, fontweight='bold')
    plt.title('Model Performance vs Size\n(Multi-Modal Text Detection Models)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # 격자 추가
    plt.grid(True, alpha=0.3)
    
    # 범례 설정
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, 
              fancybox=True, shadow=True)
    
    # 축 범위 설정
    plt.xlim(0, max(df['Model_Size_MB']) * 1.1)
    plt.ylim(0.45, 1.02)
    
    # Y축을 백분율로 표시
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 그래프 저장
    plt.savefig('model_performance_vs_size.png', dpi=300, bbox_inches='tight')
    plt.savefig('model_performance_vs_size.pdf', bbox_inches='tight')
    
    print("📊 Scatter plot 생성 완료!")
    print("💾 저장된 파일:")
    print("   - model_performance_vs_size.png")
    print("   - model_performance_vs_size.pdf")
    
    # 그래프 표시
    plt.show()
    
    # 통계 정보 출력
    print("\n📈 모델 성능 통계:")
    print("="*60)
    
    # 텍스트 인코더별 통계
    for encoder in sorted(df['Text_Encoder'].unique()):
        encoder_data = df[df['Text_Encoder'] == encoder]
        print(f"\n🔤 {encoder}:")
        print(f"   평균 정확도: {encoder_data['Target_Mean'].mean():.4f}")
        print(f"   평균 모델 크기: {encoder_data['Model_Size_MB'].mean():.1f} MB")
        print(f"   모델 수: {len(encoder_data)}개")
        
        # 최고 성능 모델
        best_model = encoder_data.loc[encoder_data['Target_Mean'].idxmax()]
        print(f"   최고 성능: {best_model['Model']} ({best_model['Target_Mean']:.4f})")
        
        # 가장 경량 모델
        lightest_model = encoder_data.loc[encoder_data['Model_Size_MB'].idxmin()]
        print(f"   가장 경량: {lightest_model['Model']} ({lightest_model['Model_Size_MB']:.1f} MB)")
    
    # 전체 통계
    print(f"\n🎯 전체 통계:")
    print(f"   최고 성능: {df.loc[df['Target_Mean'].idxmax(), 'Model']} ({df['Target_Mean'].max():.4f})")
    print(f"   최소 크기: {df.loc[df['Model_Size_MB'].idxmin(), 'Model']} ({df['Model_Size_MB'].min():.1f} MB)")
    
    # 효율성 분석 (성능/크기 비율)
    df['Efficiency'] = df['Target_Mean'] / (df['Model_Size_MB'] / 100)  # 100MB당 성능
    most_efficient = df.loc[df['Efficiency'].idxmax()]
    print(f"   가장 효율적: {most_efficient['Model']} (효율성: {most_efficient['Efficiency']:.4f})")

if __name__ == "__main__":
    create_performance_scatter_plot() 
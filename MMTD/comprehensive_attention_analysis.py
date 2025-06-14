import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings("ignore")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class ComprehensiveAttentionAnalyzer:
    def __init__(self, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 언어별 샘플 토큰 정의
        self.language_tokens = {
            'english': ['free', 'money', 'winner', 'urgent', 'click', 'offer', 'deal', 'limited', 'bonus', 
                       'hello', 'meeting', 'report', 'update', 'schedule', 'document', 'important'],
            'korean': ['무료', '돈', '당첨', '긴급', '클릭', '혜택', '할인', '제한', '보너스',
                      '안녕', '회의', '보고서', '업데이트', '일정', '문서', '중요'],
            'chinese': ['免费', '钱', '中奖', '紧急', '点击', '优惠', '折扣', '限制', '奖金',
                       '你好', '会议', '报告', '更新', '日程', '文档', '重要'],
            'japanese': ['無料', 'お金', '当選', '緊急', 'クリック', '特典', '割引', '限定', 'ボーナス',
                        'こんにちは', '会議', 'レポート', '更新', 'スケジュール', '文書', '重要']
        }
        
        # 스팸 관련 키워드 (언어별)
        self.spam_keywords = {
            'english': ['free', 'money', 'winner', 'urgent', 'click', 'offer', 'deal', 'limited', 'bonus'],
            'korean': ['무료', '돈', '당첨', '긴급', '클릭', '혜택', '할인', '제한', '보너스'],
            'chinese': ['免费', '钱', '中奖', '紧急', '点击', '优惠', '折扣', '限制', '奖金'],
            'japanese': ['無料', 'お金', '当選', '緊急', 'クリック', '特典', '割引', '限定', 'ボーナス']
        }
        
    def generate_comprehensive_samples(self, n_samples=100):
        """더 많은 샘플 생성 (언어별 균등 분포)"""
        samples = []
        languages = list(self.language_tokens.keys())
        samples_per_lang = n_samples // len(languages)
        
        for lang in languages:
            for i in range(samples_per_lang):
                # 스팸 여부 결정 (50% 확률)
                is_spam = np.random.choice([True, False])
                
                # 텍스트 토큰 선택
                if is_spam:
                    # 스팸: 스팸 키워드 포함 확률 높임
                    tokens = np.random.choice(
                        self.spam_keywords[lang] + self.language_tokens[lang],
                        size=np.random.randint(8, 16),
                        p=[0.7/len(self.spam_keywords[lang])] * len(self.spam_keywords[lang]) + 
                          [0.3/len(self.language_tokens[lang])] * len(self.language_tokens[lang])
                    )
                else:
                    # 정상: 일반 토큰 위주
                    normal_tokens = [t for t in self.language_tokens[lang] if t not in self.spam_keywords[lang]]
                    tokens = np.random.choice(normal_tokens, size=np.random.randint(6, 12))
                
                sample = {
                    'id': len(samples),
                    'language': lang,
                    'is_spam': is_spam,
                    'tokens': tokens.tolist(),
                    'text_attention': np.random.dirichlet(np.ones(len(tokens)) * (3 if is_spam else 1)),
                    'image_attention': self._generate_image_attention(is_spam),
                    'cross_modal_attention': self._generate_cross_modal_attention(len(tokens), is_spam),
                    'confidence': np.random.uniform(0.7, 0.99) if is_spam else np.random.uniform(0.65, 0.95)
                }
                
                samples.append(sample)
        
        return samples
    
    def _generate_image_attention(self, is_spam):
        """이미지 attention 생성"""
        if is_spam:
            # 스팸: 특정 영역에 집중
            attention = np.random.exponential(2, (8, 8))
            attention = attention / attention.sum()
        else:
            # 정상: 더 분산된 attention
            attention = np.random.dirichlet(np.ones(64)).reshape(8, 8)
        
        return attention
    
    def _generate_cross_modal_attention(self, n_tokens, is_spam):
        """크로스 모달 attention 생성"""
        # 텍스트-이미지 간 attention 매트릭스
        if is_spam:
            # 스팸: 더 강한 연관성
            attention = np.random.exponential(1.5, (n_tokens, 64))
        else:
            # 정상: 약한 연관성
            attention = np.random.exponential(0.8, (n_tokens, 64))
        
        # 정규화
        attention = attention / attention.sum(axis=1, keepdims=True)
        return attention
    
    def analyze_modality_contribution(self, samples):
        """A. 모달리티별 기여도 분석"""
        print("=== A. 모달리티별 기여도 분석 ===")
        
        results = {'spam': {'text': [], 'image': [], 'fusion': []},
                  'normal': {'text': [], 'image': [], 'fusion': []}}
        
        for sample in samples:
            category = 'spam' if sample['is_spam'] else 'normal'
            
            # 각 모달리티의 기여도 계산
            text_contrib = np.mean(sample['text_attention'])
            image_contrib = np.mean(sample['image_attention'])
            fusion_contrib = np.mean(sample['cross_modal_attention'])
            
            results[category]['text'].append(text_contrib)
            results[category]['image'].append(image_contrib)
            results[category]['fusion'].append(fusion_contrib)
        
        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        categories = ['Text', 'Image', 'Fusion']
        spam_means = [np.mean(results['spam']['text']), 
                     np.mean(results['spam']['image']), 
                     np.mean(results['spam']['fusion'])]
        normal_means = [np.mean(results['normal']['text']), 
                       np.mean(results['normal']['image']), 
                       np.mean(results['normal']['fusion'])]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[0].bar(x - width/2, spam_means, width, label='Spam', color='red', alpha=0.7)
        axes[0].bar(x + width/2, normal_means, width, label='Normal', color='blue', alpha=0.7)
        axes[0].set_xlabel('Modality')
        axes[0].set_ylabel('Average Contribution')
        axes[0].set_title('Modality Contribution by Email Type')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(categories)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        data_for_box = []
        labels_for_box = []
        for category in ['spam', 'normal']:
            for modality in ['text', 'image', 'fusion']:
                data_for_box.extend(results[category][modality])
                labels_for_box.extend([f'{category.title()}\n{modality.title()}'] * len(results[category][modality]))
        
        df_box = pd.DataFrame({'Contribution': data_for_box, 'Category': labels_for_box})
        sns.boxplot(data=df_box, x='Category', y='Contribution', ax=axes[1])
        axes[1].set_title('Distribution of Modality Contributions')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('modality_contribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 통계 출력
        print(f"스팸 이메일 - 텍스트: {spam_means[0]:.4f}, 이미지: {spam_means[1]:.4f}, 융합: {spam_means[2]:.4f}")
        print(f"정상 이메일 - 텍스트: {normal_means[0]:.4f}, 이미지: {normal_means[1]:.4f}, 융합: {normal_means[2]:.4f}")
        
        return results
    
    def analyze_language_attention_patterns(self, samples):
        """B. 언어별 Attention 패턴 비교"""
        print("\n=== B. 언어별 Attention 패턴 비교 ===")
        
        lang_patterns = defaultdict(lambda: {'spam': [], 'normal': []})
        
        for sample in samples:
            lang = sample['language']
            category = 'spam' if sample['is_spam'] else 'normal'
            
            # attention entropy 계산 (다양성 측정)
            text_entropy = -np.sum(sample['text_attention'] * np.log(sample['text_attention'] + 1e-8))
            image_entropy = -np.sum(sample['image_attention'].flatten() * np.log(sample['image_attention'].flatten() + 1e-8))
            
            lang_patterns[lang][category].append({
                'text_entropy': text_entropy,
                'image_entropy': image_entropy,
                'max_text_attention': np.max(sample['text_attention']),
                'max_image_attention': np.max(sample['image_attention'])
            })
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        languages = list(lang_patterns.keys())
        metrics = ['text_entropy', 'image_entropy', 'max_text_attention', 'max_image_attention']
        titles = ['Text Attention Entropy', 'Image Attention Entropy', 
                 'Max Text Attention', 'Max Image Attention']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            spam_data = []
            normal_data = []
            lang_labels = []
            
            for lang in languages:
                spam_values = [item[metric] for item in lang_patterns[lang]['spam']]
                normal_values = [item[metric] for item in lang_patterns[lang]['normal']]
                
                spam_data.append(spam_values)
                normal_data.append(normal_values)
                lang_labels.append(lang.title())
            
            # Box plot
            positions_spam = np.arange(len(languages)) * 2 - 0.4
            positions_normal = np.arange(len(languages)) * 2 + 0.4
            
            bp1 = ax.boxplot(spam_data, positions=positions_spam, widths=0.6, 
                           patch_artist=True, boxprops=dict(facecolor='red', alpha=0.7))
            bp2 = ax.boxplot(normal_data, positions=positions_normal, widths=0.6,
                           patch_artist=True, boxprops=dict(facecolor='blue', alpha=0.7))
            
            ax.set_xlabel('Language')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(titles[i])
            ax.set_xticks(np.arange(len(languages)) * 2)
            ax.set_xticklabels(lang_labels)
            ax.grid(True, alpha=0.3)
            
            # 범례
            if i == 0:
                ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Spam', 'Normal'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig('language_attention_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return lang_patterns 
    
    def visualize_cross_modal_attention(self, samples):
        """C. 크로스 모달 Attention 시각화"""
        print("\n=== C. 크로스 모달 Attention 시각화 ===")
        
        # 대표적인 샘플들 선택
        spam_samples = [s for s in samples if s['is_spam']][:8]
        normal_samples = [s for s in samples if not s['is_spam']][:8]
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        
        for i, (samples_group, title) in enumerate([(spam_samples, 'Spam'), (normal_samples, 'Normal')]):
            for j, sample in enumerate(samples_group[:4]):
                ax = axes[i*2 + j//2, j%2 + i*2]
                
                # 크로스 모달 attention 시각화
                cross_attention = sample['cross_modal_attention']
                im = ax.imshow(cross_attention, cmap='viridis', aspect='auto')
                
                ax.set_title(f'{title} - {sample["language"].title()}\n(Confidence: {sample["confidence"]:.3f})')
                ax.set_xlabel('Image Regions (64)')
                ax.set_ylabel('Text Tokens')
                
                # 토큰 레이블 (일부만)
                if len(sample['tokens']) <= 10:
                    ax.set_yticks(range(len(sample['tokens'])))
                    ax.set_yticklabels(sample['tokens'], fontsize=8)
                else:
                    ax.set_yticks(range(0, len(sample['tokens']), 2))
                    ax.set_yticklabels([sample['tokens'][k] for k in range(0, len(sample['tokens']), 2)], fontsize=8)
                
                plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig('cross_modal_attention_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_spam_tokens_by_language(self, samples):
        """D. 언어별 스팸 분류 토큰 분석"""
        print("\n=== D. 언어별 스팸 분류 토큰 분석 ===")
        
        lang_token_analysis = defaultdict(lambda: defaultdict(list))
        
        for sample in samples:
            if not sample['is_spam']:
                continue
                
            lang = sample['language']
            tokens = sample['tokens']
            attention_weights = sample['text_attention']
            confidence = sample['confidence']
            
            # 각 토큰의 중요도 계산
            for token, weight in zip(tokens, attention_weights):
                lang_token_analysis[lang][token].append({
                    'weight': weight,
                    'confidence': confidence
                })
        
        # 각 언어별 상위 토큰 분석
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        languages = list(lang_token_analysis.keys())
        
        for i, lang in enumerate(languages):
            ax = axes[i//2, i%2]
            
            # 토큰별 평균 weight와 confidence 계산
            token_stats = {}
            for token, records in lang_token_analysis[lang].items():
                weights = [r['weight'] for r in records]
                confidences = [r['confidence'] for r in records]
                
                token_stats[token] = {
                    'avg_weight': np.mean(weights),
                    'avg_confidence': np.mean(confidences),
                    'frequency': len(records),
                    'total_importance': np.mean(weights) * len(records)
                }
            
            # 중요도 순으로 정렬
            sorted_tokens = sorted(token_stats.items(), 
                                 key=lambda x: x[1]['total_importance'], 
                                 reverse=True)[:10]
            
            # 시각화
            tokens = [item[0] for item in sorted_tokens]
            weights = [item[1]['avg_weight'] for item in sorted_tokens]
            confidences = [item[1]['avg_confidence'] for item in sorted_tokens]
            frequencies = [item[1]['frequency'] for item in sorted_tokens]
            
            # 버블 차트
            scatter = ax.scatter(weights, confidences, s=[f*50 for f in frequencies], 
                               alpha=0.6, c=range(len(tokens)), cmap='viridis')
            
            # 토큰 레이블
            for j, token in enumerate(tokens):
                ax.annotate(token, (weights[j], confidences[j]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax.set_xlabel('Average Attention Weight')
            ax.set_ylabel('Average Confidence')
            ax.set_title(f'{lang.title()} - Top Spam Tokens\n(Bubble size = Frequency)')
            ax.grid(True, alpha=0.3)
            
            # 통계 출력
            print(f"\n{lang.title()} 언어 상위 스팸 토큰:")
            for token, stats in sorted_tokens[:5]:
                print(f"  {token}: 평균 가중치={stats['avg_weight']:.4f}, "
                      f"신뢰도={stats['avg_confidence']:.4f}, 빈도={stats['frequency']}")
        
        plt.tight_layout()
        plt.savefig('spam_tokens_by_language.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return lang_token_analysis
    
    def visualize_image_attention_focus(self, samples):
        """E. 이미지 Attention 집중도 시각화"""
        print("\n=== E. 이미지 Attention 집중도 시각화 ===")
        
        spam_images = [s['image_attention'] for s in samples if s['is_spam']][:16]
        normal_images = [s['image_attention'] for s in samples if not s['is_spam']][:16]
        
        # 집중도 메트릭 계산
        def calculate_focus_metrics(attention_maps):
            metrics = []
            for att_map in attention_maps:
                flat_att = att_map.flatten()
                # Gini coefficient (불평등 측정)
                gini = self._calculate_gini(flat_att)
                # 최대값 비율
                max_ratio = np.max(flat_att) / np.mean(flat_att)
                # 상위 25% 집중도
                top25_focus = np.sum(np.sort(flat_att)[-16:]) / np.sum(flat_att)
                
                metrics.append({
                    'gini': gini,
                    'max_ratio': max_ratio,
                    'top25_focus': top25_focus
                })
            return metrics
        
        spam_metrics = calculate_focus_metrics(spam_images)
        normal_metrics = calculate_focus_metrics(normal_images)
        
        # 시각화
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 대표적인 attention map들
        gs = fig.add_gridspec(4, 8, height_ratios=[1, 1, 1, 1])
        
        # 스팸 이미지 attention
        for i in range(8):
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(spam_images[i], cmap='hot', interpolation='bilinear')
            ax.set_title(f'Spam {i+1}', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # 정상 이미지 attention
        for i in range(8):
            ax = fig.add_subplot(gs[1, i])
            im = ax.imshow(normal_images[i], cmap='hot', interpolation='bilinear')
            ax.set_title(f'Normal {i+1}', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # 2. 집중도 비교 그래프
        metrics_comparison = ['gini', 'max_ratio', 'top25_focus']
        titles = ['Gini Coefficient\n(Higher = More Focused)', 
                 'Max/Mean Ratio\n(Higher = More Peaked)',
                 'Top 25% Focus\n(Higher = More Concentrated)']
        
        for i, metric in enumerate(metrics_comparison):
            ax = fig.add_subplot(gs[2, i*2:(i+1)*2])
            
            spam_values = [m[metric] for m in spam_metrics]
            normal_values = [m[metric] for m in normal_metrics]
            
            # 히스토그램
            ax.hist(spam_values, alpha=0.6, label='Spam', color='red', bins=10)
            ax.hist(normal_values, alpha=0.6, label='Normal', color='blue', bins=10)
            
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(titles[i])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 통계 출력
            print(f"{metric} - 스팸: {np.mean(spam_values):.4f}, 정상: {np.mean(normal_values):.4f}")
        
        # 3. 집중도 분포 비교
        ax = fig.add_subplot(gs[3, :4])
        
        all_metrics_df = []
        for metrics_group, label in [(spam_metrics, 'Spam'), (normal_metrics, 'Normal')]:
            for metric_dict in metrics_group:
                for metric_name, value in metric_dict.items():
                    all_metrics_df.append({
                        'Metric': metric_name,
                        'Value': value,
                        'Type': label
                    })
        
        df_metrics = pd.DataFrame(all_metrics_df)
        sns.boxplot(data=df_metrics, x='Metric', y='Value', hue='Type', ax=ax)
        ax.set_title('Distribution of Focus Metrics')
        ax.grid(True, alpha=0.3)
        
        # 4. 평균 attention map 비교
        ax_spam_avg = fig.add_subplot(gs[3, 4:6])
        ax_normal_avg = fig.add_subplot(gs[3, 6:8])
        
        spam_avg = np.mean(spam_images, axis=0)
        normal_avg = np.mean(normal_images, axis=0)
        
        im1 = ax_spam_avg.imshow(spam_avg, cmap='hot', interpolation='bilinear')
        ax_spam_avg.set_title('Average Spam\nAttention Map')
        ax_spam_avg.axis('off')
        plt.colorbar(im1, ax=ax_spam_avg, shrink=0.8)
        
        im2 = ax_normal_avg.imshow(normal_avg, cmap='hot', interpolation='bilinear')
        ax_normal_avg.set_title('Average Normal\nAttention Map')
        ax_normal_avg.axis('off')
        plt.colorbar(im2, ax=ax_normal_avg, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig('image_attention_focus_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return spam_metrics, normal_metrics
    
    def _calculate_gini(self, x):
        """Gini coefficient 계산"""
        sorted_x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(sorted_x)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def run_comprehensive_analysis(self, n_samples=100):
        """포괄적인 분석 실행"""
        print(f"=== 포괄적인 MMTD Attention 분석 ({n_samples} 샘플) ===")
        print(f"고정 시드: 42")
        print(f"언어별 샘플: {n_samples//4}개씩")
        
        # 샘플 생성
        samples = self.generate_comprehensive_samples(n_samples)
        
        # 전체 통계
        total_spam = sum(1 for s in samples if s['is_spam'])
        total_normal = len(samples) - total_spam
        avg_confidence = np.mean([s['confidence'] for s in samples])
        
        print(f"생성된 샘플: 스팸 {total_spam}개, 정상 {total_normal}개")
        print(f"평균 신뢰도: {avg_confidence:.4f}")
        
        # 각 분석 실행
        modality_results = self.analyze_modality_contribution(samples)
        language_patterns = self.analyze_language_attention_patterns(samples)
        self.visualize_cross_modal_attention(samples)
        token_analysis = self.analyze_spam_tokens_by_language(samples)
        focus_results = self.visualize_image_attention_focus(samples)
        
        # 종합 보고서
        self.generate_summary_report(samples, modality_results, language_patterns, 
                                   token_analysis, focus_results)
        
        return {
            'samples': samples,
            'modality_results': modality_results,
            'language_patterns': language_patterns,
            'token_analysis': token_analysis,
            'focus_results': focus_results
        }
    
    def generate_summary_report(self, samples, modality_results, language_patterns, 
                              token_analysis, focus_results):
        """종합 보고서 생성"""
        print("\n" + "="*60)
        print("           종합 분석 보고서")
        print("="*60)
        
        # 1. 전체 성능 요약
        total_samples = len(samples)
        spam_count = sum(1 for s in samples if s['is_spam'])
        normal_count = total_samples - spam_count
        
        print(f"\n1. 데이터셋 요약:")
        print(f"   - 총 샘플: {total_samples}개")
        print(f"   - 스팸: {spam_count}개 ({spam_count/total_samples*100:.1f}%)")
        print(f"   - 정상: {normal_count}개 ({normal_count/total_samples*100:.1f}%)")
        
        languages = list(set(s['language'] for s in samples))
        for lang in languages:
            lang_samples = [s for s in samples if s['language'] == lang]
            lang_spam = sum(1 for s in lang_samples if s['is_spam'])
            print(f"   - {lang.title()}: {len(lang_samples)}개 (스팸 {lang_spam}개)")
        
        # 2. 모달리티 기여도 요약
        print(f"\n2. 모달리티 기여도 분석:")
        spam_text = np.mean(modality_results['spam']['text'])
        spam_image = np.mean(modality_results['spam']['image'])
        spam_fusion = np.mean(modality_results['spam']['fusion'])
        
        normal_text = np.mean(modality_results['normal']['text'])
        normal_image = np.mean(modality_results['normal']['image'])
        normal_fusion = np.mean(modality_results['normal']['fusion'])
        
        print(f"   스팸 분류에서:")
        print(f"   - 텍스트 기여도: {spam_text:.4f}")
        print(f"   - 이미지 기여도: {spam_image:.4f}")
        print(f"   - 융합 기여도: {spam_fusion:.4f}")
        print(f"   정상 분류에서:")
        print(f"   - 텍스트 기여도: {normal_text:.4f}")
        print(f"   - 이미지 기여도: {normal_image:.4f}")
        print(f"   - 융합 기여도: {normal_fusion:.4f}")
        
        # 3. 언어별 패턴 요약
        print(f"\n3. 언어별 패턴 차이:")
        for lang in languages:
            if lang in language_patterns:
                spam_entropy = np.mean([item['text_entropy'] for item in language_patterns[lang]['spam']])
                normal_entropy = np.mean([item['text_entropy'] for item in language_patterns[lang]['normal']])
                print(f"   {lang.title()} - 텍스트 엔트로피: 스팸 {spam_entropy:.3f}, 정상 {normal_entropy:.3f}")
        
        # 4. 주요 스팸 토큰
        print(f"\n4. 언어별 주요 스팸 토큰:")
        for lang in languages:
            if lang in token_analysis:
                # 상위 3개 토큰
                token_stats = {}
                for token, records in token_analysis[lang].items():
                    weights = [r['weight'] for r in records]
                    token_stats[token] = np.mean(weights) * len(records)
                
                top_tokens = sorted(token_stats.items(), key=lambda x: x[1], reverse=True)[:3]
                tokens_str = ", ".join([token for token, _ in top_tokens])
                print(f"   {lang.title()}: {tokens_str}")
        
        # 5. 이미지 집중도 요약
        spam_focus, normal_focus = focus_results
        spam_gini = np.mean([m['gini'] for m in spam_focus])
        normal_gini = np.mean([m['gini'] for m in normal_focus])
        
        print(f"\n5. 이미지 Attention 집중도:")
        print(f"   - 스팸 이미지 집중도 (Gini): {spam_gini:.4f}")
        print(f"   - 정상 이미지 집중도 (Gini): {normal_gini:.4f}")
        print(f"   - 해석: {'스팸이 더 집중적' if spam_gini > normal_gini else '정상이 더 집중적'}")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    analyzer = ComprehensiveAttentionAnalyzer(seed=42)
    results = analyzer.run_comprehensive_analysis(n_samples=100)
    
    print("\n분석 완료! 생성된 파일들:")
    print("- modality_contribution_analysis.png")
    print("- language_attention_patterns.png") 
    print("- cross_modal_attention_visualization.png")
    print("- spam_tokens_by_language.png")
    print("- image_attention_focus_analysis.png") 
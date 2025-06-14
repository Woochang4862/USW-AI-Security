import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from scipy.stats import pearsonr, ttest_ind
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings("ignore")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class ExtendedComprehensiveAnalyzer:
    def __init__(self, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 확장된 언어별 토큰 사전
        self.language_tokens = {
            'english': {
                'spam': ['free', 'money', 'winner', 'urgent', 'click', 'offer', 'deal', 'limited', 'bonus', 
                        'prize', 'cash', 'discount', 'promotion', 'exclusive', 'hurry', 'instant', 'guarantee',
                        'earn', 'save', 'reward', 'million', 'lottery', 'congratulations'],
                'normal': ['hello', 'meeting', 'report', 'update', 'schedule', 'document', 'important',
                          'information', 'project', 'work', 'business', 'conference', 'team', 'please',
                          'thank', 'regards', 'sincerely', 'attachment', 'deadline', 'proposal']
            },
            'korean': {
                'spam': ['무료', '돈', '당첨', '긴급', '클릭', '혜택', '할인', '제한', '보너스',
                        '상금', '현금', '특가', '프로모션', '독점', '서둘러', '즉시', '보장',
                        '벌기', '절약', '리워드', '백만', '복권', '축하'],
                'normal': ['안녕', '회의', '보고서', '업데이트', '일정', '문서', '중요',
                          '정보', '프로젝트', '작업', '비즈니스', '컨퍼런스', '팀', '부탁',
                          '감사', '인사', '진심으로', '첨부', '마감', '제안']
            },
            'chinese': {
                'spam': ['免费', '钱', '中奖', '紧急', '点击', '优惠', '折扣', '限制', '奖金',
                        '奖品', '现金', '特价', '促销', '独家', '赶快', '立即', '保证',
                        '赚钱', '节省', '奖励', '百万', '彩票', '恭喜'],
                'normal': ['你好', '会议', '报告', '更新', '日程', '文档', '重要',
                          '信息', '项目', '工作', '商业', '会议', '团队', '请',
                          '谢谢', '问候', '真诚', '附件', '截止', '提案']
            },
            'japanese': {
                'spam': ['無料', 'お金', '当選', '緊急', 'クリック', '特典', '割引', '限定', 'ボーナス',
                        '賞品', '現金', '特価', 'プロモーション', '独占', '急いで', '即座', '保証',
                        '稼ぐ', '節約', '報酬', '百万', '宝くじ', 'おめでとう'],
                'normal': ['こんにちは', '会議', 'レポート', '更新', 'スケジュール', '文書', '重要',
                          '情報', 'プロジェクト', '仕事', 'ビジネス', '会議', 'チーム', 'お願い',
                          'ありがとう', '挨拶', '心から', '添付', '締切', '提案']
            }
        }
        
    def generate_extended_samples(self, n_samples=500):
        """더 많은 샘플 생성 (500개, 언어별 균등 분포)"""
        samples = []
        languages = list(self.language_tokens.keys())
        samples_per_lang = n_samples // len(languages)
        
        for lang in languages:
            spam_tokens = self.language_tokens[lang]['spam']
            normal_tokens = self.language_tokens[lang]['normal']
            
            for i in range(samples_per_lang):
                # 스팸 여부 결정 (50% 확률)
                is_spam = np.random.choice([True, False])
                
                # 텍스트 토큰 선택
                if is_spam:
                    # 스팸: 스팸 키워드 위주 + 일부 일반 키워드
                    primary_tokens = np.random.choice(spam_tokens, size=np.random.randint(6, 12), replace=True)
                    secondary_tokens = np.random.choice(normal_tokens, size=np.random.randint(2, 6), replace=True)
                    tokens = np.concatenate([primary_tokens, secondary_tokens])
                else:
                    # 정상: 일반 키워드 위주
                    tokens = np.random.choice(normal_tokens, size=np.random.randint(8, 16), replace=True)
                
                # 토큰 순서 섞기
                np.random.shuffle(tokens)
                
                # Attention 가중치 생성 (스팸일 때 더 집중적)
                if is_spam:
                    # 스팸 키워드에 더 높은 가중치
                    attention_weights = []
                    for token in tokens:
                        if token in spam_tokens:
                            attention_weights.append(np.random.exponential(2.0))
                        else:
                            attention_weights.append(np.random.exponential(0.8))
                    attention_weights = np.array(attention_weights)
                    attention_weights = attention_weights / attention_weights.sum()
                else:
                    # 정상: 더 균등한 분포
                    attention_weights = np.random.dirichlet(np.ones(len(tokens)) * 1.5)
                
                sample = {
                    'id': len(samples),
                    'language': lang,
                    'is_spam': is_spam,
                    'tokens': tokens.tolist(),
                    'text_attention': attention_weights,
                    'image_attention': self._generate_image_attention(is_spam),
                    'cross_modal_attention': self._generate_cross_modal_attention(len(tokens), is_spam),
                    'confidence': self._generate_confidence(is_spam, lang),
                    'prediction_logits': self._generate_prediction_logits(is_spam)
                }
                
                samples.append(sample)
        
        return samples
    
    def _generate_image_attention(self, is_spam):
        """개선된 이미지 attention 생성"""
        if is_spam:
            # 스팸: 특정 영역에 강하게 집중 (광고, 로고, 버튼 등)
            # 중심 부분과 모서리에 집중
            attention = np.zeros((8, 8))
            
            # 중심 영역 (로고/텍스트)
            center_region = np.random.exponential(3, (3, 3))
            attention[2:5, 2:5] = center_region
            
            # 모서리 영역 (버튼/링크)
            corner_intensity = np.random.exponential(2)
            corner_x, corner_y = np.random.randint(0, 2, 2) * 7
            attention[corner_x:corner_x+1, corner_y:corner_y+1] = corner_intensity
            
            # 약간의 노이즈 추가
            attention += np.random.exponential(0.1, (8, 8))
            attention = attention / attention.sum()
        else:
            # 정상: 더 분산된 attention (문서 전체)
            attention = np.random.dirichlet(np.ones(64)).reshape(8, 8)
        
        return attention
    
    def _generate_cross_modal_attention(self, n_tokens, is_spam):
        """개선된 크로스 모달 attention 생성"""
        if is_spam:
            # 스팸: 텍스트-이미지 간 강한 연관성
            # 특정 토큰들이 특정 이미지 영역과 강하게 연결
            attention = np.random.exponential(1.2, (n_tokens, 64))
            
            # 일부 토큰에 대해 특별히 강한 연결 생성
            high_attention_tokens = np.random.choice(n_tokens, size=min(3, n_tokens), replace=False)
            for token_idx in high_attention_tokens:
                hot_regions = np.random.choice(64, size=np.random.randint(5, 15), replace=False)
                attention[token_idx, hot_regions] *= np.random.exponential(2, len(hot_regions))
        else:
            # 정상: 약한 연관성
            attention = np.random.exponential(0.6, (n_tokens, 64))
        
        # 정규화
        attention = attention / attention.sum(axis=1, keepdims=True)
        return attention
    
    def _generate_confidence(self, is_spam, language):
        """언어별로 다른 신뢰도 생성"""
        # 언어별 기본 신뢰도 차이 시뮬레이션
        base_confidence = {
            'english': 0.92,
            'korean': 0.88,
            'chinese': 0.85,
            'japanese': 0.87
        }
        
        base = base_confidence[language]
        if is_spam:
            return np.random.uniform(base - 0.1, base + 0.05)
        else:
            return np.random.uniform(base - 0.08, base + 0.03)
    
    def _generate_prediction_logits(self, is_spam):
        """예측 로짓 생성"""
        if is_spam:
            spam_logit = np.random.normal(2.5, 0.8)
            normal_logit = np.random.normal(-1.8, 0.6)
        else:
            spam_logit = np.random.normal(-2.2, 0.7)
            normal_logit = np.random.normal(2.3, 0.5)
        
        return {'spam': spam_logit, 'normal': normal_logit}
    
    def analyze_detailed_modality_contribution(self, samples):
        """A. 상세한 모달리티별 기여도 분석"""
        print("=== A. 상세한 모달리티별 기여도 분석 ===")
        
        # 언어별, 클래스별 분석
        results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for sample in samples:
            lang = sample['language']
            category = 'spam' if sample['is_spam'] else 'normal'
            
            # 각 모달리티의 기여도 계산
            text_contrib = np.max(sample['text_attention'])  # 최대 attention
            text_entropy = -np.sum(sample['text_attention'] * np.log(sample['text_attention'] + 1e-8))
            
            image_contrib = np.max(sample['image_attention'])
            image_gini = self._calculate_gini(sample['image_attention'].flatten())
            
            fusion_contrib = np.mean(sample['cross_modal_attention'])
            fusion_variance = np.var(sample['cross_modal_attention'])
            
            results[lang][category]['text_max'].append(text_contrib)
            results[lang][category]['text_entropy'].append(text_entropy)
            results[lang][category]['image_max'].append(image_contrib)
            results[lang][category]['image_gini'].append(image_gini)
            results[lang][category]['fusion_mean'].append(fusion_contrib)
            results[lang][category]['fusion_var'].append(fusion_variance)
        
        # 시각화
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        
        languages = list(results.keys())
        metrics = ['text_max', 'text_entropy', 'image_max', 'image_gini', 'fusion_mean', 'fusion_var']
        titles = ['Text Max Attention', 'Text Entropy', 'Image Max Attention', 
                 'Image Gini Coefficient', 'Cross-Modal Mean', 'Cross-Modal Variance']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            # 데이터 준비
            plot_data = []
            for lang in languages:
                for category in ['spam', 'normal']:
                    values = results[lang][category][metric]
                    for val in values:
                        plot_data.append({
                            'Language': lang.title(),
                            'Type': category.title(),
                            'Value': val,
                            'Metric': metric
                        })
            
            df_plot = pd.DataFrame(plot_data)
            
            # Box plot
            sns.boxplot(data=df_plot, x='Language', y='Value', hue='Type', ax=ax)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # 통계적 유의성 검정
            for lang in languages:
                spam_vals = results[lang]['spam'][metric]
                normal_vals = results[lang]['normal'][metric]
                if len(spam_vals) > 0 and len(normal_vals) > 0:
                    t_stat, p_val = ttest_ind(spam_vals, normal_vals)
                    if p_val < 0.05:
                        ax.text(0.02, 0.98, f'{lang}: p={p_val:.3f}*', 
                               transform=ax.transAxes, va='top', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('detailed_modality_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results
    
    def analyze_language_specific_patterns(self, samples):
        """B. 언어별 특화 패턴 분석"""
        print("\n=== B. 언어별 특화 패턴 분석 ===")
        
        # 언어별 스팸 탐지 난이도 분석
        lang_performance = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})
        
        for sample in samples:
            lang = sample['language']
            is_spam = sample['is_spam']
            confidence = sample['confidence']
            
            # 예측 성공 여부 (높은 신뢰도 = 정확한 예측으로 가정)
            predicted_correctly = confidence > 0.8
            
            lang_performance[lang]['total'] += 1
            if predicted_correctly:
                lang_performance[lang]['correct'] += 1
            lang_performance[lang]['confidences'].append(confidence)
        
        # 언어별 토큰 중요도 분석
        token_importance = defaultdict(lambda: defaultdict(lambda: {'weights': [], 'confidences': []}))
        
        for sample in samples:
            if not sample['is_spam']:  # 스팸만 분석
                continue
                
            lang = sample['language']
            tokens = sample['tokens']
            weights = sample['text_attention']
            confidence = sample['confidence']
            
            spam_tokens = self.language_tokens[lang]['spam']
            
            for token, weight in zip(tokens, weights):
                if token in spam_tokens:
                    token_importance[lang][token]['weights'].append(weight)
                    token_importance[lang][token]['confidences'].append(confidence)
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. 언어별 성능
        ax = axes[0, 0]
        languages = list(lang_performance.keys())
        accuracies = [lang_performance[lang]['correct'] / lang_performance[lang]['total'] 
                     for lang in languages]
        avg_confidences = [np.mean(lang_performance[lang]['confidences']) 
                          for lang in languages]
        
        bars = ax.bar(range(len(languages)), accuracies, alpha=0.7, color='skyblue')
        ax.set_xlabel('Language')
        ax.set_ylabel('Accuracy')
        ax.set_title('Detection Accuracy by Language')
        ax.set_xticks(range(len(languages)))
        ax.set_xticklabels([lang.title() for lang in languages])
        
        # 막대 위에 신뢰도 표시
        for i, (bar, conf) in enumerate(zip(bars, avg_confidences)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'Conf: {conf:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.grid(True, alpha=0.3)
        
        # 2. 언어별 신뢰도 분포
        ax = axes[0, 1]
        conf_data = []
        for lang in languages:
            for conf in lang_performance[lang]['confidences']:
                conf_data.append({'Language': lang.title(), 'Confidence': conf})
        
        df_conf = pd.DataFrame(conf_data)
        sns.violinplot(data=df_conf, x='Language', y='Confidence', ax=ax)
        ax.set_title('Confidence Distribution by Language')
        ax.grid(True, alpha=0.3)
        
        # 3-4. 언어별 상위 스팸 토큰
        for idx, lang in enumerate(languages[:2]):
            ax = axes[1, idx]
            
            # 토큰별 평균 중요도 계산
            token_scores = {}
            for token, data in token_importance[lang].items():
                if len(data['weights']) > 0:
                    avg_weight = np.mean(data['weights'])
                    avg_conf = np.mean(data['confidences'])
                    frequency = len(data['weights'])
                    token_scores[token] = avg_weight * avg_conf * np.log(frequency + 1)
            
            # 상위 토큰 선택
            top_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            
            if top_tokens:
                tokens, scores = zip(*top_tokens)
                bars = ax.barh(range(len(tokens)), scores, alpha=0.7)
                ax.set_xlabel('Importance Score')
                ax.set_title(f'Top Spam Tokens - {lang.title()}')
                ax.set_yticks(range(len(tokens)))
                ax.set_yticklabels(tokens)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('language_specific_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 통계 출력
        print("언어별 성능 요약:")
        for lang in languages:
            acc = lang_performance[lang]['correct'] / lang_performance[lang]['total']
            conf = np.mean(lang_performance[lang]['confidences'])
            print(f"  {lang.title()}: 정확도 {acc:.3f}, 평균 신뢰도 {conf:.3f}")
        
        return lang_performance, token_importance
    
    def _calculate_gini(self, x):
        """Gini coefficient 계산"""
        sorted_x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(sorted_x)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def run_extended_analysis(self, n_samples=500):
        """확장된 포괄적 분석 실행"""
        print(f"=== 확장된 MMTD Attention 분석 ({n_samples} 샘플) ===")
        print(f"고정 시드: 42")
        print(f"언어별 샘플: {n_samples//4}개씩")
        
        # 샘플 생성
        samples = self.generate_extended_samples(n_samples)
        
        # 전체 통계
        total_spam = sum(1 for s in samples if s['is_spam'])
        total_normal = len(samples) - total_spam
        avg_confidence = np.mean([s['confidence'] for s in samples])
        
        print(f"생성된 샘플: 스팸 {total_spam}개, 정상 {total_normal}개")
        print(f"평균 신뢰도: {avg_confidence:.4f}")
        
        # 분석 실행
        modality_results = self.analyze_detailed_modality_contribution(samples)
        language_results = self.analyze_language_specific_patterns(samples)
        
        print("\n=== 확장 분석 완료 ===")
        return {
            'samples': samples,
            'modality_results': modality_results,
            'language_results': language_results
        }


if __name__ == "__main__":
    analyzer = ExtendedComprehensiveAnalyzer(seed=42)
    results = analyzer.run_extended_analysis(n_samples=500)
    
    print("\n확장 분석 완료! 생성된 파일들:")
    print("- detailed_modality_analysis.png")
    print("- language_specific_patterns.png") 
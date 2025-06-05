"""
Attention-based Interpretability Analysis Experiment
MMTD 모델의 Attention 기반 해석가능성 분석 실험
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
from tqdm import tqdm
import argparse
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Custom imports
from src.analysis.attention_analyzer import AttentionAnalyzer
from src.analysis.attention_visualizer import AttentionVisualizer
from src.models.interpretable_mmtd import InterpretableMMTD
from src.data_loader import EDPDataModule
from transformers import AutoTokenizer
from torchvision import transforms

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/attention_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AttentionAnalysisExperiment:
    """Attention 기반 해석가능성 분석 실험 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 실험 설정 딕셔너리
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 출력 디렉토리 생성
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 및 분석기 초기화
        self.model = None
        self.tokenizer = None
        self.analyzer = None
        self.visualizer = None
        self.data_module = None
        
        logger.info(f"🚀 Attention 분석 실험 초기화 (디바이스: {self.device})")
    
    def load_model_and_tokenizer(self):
        """모델과 토크나이저 로딩"""
        
        logger.info("📥 모델 및 토크나이저 로딩 중...")
        
        try:
            # 모델 로딩
            model_path = self.config['model_path']
            self.model = InterpretableMMTD.load_from_checkpoint(
                model_path,
                map_location=self.device
            )
            self.model.to(self.device)
            self.model.eval()
            
            # 토크나이저 로딩
            tokenizer_name = self.config.get('tokenizer_name', 'bert-base-uncased')
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # 분석기 및 시각화 도구 초기화
            self.analyzer = AttentionAnalyzer(self.model, self.tokenizer, self.device)
            self.visualizer = AttentionVisualizer()
            
            logger.info("✅ 모델 및 도구 로딩 완료")
            
        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {e}")
            raise
    
    def load_data(self):
        """데이터 로딩"""
        
        logger.info("📊 데이터 로딩 중...")
        
        try:
            self.data_module = EDPDataModule(
                data_dir=self.config['data_dir'],
                batch_size=1,  # 분석을 위해 배치 크기 1
                num_workers=0
            )
            self.data_module.setup()
            
            logger.info("✅ 데이터 로딩 완료")
            
        except Exception as e:
            logger.error(f"❌ 데이터 로딩 실패: {e}")
            raise
    
    def analyze_single_sample(self, text: str, image: torch.Tensor, 
                            label: int, sample_id: str) -> Dict[str, Any]:
        """단일 샘플에 대한 attention 분석"""
        
        logger.info(f"🔍 샘플 {sample_id} 분석 중...")
        
        try:
            # Attention 분석 수행
            explanation = self.analyzer.explain_prediction(
                text=text,
                image=image,
                return_attention_maps=True
            )
            
            # 실제 라벨 정보 추가
            explanation['ground_truth'] = {
                'label': 'SPAM' if label == 1 else 'HAM',
                'class': int(label)
            }
            
            # 예측 정확성 계산
            predicted_class = explanation['prediction']['class']
            is_correct = (predicted_class == label)
            explanation['prediction']['is_correct'] = is_correct
            explanation['prediction']['accuracy'] = 1.0 if is_correct else 0.0
            
            return explanation
            
        except Exception as e:
            logger.error(f"❌ 샘플 {sample_id} 분석 실패: {e}")
            return None
    
    def create_visualizations(self, explanation: Dict[str, Any], 
                            image: torch.Tensor, sample_id: str):
        """단일 샘플에 대한 시각화 생성"""
        
        try:
            vis_dir = self.output_dir / 'visualizations' / sample_id
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. 텍스트 attention 시각화
            if explanation['text_analysis']['important_tokens']:
                text_fig = self.visualizer.visualize_text_attention(
                    tokens=explanation['text_analysis']['tokens'],
                    token_importance=explanation['text_analysis']['important_tokens'],
                    title=f"텍스트 Attention 분석 - {sample_id}",
                    save_path=str(vis_dir / 'text_attention.png')
                )
                
            # 2. 이미지 attention 시각화
            if explanation['image_analysis']['important_patches']:
                image_fig = self.visualizer.visualize_image_attention(
                    image=image,
                    patch_importance=explanation['image_analysis']['important_patches'],
                    title=f"이미지 Attention 분석 - {sample_id}",
                    save_path=str(vis_dir / 'image_attention.png')
                )
            
            # 3. Cross-modal attention 시각화
            if 'attention_maps' in explanation and explanation['attention_maps']['cross_modal_attention']:
                cross_modal_fig = self.visualizer.visualize_cross_modal_attention(
                    cross_modal_attention=explanation['attention_maps']['cross_modal_attention'],
                    tokens=explanation['text_analysis']['tokens'],
                    title=f"Cross-Modal Attention 분석 - {sample_id}",
                    save_path=str(vis_dir / 'cross_modal_attention.png')
                )
            
            # 4. 종합 분석 시각화
            comprehensive_fig = self.visualizer.visualize_comprehensive_explanation(
                explanation=explanation,
                image=image,
                title=f"종합 Attention 분석 - {sample_id}",
                save_path=str(vis_dir / 'comprehensive_analysis.png')
            )
            
            logger.info(f"📊 샘플 {sample_id} 시각화 완료")
            
        except Exception as e:
            logger.error(f"❌ 샘플 {sample_id} 시각화 실패: {e}")
    
    def run_batch_analysis(self, num_samples: int = 50) -> List[Dict[str, Any]]:
        """배치 분석 실행"""
        
        logger.info(f"🔍 {num_samples}개 샘플 배치 분석 시작")
        
        # 테스트 데이터로더 가져오기
        test_loader = self.data_module.test_dataloader()
        
        explanations = []
        sample_count = 0
        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Attention 분석 진행중")):
            if sample_count >= num_samples:
                break
            
            # 배치에서 데이터 추출
            text = batch['text'][0] if isinstance(batch['text'], list) else batch['text'][0].item()
            image = batch['image'][0]
            label = batch['label'][0].item()
            
            sample_id = f"sample_{batch_idx:04d}"
            
            # 단일 샘플 분석
            explanation = self.analyze_single_sample(text, image, label, sample_id)
            
            if explanation is not None:
                explanations.append(explanation)
                
                # 시각화 생성 (처음 10개 샘플만)
                if sample_count < 10:
                    self.create_visualizations(explanation, image, sample_id)
                
                # 개별 결과 저장
                self.analyzer.save_explanation(
                    explanation,
                    str(self.output_dir / f'{sample_id}_explanation.json'),
                    include_attention_maps=False
                )
                
                sample_count += 1
            
            # 메모리 정리
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info(f"✅ 배치 분석 완료: {len(explanations)}개 샘플 처리")
        return explanations
    
    def analyze_attention_patterns(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Attention 패턴 통계 분석"""
        
        logger.info("📈 Attention 패턴 분석 중...")
        
        analysis = {
            'total_samples': len(explanations),
            'accuracy': sum(exp['prediction']['accuracy'] for exp in explanations) / len(explanations),
            'spam_samples': sum(1 for exp in explanations if exp['ground_truth']['class'] == 1),
            'ham_samples': sum(1 for exp in explanations if exp['ground_truth']['class'] == 0),
            'text_importance_stats': {},
            'image_importance_stats': {},
            'modality_balance_stats': {},
            'error_analysis': {}
        }
        
        # 텍스트 중요도 분석
        text_importances = []
        important_tokens_all = []
        
        for exp in explanations:
            if exp['text_analysis']['important_tokens']:
                token_scores = [t['combined_importance'] for t in exp['text_analysis']['important_tokens']]
                text_importances.extend(token_scores)
                important_tokens_all.extend([t['token'] for t in exp['text_analysis']['important_tokens'][:5]])
        
        if text_importances:
            analysis['text_importance_stats'] = {
                'mean': np.mean(text_importances),
                'std': np.std(text_importances),
                'min': np.min(text_importances),
                'max': np.max(text_importances),
                'most_common_tokens': pd.Series(important_tokens_all).value_counts().head(10).to_dict()
            }
        
        # 이미지 중요도 분석
        image_importances = []
        important_patches_all = []
        
        for exp in explanations:
            if exp['image_analysis']['important_patches']:
                patch_scores = [p['combined_importance'] for p in exp['image_analysis']['important_patches']]
                image_importances.extend(patch_scores)
                important_patches_all.extend([str(p['coordinates']) for p in exp['image_analysis']['important_patches'][:5]])
        
        if image_importances:
            analysis['image_importance_stats'] = {
                'mean': np.mean(image_importances),
                'std': np.std(image_importances),
                'min': np.min(image_importances),
                'max': np.max(image_importances),
                'most_common_patches': pd.Series(important_patches_all).value_counts().head(10).to_dict()
            }
        
        # 모달리티 균형 분석
        modality_balances = [exp['cross_modal_analysis']['modality_balance'] for exp in explanations]
        
        analysis['modality_balance_stats'] = {
            'mean_balance': np.mean(modality_balances),
            'std_balance': np.std(modality_balances),
            'text_dominant_samples': sum(1 for b in modality_balances if b < 0.4),
            'image_dominant_samples': sum(1 for b in modality_balances if b > 0.6),
            'balanced_samples': sum(1 for b in modality_balances if 0.4 <= b <= 0.6)
        }
        
        # 오류 분석
        correct_predictions = [exp for exp in explanations if exp['prediction']['is_correct']]
        incorrect_predictions = [exp for exp in explanations if not exp['prediction']['is_correct']]
        
        analysis['error_analysis'] = {
            'correct_count': len(correct_predictions),
            'incorrect_count': len(incorrect_predictions),
            'false_positives': sum(1 for exp in incorrect_predictions 
                                 if exp['prediction']['class'] == 1 and exp['ground_truth']['class'] == 0),
            'false_negatives': sum(1 for exp in incorrect_predictions 
                                 if exp['prediction']['class'] == 0 and exp['ground_truth']['class'] == 1)
        }
        
        # 정확/오답 예측의 attention 패턴 차이
        if correct_predictions and incorrect_predictions:
            correct_modality_balance = np.mean([exp['cross_modal_analysis']['modality_balance'] 
                                              for exp in correct_predictions])
            incorrect_modality_balance = np.mean([exp['cross_modal_analysis']['modality_balance'] 
                                                for exp in incorrect_predictions])
            
            analysis['error_analysis']['modality_balance_difference'] = {
                'correct_mean': correct_modality_balance,
                'incorrect_mean': incorrect_modality_balance,
                'difference': abs(correct_modality_balance - incorrect_modality_balance)
            }
        
        logger.info("✅ Attention 패턴 분석 완료")
        return analysis
    
    def create_summary_report(self, explanations: List[Dict[str, Any]], 
                            analysis: Dict[str, Any]):
        """요약 보고서 생성"""
        
        logger.info("📋 요약 보고서 생성 중...")
        
        # 대표 샘플들 선택 (정확한 예측, 틀린 예측, 다양한 모달리티 균형)
        representative_samples = []
        
        # 정확한 예측 중 신뢰도 높은 샘플
        correct_samples = [exp for exp in explanations if exp['prediction']['is_correct']]
        if correct_samples:
            best_correct = max(correct_samples, key=lambda x: x['prediction']['confidence'])
            representative_samples.append(best_correct)
        
        # 틀린 예측 중 신뢰도 높은 샘플 (흥미로운 케이스)
        incorrect_samples = [exp for exp in explanations if not exp['prediction']['is_correct']]
        if incorrect_samples:
            worst_incorrect = max(incorrect_samples, key=lambda x: x['prediction']['confidence'])
            representative_samples.append(worst_incorrect)
        
        # 텍스트 중심 샘플
        text_dominant = [exp for exp in explanations if exp['cross_modal_analysis']['modality_balance'] < 0.3]
        if text_dominant:
            representative_samples.append(text_dominant[0])
        
        # 이미지 중심 샘플
        image_dominant = [exp for exp in explanations if exp['cross_modal_analysis']['modality_balance'] > 0.7]
        if image_dominant:
            representative_samples.append(image_dominant[0])
        
        # 요약 시각화 생성
        if representative_samples:
            self.visualizer.create_attention_summary_report(
                representative_samples,
                save_path=str(self.output_dir / 'attention_summary_report.png')
            )
        
        # 분석 결과 저장
        with open(self.output_dir / 'attention_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # 상세 결과 저장
        detailed_results = {
            'config': self.config,
            'analysis': analysis,
            'representative_samples': representative_samples
        }
        
        with open(self.output_dir / 'detailed_attention_results.json', 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ 요약 보고서 생성 완료")
    
    def run_experiment(self):
        """전체 실험 실행"""
        
        logger.info("🚀 Attention 분석 실험 시작")
        
        try:
            # 1. 모델 및 데이터 로딩
            self.load_model_and_tokenizer()
            self.load_data()
            
            # 2. 배치 분석 실행
            explanations = self.run_batch_analysis(self.config.get('num_samples', 50))
            
            # 3. 패턴 분석
            analysis = self.analyze_attention_patterns(explanations)
            
            # 4. 요약 보고서 생성
            self.create_summary_report(explanations, analysis)
            
            # 5. 결과 출력
            self.print_summary(analysis)
            
            logger.info("✅ Attention 분석 실험 완료")
            
        except Exception as e:
            logger.error(f"❌ 실험 실행 실패: {e}")
            raise
    
    def print_summary(self, analysis: Dict[str, Any]):
        """결과 요약 출력"""
        
        print("\n" + "="*80)
        print("🔍 ATTENTION 기반 해석가능성 분석 결과 요약")
        print("="*80)
        
        print(f"\n📊 전체 통계:")
        print(f"  • 분석 샘플 수: {analysis['total_samples']}개")
        print(f"  • 전체 정확도: {analysis['accuracy']*100:.2f}%")
        print(f"  • 스팸 샘플: {analysis['spam_samples']}개")
        print(f"  • 햄 샘플: {analysis['ham_samples']}개")
        
        if analysis['text_importance_stats']:
            print(f"\n📝 텍스트 Attention 통계:")
            stats = analysis['text_importance_stats']
            print(f"  • 평균 중요도: {stats['mean']:.4f}")
            print(f"  • 표준편차: {stats['std']:.4f}")
            print(f"  • 최대값: {stats['max']:.4f}")
            
            print(f"  • 자주 등장하는 중요 토큰:")
            for token, count in list(stats['most_common_tokens'].items())[:5]:
                print(f"    - {token}: {count}회")
        
        if analysis['image_importance_stats']:
            print(f"\n🖼️ 이미지 Attention 통계:")
            stats = analysis['image_importance_stats']
            print(f"  • 평균 중요도: {stats['mean']:.4f}")
            print(f"  • 표준편차: {stats['std']:.4f}")
            print(f"  • 최대값: {stats['max']:.4f}")
        
        print(f"\n⚖️ 모달리티 균형 분석:")
        balance_stats = analysis['modality_balance_stats']
        print(f"  • 평균 균형도: {balance_stats['mean_balance']:.3f} (0=텍스트 중심, 1=이미지 중심)")
        print(f"  • 텍스트 중심 샘플: {balance_stats['text_dominant_samples']}개")
        print(f"  • 이미지 중심 샘플: {balance_stats['image_dominant_samples']}개")
        print(f"  • 균형 잡힌 샘플: {balance_stats['balanced_samples']}개")
        
        print(f"\n❌ 오류 분석:")
        error_stats = analysis['error_analysis']
        print(f"  • 정확한 예측: {error_stats['correct_count']}개")
        print(f"  • 틀린 예측: {error_stats['incorrect_count']}개")
        print(f"  • False Positive (햄→스팸): {error_stats['false_positives']}개")
        print(f"  • False Negative (스팸→햄): {error_stats['false_negatives']}개")
        
        if 'modality_balance_difference' in error_stats:
            diff_stats = error_stats['modality_balance_difference']
            print(f"  • 정확한 예측의 모달리티 균형: {diff_stats['correct_mean']:.3f}")
            print(f"  • 틀린 예측의 모달리티 균형: {diff_stats['incorrect_mean']:.3f}")
            print(f"  • 차이: {diff_stats['difference']:.3f}")
        
        print(f"\n💾 결과 저장 위치: {self.output_dir}")
        print("="*80)


def main():
    """메인 실행 함수"""
    
    parser = argparse.ArgumentParser(description='Attention 기반 해석가능성 분석 실험')
    parser.add_argument('--model_path', type=str, required=True,
                       help='학습된 모델 체크포인트 경로')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='데이터셋 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, 
                       default=f'outputs/attention_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                       help='결과 저장 디렉토리')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='분석할 샘플 수')
    parser.add_argument('--device', type=str, default='auto',
                       help='사용할 디바이스 (cuda/cpu/auto)')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased',
                       help='사용할 토크나이저')
    
    args = parser.parse_args()
    
    # 디바이스 자동 설정
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # 실험 설정
    config = {
        'model_path': args.model_path,
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'num_samples': args.num_samples,
        'device': device,
        'tokenizer_name': args.tokenizer_name
    }
    
    # 실험 실행
    experiment = AttentionAnalysisExperiment(config)
    experiment.run_experiment()


if __name__ == "__main__":
    main() 
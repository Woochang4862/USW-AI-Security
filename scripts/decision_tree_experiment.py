#!/usr/bin/env python3
"""
Decision Tree를 이용한 해석가능한 MMTD 실험
규칙 기반 해석성과 높은 성능을 목표로 함
"""

import os
import sys
import torch
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.interpretable_mmtd import create_interpretable_mmtd
from models.mmtd_data_loader import create_mmtd_data_module

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DecisionTreeExperimentRunner:
    """Decision Tree 기반 해석가능한 MMTD 실험"""
    
    def __init__(self, output_dir: str = None):
        self.device = torch.device("mps" if torch.backends.mps.is_available() 
                                  else "cuda" if torch.cuda.is_available() 
                                  else "cpu")
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or f"outputs/decision_tree_{timestamp}"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🌳 Decision Tree 해석가능한 MMTD 실험 시작")
        logger.info(f"💻 Device: {self.device}")
        logger.info(f"📁 Output: {self.output_dir}")
    
    def create_decision_tree_model(self):
        """Decision Tree 기반 모델 생성"""
        logger.info("🌳 Decision Tree 모델 생성 중...")
        
        # Create model with Decision Tree classifier
        model = create_interpretable_mmtd(
            classifier_type="decision_tree",
            classifier_config={
                'max_depth': 15,           # 적절한 깊이로 과적합 방지
                'min_samples_split': 20,   # 분할을 위한 최소 샘플 수
                'min_samples_leaf': 10,    # 리프 노드 최소 샘플 수
                'max_features': 'sqrt',    # 특성 선택 방법
                'criterion': 'gini',       # 분할 기준
                'random_state': 42         # 재현성을 위한 시드
            },
            device=self.device,
            checkpoint_path="MMTD/checkpoints/fold1/checkpoint-939/pytorch_model.bin"
        )
        
        logger.info(f"✅ Decision Tree 모델 생성 완료")
        logger.info(f"📊 총 파라미터: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, model.get_model_summary()
    
    def prepare_data(self):
        """데이터셋 준비"""
        logger.info("📚 데이터셋 로딩 중...")
        
        data_module = create_mmtd_data_module(
            csv_path="MMTD/DATA/email_data/EDP.csv",
            data_path="MMTD/DATA/email_data/pics",
            batch_size=64,      # Decision Tree는 더 큰 배치 처리 가능
            max_samples=None,   # 전체 데이터셋 사용
            num_workers=4,
            test_split=0.2,
            val_split=0.1
        )
        
        logger.info("✅ 데이터셋 로딩 완료")
        logger.info(f"📊 훈련 샘플: {len(data_module.train_dataset):,}")
        logger.info(f"📊 검증 샘플: {len(data_module.val_dataset):,}")
        logger.info(f"📊 테스트 샘플: {len(data_module.test_dataset):,}")
        
        return data_module
    
    def run_decision_tree_training(self):
        """Decision Tree 훈련 실행"""
        logger.info("🚀 Decision Tree 훈련 시작")
        
        # Create model and data
        model, model_summary = self.create_decision_tree_model()
        data_module = self.prepare_data()
        
        # Decision Tree 전용 훈련: 2단계 접근법
        # 1단계: 특성 추출 및 Decision Tree 훈련
        # 2단계: 성능 평가
        
        logger.info("🔄 1단계: Decision Tree 데이터 수집 및 훈련")
        
        # Freeze backbone for feature extraction
        model.eval()
        for name, param in model.named_parameters():
            if 'interpretable_classifier' not in name:
                param.requires_grad = False
        
        # 훈련 설정
        num_epochs = 3  # Decision Tree는 빠른 데이터 수집만 필요
        training_history = []
        
        start_time = time.time()
        
        # 전체 데이터를 한 번에 처리하여 Decision Tree 훈련
        train_loader = data_module.train_dataloader()
        
        for epoch in range(num_epochs):
            logger.info(f"📚 에포크 {epoch + 1}/{num_epochs} - 데이터 수집 및 Decision Tree 훈련")
            
            train_correct = 0
            train_total = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Extract features only (no gradient computation)
                with torch.no_grad():
                    features = model.extract_features(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        pixel_values=batch['pixel_values']
                    )
                
                # Train Decision Tree incrementally
                model.interpretable_classifier.fit_incremental(features, batch['labels'])
                
                # Get predictions after fitting
                with torch.no_grad():
                    logits = model.interpretable_classifier(features)
                    predictions = torch.argmax(logits, dim=1)
                    train_correct += (predictions == batch['labels']).sum().item()
                    train_total += batch['labels'].size(0)
                
                # 로깅
                if batch_idx % 80 == 0:
                    current_acc = train_correct / train_total if train_total > 0 else 0
                    logger.info(f"  배치 {batch_idx}: 정확도 {current_acc:.4f}")
            
            # Validation phase
            logger.info("🔍 검증 단계")
            val_correct = 0
            val_total = 0
            
            val_loader = data_module.val_dataloader()
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Extract features
                    features = model.extract_features(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        pixel_values=batch['pixel_values']
                    )
                    
                    # Get predictions
                    logits = model.interpretable_classifier(features)
                    predictions = torch.argmax(logits, dim=1)
                    val_correct += (predictions == batch['labels']).sum().item()
                    val_total += batch['labels'].size(0)
            
            # Calculate metrics
            train_accuracy = train_correct / train_total
            val_accuracy = val_correct / val_total
            
            # Log results
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': 0.0,  # Decision Tree doesn't use loss
                'train_accuracy': train_accuracy,
                'val_loss': 0.0,    # Decision Tree doesn't use loss
                'val_accuracy': val_accuracy,
                'learning_rate': 0.0  # No gradient-based learning
            }
            training_history.append(epoch_results)
            
            logger.info(f"  훈련 정확도: {train_accuracy:.4f}, 검증 정확도: {val_accuracy:.4f}")
            
            # Decision Tree는 데이터가 충분하면 일찍 종료 가능
            if epoch >= 1 and val_accuracy > 0.8:  # 기본 성능이 나오면 종료
                logger.info(f"✅ Decision Tree 충분히 훈련됨 (검증 정확도: {val_accuracy:.4f})")
                break
        
        training_time = time.time() - start_time
        
        # Final validation
        best_val_accuracy = val_accuracy
        
        # 2단계: 최종 성능 검증
        logger.info("🔄 2단계: 최종 Decision Tree 성능 검증")
        
        final_val_correct = 0
        final_val_total = 0
        
        val_loader = data_module.val_dataloader()
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Extract features
                features = model.extract_features(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch['pixel_values']
                )
                
                # Get predictions
                logits = model.interpretable_classifier(features)
                predictions = torch.argmax(logits, dim=1)
                final_val_correct += (predictions == batch['labels']).sum().item()
                final_val_total += batch['labels'].size(0)
        
        final_val_accuracy = final_val_correct / final_val_total
        best_val_accuracy = max(best_val_accuracy, final_val_accuracy)
        
        training_results = {
            'training_history': training_history,
            'best_val_accuracy': best_val_accuracy,
            'final_val_accuracy': final_val_accuracy,
            'total_training_time': training_time,
            'epochs_trained': epoch + 1,
            'early_stopped': False,
            'backbone_parameters': 0,  # Backbone은 freeze됨
            'decision_tree_fitted': hasattr(model.interpretable_classifier, 'is_fitted') and model.interpretable_classifier.is_fitted,
            'training_method': 'feature_extraction_and_decision_tree_fitting'
        }
        
        logger.info(f"⏱️ 훈련 완료: {training_time:.2f}초")
        logger.info(f"📈 최고 검증 정확도: {best_val_accuracy:.4f}")
        logger.info(f"📊 최종 검증 정확도: {final_val_accuracy:.4f}")
        logger.info(f"🌳 Decision Tree 피팅 상태: {training_results['decision_tree_fitted']}")
        
        return model, training_results, model_summary
    
    def run_comprehensive_evaluation(self, model, data_module):
        """포괄적 평가 실행"""
        logger.info("📊 포괄적 평가 시작")
        
        # Test evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        test_loader = data_module.test_dataloader()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss
                
                test_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                test_correct += (predictions == batch['labels']).sum().item()
                test_total += batch['labels'].size(0)
                
                # Get probabilities
                probs = torch.softmax(outputs.logits, dim=1)
                all_probabilities.extend(probs.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                
                if batch_idx % 40 == 0:
                    logger.info(f"  평가 배치 {batch_idx} 완료")
        
        # Calculate comprehensive metrics
        from sklearn.metrics import (
            precision_recall_fscore_support, confusion_matrix, 
            roc_auc_score, classification_report
        )
        import numpy as np
        
        accuracy = test_correct / test_total
        avg_loss = test_loss / len(test_loader)
        
        # Get probabilities for class 1 (spam)
        probs_class1 = np.array(all_probabilities)[:, 1]
        
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        auc_score = roc_auc_score(all_labels, probs_class1)
        
        test_results = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'auc_score': auc_score,
            'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist(),
            'classification_report': classification_report(all_labels, all_predictions, output_dict=True)
        }
        
        logger.info(f"🎯 테스트 정확도: {accuracy:.4f}")
        logger.info(f"📉 테스트 손실: {avg_loss:.4f}")
        logger.info(f"🔍 F1 스코어: {f1_score:.4f}")
        logger.info(f"📈 AUC 스코어: {auc_score:.4f}")
        
        return test_results
    
    def analyze_decision_tree_interpretability(self, model):
        """Decision Tree 해석성 분석"""
        logger.info("🌳 Decision Tree 해석성 분석 시작")
        
        try:
            # Get Decision Tree specific interpretability
            interpretability_results = {}
            
            # 1. Feature importance from Decision Tree
            if hasattr(model, 'get_decision_tree_rules'):
                tree_rules = model.get_decision_tree_rules()
                interpretability_results['tree_rules'] = tree_rules
                logger.info(f"📋 추출된 규칙 수: {len(tree_rules) if tree_rules else 0}")
            
            # 2. Feature importance scores
            if hasattr(model, 'get_feature_importance'):
                feature_importance = model.get_feature_importance()
                if feature_importance is not None:
                    importance_stats = {
                        'mean_importance': float(torch.mean(torch.abs(feature_importance))),
                        'max_importance': float(torch.max(torch.abs(feature_importance))),
                        'std_importance': float(torch.std(feature_importance)),
                        'top_10_features': torch.topk(torch.abs(feature_importance), k=10).indices.tolist()
                    }
                    interpretability_results['importance_stats'] = importance_stats
            
            # 3. Tree structure analysis
            if hasattr(model, 'get_tree_structure'):
                tree_structure = model.get_tree_structure()
                interpretability_results['tree_structure'] = tree_structure
                
            logger.info("✅ Decision Tree 해석성 분석 완료")
            return interpretability_results
            
        except Exception as e:
            logger.warning(f"⚠️ 해석성 분석 실패: {e}")
            return {}
    
    def run_full_decision_tree_experiment(self):
        """전체 Decision Tree 실험 실행"""
        try:
            logger.info("🚀 Decision Tree 전체 실험 시작")
            
            # 1. Training
            trained_model, training_results, model_summary = self.run_decision_tree_training()
            
            # 2. Data preparation for evaluation  
            data_module = self.prepare_data()
            
            # 3. Comprehensive evaluation
            test_results = self.run_comprehensive_evaluation(trained_model, data_module)
            
            # 4. Decision Tree specific interpretability analysis
            interpretability_analysis = self.analyze_decision_tree_interpretability(trained_model)
            
            # 5. Save results
            final_results = {
                'experiment_info': {
                    'timestamp': datetime.now().isoformat(),
                    'device': str(self.device),
                    'model_type': 'InterpretableMMTD_DecisionTree',
                    'classifier_type': 'decision_tree',
                    'note': 'Decision Tree experiment for rule-based interpretability',
                    'advantages': [
                        'Rule-based interpretability',
                        'No black-box components in classifier',
                        'Human-readable decision paths',
                        'Feature importance ranking',
                        'Fast inference',
                        'No gradient computation needed for classifier'
                    ]
                },
                'model_summary': model_summary,
                'training_results': training_results,
                'test_results': test_results,
                'interpretability_analysis': interpretability_analysis
            }
            
            # Save results
            results_path = os.path.join(self.output_dir, "decision_tree_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 결과 저장 완료: {results_path}")
            
            # Print summary
            self.print_decision_tree_summary(final_results)
            
            logger.info("✅ Decision Tree 실험 완료!")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Decision Tree 실험 실패: {e}")
            raise
    
    def print_decision_tree_summary(self, results):
        """Decision Tree 결과 요약 출력"""
        print("\n" + "="*70)
        print("🌳 Decision Tree 해석가능한 MMTD 모델 실험 결과")
        print("="*70)
        
        # Advantages
        advantages = results['experiment_info']['advantages']
        print("✨ Decision Tree의 장점:")
        for advantage in advantages:
            print(f"  🌟 {advantage}")
        
        # Performance results
        test_results = results['test_results']
        print(f"\n📊 성능 결과:")
        print(f"  🎯 테스트 정확도: {test_results['accuracy']:.4f}")
        print(f"  🔍 F1 스코어: {test_results['f1_score']:.4f}")
        print(f"  📈 AUC 스코어: {test_results['auc_score']:.4f}")
        print(f"  📉 테스트 손실: {test_results['loss']:.4f}")
        
        # Training info
        training_results = results['training_results']
        print(f"\n📈 훈련 정보:")
        print(f"  📊 최고 검증 정확도: {training_results['best_val_accuracy']:.4f}")
        print(f"  ⏱️ 훈련 시간: {training_results['total_training_time']:.1f}초")
        print(f"  🔄 에포크 수: {training_results['epochs_trained']}")
        
        # Interpretability info
        if 'interpretability_analysis' in results and results['interpretability_analysis']:
            analysis = results['interpretability_analysis']
            print(f"\n🌳 해석성 분석:")
            
            if 'tree_rules' in analysis:
                rule_count = len(analysis['tree_rules']) if analysis['tree_rules'] else 0
                print(f"  📋 추출된 규칙 수: {rule_count}")
            
            if 'importance_stats' in analysis:
                stats = analysis['importance_stats']
                print(f"  📊 평균 특성 중요도: {stats['mean_importance']:.6f}")
                print(f"  🔝 최대 특성 중요도: {stats['max_importance']:.6f}")
        
        # Model info
        model_info = results['model_summary']
        print(f"\n🧠 모델 정보:")
        print(f"  📊 총 파라미터: {model_info['total_parameters']:,}")
        print(f"  🌳 분류기 파라미터: {model_info['classifier_parameters']:,}")
        
        print(f"\n📁 상세 결과: {self.output_dir}")
        print("="*70)


def main():
    """메인 실행 함수"""
    runner = DecisionTreeExperimentRunner()
    results = runner.run_full_decision_tree_experiment()
    return results


if __name__ == "__main__":
    results = main() 
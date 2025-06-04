#!/usr/bin/env python3
"""
개선된 수정 모델 재실험
전체 데이터셋과 최적화된 하이퍼파라미터 사용
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


class ImprovedRevisedExperimentRunner:
    """개선된 수정 모델 재실험"""
    
    def __init__(self, output_dir: str = None):
        self.device = torch.device("mps" if torch.backends.mps.is_available() 
                                  else "cuda" if torch.cuda.is_available() 
                                  else "cpu")
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or f"outputs/improved_revised_{timestamp}"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎯 개선된 수정 모델 재실험 시작")
        logger.info(f"💻 Device: {self.device}")
        logger.info(f"📁 Output: {self.output_dir}")
    
    def create_improved_model(self):
        """개선된 설정으로 모델 생성"""
        logger.info("🏗️ 개선된 모델 생성 중...")
        
        # Create model with improved logistic regression settings
        model = create_interpretable_mmtd(
            classifier_type="logistic_regression",
            classifier_config={
                'l1_lambda': 0.00001,  # 더 작은 L1 정규화
                'l2_lambda': 0.0001,   # 더 작은 L2 정규화
                'dropout_rate': 0.05   # 더 작은 드롭아웃
            },
            device=self.device,
            checkpoint_path="MMTD/checkpoints/fold1/checkpoint-939/pytorch_model.bin"
        )
        
        logger.info(f"✅ 개선된 모델 생성 완료")
        logger.info(f"📊 총 파라미터: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, model.get_model_summary()
    
    def prepare_full_data(self):
        """전체 데이터셋 준비"""
        logger.info("📚 전체 데이터셋 로딩 중...")
        
        data_module = create_mmtd_data_module(
            csv_path="MMTD/DATA/email_data/EDP.csv",
            data_path="MMTD/DATA/email_data/pics",
            batch_size=32,  # 더 큰 배치 크기
            max_samples=None,  # 전체 데이터셋 사용
            num_workers=4,  # 더 많은 워커
            test_split=0.2,  # 테스트 비율 조정
            val_split=0.1
        )
        
        logger.info("✅ 전체 데이터셋 로딩 완료")
        logger.info(f"📊 훈련 샘플: {len(data_module.train_dataset):,}")
        logger.info(f"📊 검증 샘플: {len(data_module.val_dataset):,}")
        logger.info(f"📊 테스트 샘플: {len(data_module.test_dataset):,}")
        
        return data_module
    
    def run_improved_training(self):
        """개선된 훈련 실행"""
        logger.info("🚀 개선된 훈련 시작")
        
        # Create model and data
        model, model_summary = self.create_improved_model()
        data_module = self.prepare_full_data()
        
        # 개선된 훈련 설정
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=2e-5,  # 더 낮은 학습률
            weight_decay=1e-5,  # 더 작은 weight decay
            betas=(0.9, 0.999)
        )
        
        # 학습률 스케줄러 추가
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
        
        criterion = torch.nn.CrossEntropyLoss()
        
        # 훈련 설정
        num_epochs = 10  # 더 많은 에포크
        best_val_accuracy = 0.0
        best_model_state = None
        training_history = []
        patience_counter = 0
        early_stop_patience = 5
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            logger.info(f"📚 에포크 {epoch + 1}/{num_epochs}")
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_loader = data_module.train_dataloader()
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                train_correct += (predictions == batch['labels']).sum().item()
                train_total += batch['labels'].size(0)
                
                # 더 자주 로깅
                if batch_idx % 100 == 0:
                    current_acc = train_correct / train_total if train_total > 0 else 0
                    logger.info(f"  배치 {batch_idx}: 손실 {loss.item():.4f}, 정확도 {current_acc:.4f}")
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            val_loader = data_module.val_dataloader()
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    val_loss += loss.item()
                    predictions = torch.argmax(outputs.logits, dim=1)
                    val_correct += (predictions == batch['labels']).sum().item()
                    val_total += batch['labels'].size(0)
            
            # Calculate metrics
            train_accuracy = train_correct / train_total
            val_accuracy = val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Learning rate scheduling
            scheduler.step(val_accuracy)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                logger.info(f"  🎯 새로운 최고 검증 정확도: {val_accuracy:.4f}")
            else:
                patience_counter += 1
            
            # Log results
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            training_history.append(epoch_results)
            
            logger.info(f"  훈련 정확도: {train_accuracy:.4f}, 검증 정확도: {val_accuracy:.4f}")
            logger.info(f"  학습률: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                logger.info(f"🛑 조기 종료: {early_stop_patience} 에포크 동안 개선 없음")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"✅ 최고 성능 모델 로드 완료")
        
        training_time = time.time() - start_time
        
        training_results = {
            'training_history': training_history,
            'best_val_accuracy': best_val_accuracy,
            'total_training_time': training_time,
            'epochs_trained': epoch + 1,
            'early_stopped': patience_counter >= early_stop_patience
        }
        
        logger.info(f"⏱️ 훈련 완료: {training_time:.2f}초")
        logger.info(f"📈 최고 검증 정확도: {best_val_accuracy:.4f}")
        
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
                
                if batch_idx % 50 == 0:
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
    
    def run_full_improved_experiment(self):
        """전체 개선된 실험 실행"""
        try:
            logger.info("🚀 개선된 전체 재실험 시작")
            
            # 1. Training
            trained_model, training_results, model_summary = self.run_improved_training()
            
            # 2. Data preparation for evaluation  
            data_module = self.prepare_full_data()
            
            # 3. Comprehensive evaluation
            test_results = self.run_comprehensive_evaluation(trained_model, data_module)
            
            # 4. Interpretability analysis
            try:
                feature_importance = trained_model.get_feature_importance()
                importance_stats = {
                    'mean_importance': float(torch.mean(torch.abs(feature_importance))),
                    'max_importance': float(torch.max(torch.abs(feature_importance))),
                    'std_importance': float(torch.std(feature_importance)),
                    'sparsity': float(torch.sum(torch.abs(feature_importance) < 0.001) / len(feature_importance))
                }
            except Exception as e:
                logger.warning(f"해석성 분석 실패: {e}")
                importance_stats = {}
            
            # 5. Save results
            final_results = {
                'experiment_info': {
                    'timestamp': datetime.now().isoformat(),
                    'device': str(self.device),
                    'model_type': 'InterpretableMMTD_Improved',
                    'classifier_type': 'logistic_regression',
                    'note': 'Improved experiment with full dataset and optimized hyperparameters',
                    'improvements': [
                        'Full dataset (30K+ samples)',
                        'Lower learning rate (2e-5)',
                        'Larger batch size (32)',
                        'More epochs (10)',
                        'Learning rate scheduling',
                        'Early stopping',
                        'Reduced regularization'
                    ]
                },
                'model_summary': model_summary,
                'training_results': training_results,
                'test_results': test_results,
                'interpretability_analysis': importance_stats
            }
            
            # Save results
            results_path = os.path.join(self.output_dir, "improved_revised_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 결과 저장 완료: {results_path}")
            
            # Print summary
            self.print_improved_summary(final_results)
            
            logger.info("✅ 개선된 재실험 완료!")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ 개선된 실험 실패: {e}")
            raise
    
    def print_improved_summary(self, results):
        """개선된 결과 요약 출력"""
        print("\n" + "="*70)
        print("🎯 개선된 수정 MMTD 모델 재실험 결과")
        print("="*70)
        
        # Improvements
        improvements = results['experiment_info']['improvements']
        print("🔧 적용된 개선사항:")
        for improvement in improvements:
            print(f"  ✅ {improvement}")
        
        # Performance comparison
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
        
        # Model info
        model_info = results['model_summary']
        print(f"\n🧠 모델 정보:")
        print(f"  📊 총 파라미터: {model_info['total_parameters']:,}")
        print(f"  🎛️ 분류기 파라미터: {model_info['classifier_parameters']:,}")
        
        print(f"\n📁 상세 결과: {self.output_dir}")
        print("="*70)


def main():
    """메인 실행 함수"""
    runner = ImprovedRevisedExperimentRunner()
    results = runner.run_full_improved_experiment()
    return results


if __name__ == "__main__":
    results = main() 
#!/usr/bin/env python3
"""
수정된 MMTD 모델로 재실험
pooler 수정사항을 반영한 Logistic Regression 분류기 실험
"""

import os
import sys
import torch
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import sklearn

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.interpretable_mmtd import create_interpretable_mmtd
from models.mmtd_data_loader import create_mmtd_data_module
from models.mmtd_trainer import MMTDTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RevisedExperimentRunner:
    """수정된 모델로 재실험 실행"""
    
    def __init__(self, output_dir: str = None):
        self.device = torch.device("mps" if torch.backends.mps.is_available() 
                                  else "cuda" if torch.cuda.is_available() 
                                  else "cpu")
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or f"outputs/revised_experiment_{timestamp}"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎯 수정된 모델 재실험 시작")
        logger.info(f"💻 Device: {self.device}")
        logger.info(f"📁 Output: {self.output_dir}")
    
    def create_revised_model(self):
        """수정된 pooler를 포함한 InterpretableMMTD 모델 생성"""
        logger.info("🏗️ 수정된 모델 생성 중...")
        
        # Create model with logistic regression classifier
        model = create_interpretable_mmtd(
            classifier_type="logistic_regression",
            classifier_config={
                'l1_lambda': 0.0001,
                'l2_lambda': 0.001,
                'dropout_rate': 0.1
            },
            device=self.device,
            checkpoint_path="MMTD/checkpoints/fold1/checkpoint-939/pytorch_model.bin"
        )
        
        logger.info(f"✅ 모델 생성 완료")
        logger.info(f"📊 총 파라미터: {sum(p.numel() for p in model.parameters()):,}")
        
        # Get model summary
        model_summary = model.get_model_summary()
        logger.info(f"🧠 분류기 파라미터: {model_summary['classifier_parameters']:,}")
        
        return model, model_summary
    
    def prepare_data(self):
        """데이터 준비"""
        logger.info("📚 데이터 로딩 중...")
        
        data_module = create_mmtd_data_module(
            csv_path="MMTD/DATA/email_data/EDP.csv",
            data_path="MMTD/DATA/email_data/pics",
            batch_size=16,  # 안정적인 배치 크기
            max_samples=1000,  # 빠른 실험을 위해 샘플 제한
            num_workers=2,
            test_split=0.2,
            val_split=0.1
        )
        
        logger.info("✅ 데이터 로딩 완료")
        logger.info(f"📊 훈련 샘플: {len(data_module.train_dataset)}")
        logger.info(f"📊 검증 샘플: {len(data_module.val_dataset)}")
        logger.info(f"📊 테스트 샘플: {len(data_module.test_dataset)}")
        
        return data_module
    
    def run_training_experiment(self):
        """훈련 실험 실행"""
        logger.info("🚀 훈련 실험 시작")
        
        # Create model and data
        model, model_summary = self.create_revised_model()
        data_module = self.prepare_data()
        
        # Setup simple training
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training configuration
        num_epochs = 3
        best_val_accuracy = 0.0
        training_history = []
        
        # Train model
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
                
                if batch_idx % 10 == 0:
                    logger.info(f"  배치 {batch_idx}: 손실 {loss.item():.4f}")
            
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
            
            # Update best accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
            
            # Log results
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy
            }
            training_history.append(epoch_results)
            
            logger.info(f"  훈련 정확도: {train_accuracy:.4f}, 검증 정확도: {val_accuracy:.4f}")
        
        training_time = time.time() - start_time
        
        training_results = {
            'training_history': training_history,
            'best_val_accuracy': best_val_accuracy,
            'total_training_time': training_time
        }
        
        logger.info(f"⏱️ 훈련 완료: {training_time:.2f}초")
        logger.info(f"📈 최고 검증 정확도: {best_val_accuracy:.4f}")
        
        return model, training_results, model_summary
    
    def run_evaluation_experiment(self, model, data_module):
        """평가 실험 실행"""
        logger.info("📊 평가 실험 시작")
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []
        
        test_loader = data_module.test_dataloader()
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss
                
                test_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                test_correct += (predictions == batch['labels']).sum().item()
                test_total += batch['labels'].size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate detailed metrics
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        
        accuracy = test_correct / test_total
        avg_loss = test_loss / len(test_loader)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        test_results = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist()
        }
        
        logger.info(f"🎯 테스트 정확도: {accuracy:.4f}")
        logger.info(f"📉 테스트 손실: {avg_loss:.4f}")
        logger.info(f"🔍 F1 스코어: {f1_score:.4f}")
        
        return test_results
    
    def run_interpretability_analysis(self, model, data_module):
        """해석성 분석 실행"""
        logger.info("🔍 해석성 분석 시작")
        
        # Get feature importance
        try:
            feature_importance = model.get_feature_importance()
            
            # Simple statistics
            importance_stats = {
                'mean_importance': float(torch.mean(torch.abs(feature_importance))),
                'max_importance': float(torch.max(torch.abs(feature_importance))),
                'min_importance': float(torch.min(torch.abs(feature_importance))),
                'std_importance': float(torch.std(feature_importance)),
                'sparsity': float(torch.sum(torch.abs(feature_importance) < 0.001) / len(feature_importance))
            }
            
            logger.info(f"📊 평균 중요도: {importance_stats['mean_importance']:.6f}")
            logger.info(f"📊 최대 중요도: {importance_stats['max_importance']:.6f}")
            logger.info(f"📊 희소성: {importance_stats['sparsity']:.3f}")
            
            return importance_stats
            
        except Exception as e:
            logger.warning(f"해석성 분석 실패: {e}")
            return {}
    
    def save_results(self, training_results, test_results, model_summary, importance_stats):
        """결과 저장"""
        logger.info("💾 결과 저장 중...")
        
        # Combine all results
        final_results = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'model_type': 'InterpretableMMTD_Revised',
                'classifier_type': 'logistic_regression',
                'note': 'Experiment with corrected pooler implementation'
            },
            'model_summary': model_summary,
            'training_results': training_results,
            'test_results': test_results,
            'interpretability_analysis': importance_stats
        }
        
        # Save to JSON
        results_path = os.path.join(self.output_dir, "revised_experiment_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 결과 저장 완료: {results_path}")
        
        # Print summary
        self.print_summary(final_results)
        
        return final_results
    
    def print_summary(self, results):
        """결과 요약 출력"""
        print("\n" + "="*60)
        print("🎯 수정된 MMTD 모델 재실험 결과")
        print("="*60)
        
        # Model info
        model_info = results['model_summary']
        print(f"📊 모델 타입: {model_info['model_type']}")
        print(f"🧠 분류기: {model_info['classifier_type']}")
        print(f"📈 총 파라미터: {model_info['total_parameters']:,}")
        print(f"🎛️ 훈련 가능 파라미터: {model_info['trainable_parameters']:,}")
        
        # Performance
        test_results = results['test_results']
        print(f"\n📊 성능 결과:")
        print(f"  🎯 테스트 정확도: {test_results['accuracy']:.4f}")
        print(f"  📉 테스트 손실: {test_results['loss']:.4f}")
        print(f"  🔍 F1 스코어: {test_results['f1_score']:.4f}")
        
        # Training info
        training_results = results['training_results']
        if 'best_val_accuracy' in training_results:
            print(f"  📈 최고 검증 정확도: {training_results['best_val_accuracy']:.4f}")
        
        # Interpretability
        if results['interpretability_analysis']:
            interp = results['interpretability_analysis']
            print(f"\n🔍 해석성 분석:")
            print(f"  📊 평균 특징 중요도: {interp['mean_importance']:.6f}")
            print(f"  📊 희소성: {interp['sparsity']:.3f}")
        
        print(f"\n📁 상세 결과: {self.output_dir}")
        print("="*60)
    
    def run_full_experiment(self):
        """전체 실험 실행"""
        try:
            logger.info("🚀 수정된 모델 전체 재실험 시작")
            
            # 1. Training experiment
            trained_model, training_results, model_summary = self.run_training_experiment()
            
            # 2. Prepare data for evaluation
            data_module = self.prepare_data()
            
            # 3. Evaluation experiment
            test_results = self.run_evaluation_experiment(trained_model, data_module)
            
            # 4. Interpretability analysis
            importance_stats = self.run_interpretability_analysis(trained_model, data_module)
            
            # 5. Save results
            final_results = self.save_results(
                training_results, test_results, model_summary, importance_stats
            )
            
            logger.info("✅ 수정된 모델 재실험 완료!")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ 실험 실패: {e}")
            raise


def main():
    """메인 실행 함수"""
    runner = RevisedExperimentRunner()
    results = runner.run_full_experiment()
    return results


if __name__ == "__main__":
    results = main() 
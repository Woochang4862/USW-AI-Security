"""
Test MMTD Evaluation on Sample Data
Quick test to verify the evaluation pipeline works
"""

import sys
import os
sys.path.append('src')

from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from evaluation.evaluate_pretrained_mmtd import MMTDEvaluator

def main():
    """Test evaluation on sample data"""
    print("🧪 Testing MMTD Evaluation Pipeline")
    print("="*50)
    
    # Paths
    checkpoint_dir = Path("MMTD/checkpoints")
    data_path = Path("MMTD/DATA/email_data/pics")
    csv_path = Path("MMTD/DATA/email_data/EDP_sample.csv")  # Use sample data
    output_dir = Path("outputs/test_evaluation")
    
    # Create evaluator
    evaluator = MMTDEvaluator(
        checkpoint_dir=checkpoint_dir,
        data_path=data_path,
        csv_path=csv_path,
        batch_size=8  # Smaller batch for testing
    )
    
    print(f"📊 Dataset size: {len(evaluator.data_df)} samples")
    
    # Test single fold evaluation first
    print("\n🔍 Testing single fold evaluation...")
    
    # Create a small test split
    test_df = evaluator.data_df.head(100)  # Use first 100 samples
    
    try:
        # Test fold1
        result = evaluator.evaluate_fold('fold1', test_df)
        
        print(f"✅ Single fold test successful!")
        print(f"   Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        print(f"   Samples: {result['num_test_samples']}")
        
        # Save test results
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluator.save_results({'test_result': result}, output_dir / "test_results.json")
        
        print(f"📁 Test results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
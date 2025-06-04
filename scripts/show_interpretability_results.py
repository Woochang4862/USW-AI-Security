#!/usr/bin/env python3
"""
Display key interpretability analysis results
"""

import json
from pathlib import Path

def main():
    report_path = "outputs/interpretability_analysis/comprehensive_interpretability_report.json"
    
    with open(report_path, 'r') as f:
        report = json.load(f)

    print('🔍 MMTD Logistic Regression 해석성 분석 결과')
    print('=' * 60)

    # 모델 정보
    model_info = report['model_info']
    print(f'📊 모델 타입: {model_info["model_type"]}')
    print(f'📊 분류기 타입: {model_info["classifier_type"]}')
    print(f'📊 총 파라미터: {model_info["total_parameters"]:,}')
    print(f'📊 훈련 가능한 파라미터: {model_info["classifier_parameters"]:,} (분류기만)')

    # 특징 중요도 분석
    feature_analysis = report['feature_importance_analysis']
    print(f'\n🎯 특징 중요도 분석:')
    print(f'📈 최대 중요도: {feature_analysis["feature_importance"]["max"]:.6f}')
    print(f'📈 평균 중요도: {feature_analysis["feature_importance"]["mean"]:.6f}')
    print(f'📈 표준편차: {feature_analysis["feature_importance"]["std"]:.6f}')

    # 정규화 효과
    reg_effects = feature_analysis['regularization_effects']
    print(f'\n⚖️ 정규화 효과:')
    print(f'📊 L1 Norm: {reg_effects["l1_norm"]:.4f}')
    print(f'📊 L2 Norm: {reg_effects["l2_norm"]:.4f}')
    print(f'📊 희소성 수준: {reg_effects["sparsity"]:.4f}')

    # 결정 경계 분석
    decision_analysis = report['decision_boundary_analysis']
    print(f'\n🎯 결정 경계 분석:')
    print(f'📊 고도 판별 특징: {decision_analysis["feature_discrimination"]["highly_discriminative_features"]}개')
    print(f'📊 중도 판별 특징: {decision_analysis["feature_discrimination"]["moderately_discriminative_features"]}개')
    print(f'📊 저도 판별 특징: {decision_analysis["feature_discrimination"]["low_discriminative_features"]}개')
    print(f'📊 코사인 유사도: {decision_analysis["separability"]["cosine_similarity"]:.4f}')

    # 예측 해석
    prediction_analysis = report['prediction_interpretations']
    correct_preds = sum(1 for i in prediction_analysis['sample_interpretations'] if i['correct_prediction'])
    total_preds = len(prediction_analysis['sample_interpretations'])
    print(f'\n🔍 예측 해석 (샘플 {total_preds}개):')
    print(f'✅ 정확한 예측: {correct_preds}/{total_preds} ({correct_preds/total_preds*100:.1f}%)')

    # 인사이트
    insights = report['interpretability_insights']
    print(f'\n💡 핵심 인사이트:')
    for insight in insights['key_findings']:
        print(f'• {insight}')
    for insight in insights['feature_insights']:
        print(f'• {insight}')
    for insight in insights['model_behavior']:
        print(f'• {insight}')

    print(f'\n📝 요약: {insights["interpretability_summary"]}')
    
    # 샘플 예측 해석 예시
    print(f'\n🔍 샘플 예측 해석 예시:')
    sample_interp = prediction_analysis['sample_interpretations'][0]
    print(f'샘플 ID: {sample_interp["sample_id"]}')
    print(f'실제 라벨: {sample_interp["true_label"]} ({"Spam" if sample_interp["true_label"] == 1 else "Ham"})')
    print(f'예측 라벨: {sample_interp["predicted_label"]} ({"Spam" if sample_interp["predicted_label"] == 1 else "Ham"})')
    print(f'예측 확률: Ham {sample_interp["prediction_probability"][0]:.4f}, Spam {sample_interp["prediction_probability"][1]:.4f}')
    print(f'정확한 예측: {"✅" if sample_interp["correct_prediction"] else "❌"}')
    
    # 특징 기여도
    contributions = sample_interp['feature_contributions']
    print(f'Spam 총 기여도: {contributions["spam_total"]:.4f}')
    print(f'Ham 총 기여도: {contributions["ham_total"]:.4f}')

if __name__ == "__main__":
    main() 
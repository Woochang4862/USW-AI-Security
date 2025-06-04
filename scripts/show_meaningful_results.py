#!/usr/bin/env python3
"""
의미있는 해석성 분석 결과 출력 스크립트
"""

import json
from pathlib import Path

def main():
    report_path = "outputs/meaningful_interpretability/meaningful_interpretability_report.json"
    
    with open(report_path, 'r') as f:
        report = json.load(f)

    print('🎯 의미있는 해석성 분석 결과')
    print('=' * 50)

    # 텍스트 어텐션 분석
    text_attention = report['attention_analysis']['text_attention']
    print(f'📝 텍스트 어텐션 통계:')
    print(f'• 평균 어텐션: {text_attention["attention_statistics"]["mean_attention"]:.6f}')
    print(f'• 최대 어텐션: {text_attention["attention_statistics"]["max_attention"]:.6f}')
    print(f'• 어텐션 엔트로피: {text_attention["attention_statistics"]["attention_entropy"]:.4f}')

    print(f'\n🔥 가장 주목받는 토큰들:')
    for i, (token, score) in enumerate(text_attention['top_attended_tokens'][:5]):
        if token:
            print(f'{i+1}. "{token}": {score:.6f}')

    # 이미지 어텐션 분석
    image_attention = report['attention_analysis']['image_attention']
    print(f'\n🖼️ 이미지 어텐션 통계:')
    print(f'• 평균 어텐션: {image_attention["attention_statistics"]["mean_attention"]:.6f}')
    print(f'• 최대 어텐션: {image_attention["attention_statistics"]["max_attention"]:.6f}')
    print(f'• 어텐션 집중도: {image_attention["attention_statistics"]["attention_concentration"]:.6f}')

    # 모달리티 중요도
    modality = report['modality_importance']
    print(f'\n⚖️ 모달리티 중요도 (50샘플 분석):')
    print(f'• 텍스트만: {modality["text_only_accuracy"]:.4f} ({modality["text_only_accuracy"]*100:.1f}%)')
    print(f'• 이미지만: {modality["image_only_accuracy"]:.4f} ({modality["image_only_accuracy"]*100:.1f}%)')
    print(f'• 멀티모달: {modality["multimodal_accuracy"]:.4f} ({modality["multimodal_accuracy"]*100:.1f}%)')
    print(f'• 시너지 효과: {modality["modality_contribution"]["synergy_effect"]:.4f}')

    # 그래디언트 중요도 (fallback 정보)
    gradient = report['gradient_importance']
    if 'text_importance' in gradient:
        top_tokens = gradient['text_importance'].get('top_important_tokens', [])
        if top_tokens:
            print(f'\n📈 그래디언트 기반 중요 토큰들:')
            for i, (token, score) in enumerate(top_tokens[:3]):
                if token:
                    print(f'{i+1}. "{token}": {score:.6f}')

    # 인사이트
    insights = report['interpretability_insights']
    print(f'\n💡 핵심 인사이트:')
    for insight in insights['attention_insights']:
        print(f'• {insight}')
    for insight in insights['modality_insights']:
        print(f'• {insight}')

    print(f'\n🎯 실행 가능한 권장사항:')
    for rec in insights['actionable_recommendations']:
        print(f'• {rec}')

    # 샘플 정보
    sample_info = report['attention_analysis']['sample_info']
    print(f'\n📊 분석 대상 샘플:')
    print(f'• 라벨: {sample_info["label"]} ({"Spam" if sample_info["label"] == 1 else "Ham"})')
    print(f'• 토큰 수: {len([t for t in sample_info["tokens"] if t not in ["[CLS]", "[SEP]", "[PAD]"]])}개')
    
    # 실제 토큰들 중 의미있는 것들 표시
    meaningful_tokens = [t for t in sample_info["tokens"] if t not in ["[CLS]", "[SEP]", "[PAD]"] and t is not None][:10]
    print(f'• 주요 토큰들: {", ".join(meaningful_tokens)}')

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
어텐션 통계 상세 분석 스크립트
"""

import json
import numpy as np

def main():
    with open('outputs/meaningful_interpretability/meaningful_interpretability_report.json', 'r') as f:
        report = json.load(f)

    # 어텐션 점수들 분석
    scores = report['attention_analysis']['text_attention']['cls_attention_scores']
    tokens = report['attention_analysis']['sample_info']['tokens']

    print('🔍 어텐션 분포 상세 분석')
    print('=' * 40)

    # 기본 통계
    print(f'총 토큰 수: {len(scores)}개')
    print(f'평균: {np.mean(scores):.6f}')
    print(f'최대: {np.max(scores):.6f}') 
    print(f'최소: {np.min(scores):.6f}')
    print(f'표준편차: {np.std(scores):.6f}')

    # 상위 토큰들
    sorted_indices = np.argsort(scores)[::-1]
    print(f'\n🔥 상위 5개 토큰:')
    for i in range(5):
        idx = sorted_indices[i]
        if idx < len(tokens):
            token = tokens[idx]
            score = scores[idx]
            print(f'{i+1}. "{token}": {score:.6f} ({score*100:.2f}%)')

    # 어텐션 집중도 분석
    high_attention = sum(1 for s in scores if s > 0.01)
    medium_attention = sum(1 for s in scores if 0.001 < s <= 0.01) 
    low_attention = sum(1 for s in scores if s <= 0.001)

    print(f'\n📊 어텐션 집중도:')
    print(f'고어텐션 (>1%): {high_attention}개 토큰')
    print(f'중어텐션 (0.1-1%): {medium_attention}개 토큰')
    print(f'저어텐션 (<0.1%): {low_attention}개 토큰')

    # SEP 토큰 제외한 분석
    non_sep_indices = [i for i, token in enumerate(tokens) if token != '[SEP]']
    non_sep_scores = [scores[i] for i in non_sep_indices if i < len(scores)]
    non_sep_tokens = [tokens[i] for i in non_sep_indices]

    print(f'\n🚫 [SEP] 토큰 제외 분석:')
    if non_sep_scores:
        print(f'평균: {np.mean(non_sep_scores):.6f}')
        print(f'최대: {np.max(non_sep_scores):.6f}')
        print(f'어텐션 집중도: {np.std(non_sep_scores):.6f}')
        
        # SEP 제외 상위 토큰들
        non_sep_sorted = np.argsort(non_sep_scores)[::-1]
        print(f'\n🎯 [SEP] 제외 상위 3개 실제 중요 토큰:')
        for i in range(min(3, len(non_sep_sorted))):
            idx = non_sep_sorted[i]
            if idx < len(non_sep_tokens):
                token = non_sep_tokens[idx]
                score = non_sep_scores[idx]
                print(f'{i+1}. "{token}": {score:.6f} ({score*100:.2f}%)')

    # 엔트로피 의미 설명
    entropy = report['attention_analysis']['text_attention']['attention_statistics']['attention_entropy']
    max_entropy = np.log(len(scores))
    
    print(f'\n📈 엔트로피 분석:')
    print(f'현재 엔트로피: {entropy:.4f}')
    print(f'최대 엔트로피: {max_entropy:.4f} (완전 균등 분포)')
    print(f'정규화된 엔트로피: {entropy/max_entropy:.4f} (0=완전집중, 1=완전균등)')
    
    if entropy/max_entropy < 0.3:
        print('→ 매우 집중된 어텐션 패턴')
    elif entropy/max_entropy < 0.6:
        print('→ 중간 정도 집중된 어텐션 패턴')
    else:
        print('→ 고르게 분산된 어텐션 패턴')

if __name__ == "__main__":
    main() 
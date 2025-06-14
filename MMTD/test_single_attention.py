from real_attention_extractor import RealAttentionExtractor

# 단일 샘플 테스트
extractor = RealAttentionExtractor()
result = extractor.run_real_attention_analysis('무료 상품을 받으세요! 지금 클릭하세요!')

print(f"\n테스트 결과: {'성공' if result else '실패'}") 
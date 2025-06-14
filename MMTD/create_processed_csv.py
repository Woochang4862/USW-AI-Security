#!/usr/bin/env python3
"""
EDP.csv에서 누락된 이미지 파일들을 제거하여 EDP_processed.csv 생성
- 실제 존재하는 이미지 파일만 포함
- 데이터 무결성 보장
- 통계 정보 제공
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def create_processed_csv():
    """누락된 이미지를 제거한 정제된 CSV 파일을 생성합니다."""
    
    # 파일 경로 설정
    csv_path = "DATA/email_data/EDP.csv"
    pics_dir = "DATA/email_data/pics"
    output_path = "DATA/email_data/EDP_processed.csv"
    
    print("=" * 60)
    print("🔧 EDP_processed.csv 생성")
    print("=" * 60)
    
    # 1. 원본 CSV 로드
    if not os.path.exists(csv_path):
        print(f"❌ 원본 CSV 파일을 찾을 수 없습니다: {csv_path}")
        return
    
    print(f"📖 원본 CSV 로딩: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ 원본 데이터: {len(df):,}개 행")
    print(f"📋 컬럼: {list(df.columns)}")
    
    # 2. 실제 존재하는 이미지 파일 목록 생성
    print(f"\n🔍 실제 이미지 파일 스캔 중...")
    existing_images = set()
    
    for root, dirs, files in os.walk(pics_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                # 상대 경로 생성 (spam/filename.jpg 형태)
                rel_path = os.path.relpath(os.path.join(root, file), pics_dir)
                existing_images.add(rel_path.replace('\\', '/'))  # Windows 호환성
    
    print(f"✅ 실제 이미지 파일: {len(existing_images):,}개")
    
    # 3. 각 행의 이미지 파일 존재 여부 확인
    print(f"\n🔍 이미지 파일 존재 여부 확인 중...")
    
    valid_rows = []
    invalid_rows = []
    
    for idx, row in df.iterrows():
        img_path = str(row['pics']).strip()
        
        # NaN 값 처리
        if pd.isna(row['pics']) or img_path == 'nan':
            invalid_rows.append((idx, img_path, "NaN 값"))
            continue
        
        # 이미지 파일 존재 확인
        if img_path in existing_images:
            valid_rows.append(idx)
        else:
            invalid_rows.append((idx, img_path, "파일 없음"))
        
        # 진행률 표시
        if (idx + 1) % 1000 == 0:
            print(f"   진행률: {idx + 1:,}/{len(df):,} ({(idx + 1)/len(df)*100:.1f}%)")
    
    # 4. 정제된 데이터프레임 생성
    print(f"\n📊 데이터 정제 결과:")
    print(f"   ✅ 유효한 행: {len(valid_rows):,}개")
    print(f"   ❌ 제거된 행: {len(invalid_rows):,}개")
    print(f"   📈 데이터 보존율: {len(valid_rows)/len(df)*100:.2f}%")
    
    # 유효한 행만 선택
    df_processed = df.iloc[valid_rows].copy()
    
    # 5. 라벨 분포 확인
    print(f"\n📊 정제 전후 라벨 분포 비교:")
    
    # 원본 분포
    original_dist = df['labels'].value_counts().sort_index()
    print(f"📋 원본 분포:")
    for label, count in original_dist.items():
        percentage = count / len(df) * 100
        print(f"   라벨 {label}: {count:,}개 ({percentage:.1f}%)")
    
    # 정제 후 분포
    processed_dist = df_processed['labels'].value_counts().sort_index()
    print(f"📋 정제 후 분포:")
    for label, count in processed_dist.items():
        percentage = count / len(df_processed) * 100
        print(f"   라벨 {label}: {count:,}개 ({percentage:.1f}%)")
    
    # 6. 제거된 샘플 분석
    if invalid_rows:
        print(f"\n❌ 제거된 샘플 분석 (처음 10개):")
        for i, (idx, img_path, reason) in enumerate(invalid_rows[:10]):
            print(f"   {i+1}. 행 {idx}: {img_path} ({reason})")
        
        if len(invalid_rows) > 10:
            print(f"   ... 및 {len(invalid_rows) - 10}개 더")
        
        # 제거된 샘플의 라벨 분포
        removed_indices = [idx for idx, _, _ in invalid_rows]
        removed_labels = df.iloc[removed_indices]['labels'].value_counts().sort_index()
        print(f"\n📊 제거된 샘플의 라벨 분포:")
        for label, count in removed_labels.items():
            percentage = count / len(invalid_rows) * 100
            print(f"   라벨 {label}: {count:,}개 ({percentage:.1f}%)")
    
    # 7. 정제된 CSV 저장
    print(f"\n💾 정제된 CSV 저장 중...")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # CSV 저장
    df_processed.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ 저장 완료: {output_path}")
    
    # 8. 파일 크기 비교
    original_size = os.path.getsize(csv_path) / (1024 * 1024)  # MB
    processed_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    
    print(f"\n📏 파일 크기 비교:")
    print(f"   원본: {original_size:.2f} MB")
    print(f"   정제 후: {processed_size:.2f} MB")
    print(f"   크기 감소: {(original_size - processed_size):.2f} MB ({(original_size - processed_size)/original_size*100:.1f}%)")
    
    # 9. 검증
    print(f"\n🔍 생성된 파일 검증:")
    df_verify = pd.read_csv(output_path)
    print(f"   ✅ 로드 성공: {len(df_verify):,}개 행")
    print(f"   ✅ 컬럼 일치: {list(df_verify.columns) == list(df.columns)}")
    
    # 10. 요약 통계
    print(f"\n" + "=" * 60)
    print(f"📊 EDP_processed.csv 생성 완료")
    print(f"=" * 60)
    print(f"📁 출력 파일: {output_path}")
    print(f"📊 원본 데이터: {len(df):,}개 행")
    print(f"📊 정제 후 데이터: {len(df_processed):,}개 행")
    print(f"❌ 제거된 데이터: {len(invalid_rows):,}개 행")
    print(f"📈 데이터 보존율: {len(df_processed)/len(df)*100:.2f}%")
    print(f"🎯 이미지 무결성: 100.00% (모든 이미지 파일 존재 확인)")
    
    return {
        'original_count': len(df),
        'processed_count': len(df_processed),
        'removed_count': len(invalid_rows),
        'preservation_rate': len(df_processed)/len(df)*100,
        'output_file': output_path
    }

def verify_processed_csv():
    """생성된 EDP_processed.csv의 무결성을 재검증합니다."""
    
    output_path = "DATA/email_data/EDP_processed.csv"
    pics_dir = "DATA/email_data/pics"
    
    if not os.path.exists(output_path):
        print(f"❌ 정제된 CSV 파일이 없습니다: {output_path}")
        return
    
    print(f"\n🔍 EDP_processed.csv 무결성 재검증:")
    
    df = pd.read_csv(output_path)
    missing_count = 0
    
    for idx, row in df.iterrows():
        img_path = str(row['pics']).strip()
        full_path = os.path.join(pics_dir, img_path)
        
        if not os.path.exists(full_path):
            missing_count += 1
            if missing_count <= 5:  # 처음 5개만 출력
                print(f"   ❌ 누락: {img_path}")
    
    if missing_count == 0:
        print(f"   ✅ 모든 이미지 파일 존재 확인 ({len(df):,}개)")
    else:
        print(f"   ❌ {missing_count}개 파일 여전히 누락")
    
    return missing_count == 0

if __name__ == "__main__":
    # 정제된 CSV 생성
    result = create_processed_csv()
    
    # 무결성 재검증
    is_valid = verify_processed_csv()
    
    if result and is_valid:
        print(f"\n🎉 성공적으로 EDP_processed.csv를 생성했습니다!")
        print(f"📁 파일 위치: {result['output_file']}")
        print(f"📊 데이터 보존율: {result['preservation_rate']:.2f}%")
    else:
        print(f"\n❌ 파일 생성 중 문제가 발생했습니다.") 
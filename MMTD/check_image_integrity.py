#!/usr/bin/env python3
"""
EDP.csv 파일의 이미지 경로 무결성 검사 스크립트
- CSV 파일에서 참조된 모든 이미지가 실제로 존재하는지 확인
- 누락된 이미지 파일 리스트 생성
- 데이터셋 통계 정보 제공
"""

import os
import pandas as pd
from pathlib import Path
import numpy as np
from collections import Counter

def check_image_integrity():
    """EDP.csv와 pics 폴더의 이미지 무결성을 검사합니다."""
    
    # 파일 경로 설정
    csv_path = "DATA/email_data/EDP.csv"
    pics_dir = "DATA/email_data/pics"
    
    print("=" * 60)
    print("📊 EDP.csv 이미지 무결성 검사")
    print("=" * 60)
    
    # 1. 파일 존재 확인
    if not os.path.exists(csv_path):
        print(f"❌ CSV 파일을 찾을 수 없습니다: {csv_path}")
        return
    
    if not os.path.exists(pics_dir):
        print(f"❌ 이미지 디렉토리를 찾을 수 없습니다: {pics_dir}")
        return
    
    print(f"✅ CSV 파일 발견: {csv_path}")
    print(f"✅ 이미지 디렉토리 발견: {pics_dir}")
    print()
    
    # 2. CSV 파일 로드
    print("📖 CSV 파일 로딩 중...")
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ CSV 로드 완료: {len(df):,}개 행")
    except Exception as e:
        print(f"❌ CSV 로드 실패: {e}")
        return
    
    # 3. CSV 구조 분석
    print(f"📋 CSV 컬럼: {list(df.columns)}")
    print(f"📏 데이터 크기: {df.shape}")
    print()
    
    # 4. 이미지 경로 컬럼 찾기
    image_columns = []
    for col in df.columns:
        if 'image' in col.lower() or 'pic' in col.lower() or 'path' in col.lower():
            image_columns.append(col)
    
    if not image_columns:
        # 일반적인 패턴으로 추정
        for col in df.columns:
            sample_values = df[col].dropna().head(10).astype(str)
            if any('.jpg' in val or '.png' in val or '.jpeg' in val for val in sample_values):
                image_columns.append(col)
    
    print(f"🖼️  이미지 경로 컬럼 후보: {image_columns}")
    
    if not image_columns:
        print("❌ 이미지 경로 컬럼을 찾을 수 없습니다.")
        print("📋 첫 5행 샘플:")
        print(df.head())
        return
    
    # 5. 각 이미지 컬럼에 대해 검사
    total_missing = 0
    total_checked = 0
    
    for col in image_columns:
        print(f"\n🔍 '{col}' 컬럼 검사 중...")
        
        # NaN 값 제거
        image_paths = df[col].dropna()
        print(f"📊 유효한 이미지 경로: {len(image_paths):,}개")
        
        if len(image_paths) == 0:
            print("⚠️  유효한 이미지 경로가 없습니다.")
            continue
        
        # 경로 샘플 출력
        print(f"📝 경로 샘플:")
        for i, path in enumerate(image_paths.head(3)):
            print(f"   {i+1}. {path}")
        
        # 이미지 파일 존재 확인
        missing_files = []
        existing_files = []
        
        for idx, img_path in enumerate(image_paths):
            if pd.isna(img_path):
                continue
                
            # 경로 정규화
            img_path = str(img_path).strip()
            
            # 상대 경로를 절대 경로로 변환
            if img_path.startswith('pics/'):
                full_path = os.path.join("DATA/email_data", img_path)
            elif img_path.startswith('DATA/'):
                full_path = img_path
            else:
                full_path = os.path.join(pics_dir, img_path)
            
            total_checked += 1
            
            if os.path.exists(full_path):
                existing_files.append(img_path)
            else:
                missing_files.append((idx, img_path, full_path))
                total_missing += 1
            
            # 진행률 표시 (1000개마다)
            if (idx + 1) % 1000 == 0:
                print(f"   진행률: {idx + 1:,}/{len(image_paths):,} ({(idx + 1)/len(image_paths)*100:.1f}%)")
        
        # 결과 출력
        print(f"\n📈 '{col}' 컬럼 검사 결과:")
        print(f"   ✅ 존재하는 파일: {len(existing_files):,}개")
        print(f"   ❌ 누락된 파일: {len(missing_files):,}개")
        print(f"   📊 무결성: {len(existing_files)/len(image_paths)*100:.2f}%")
        
        # 누락된 파일 상세 정보 (처음 10개만)
        if missing_files:
            print(f"\n❌ 누락된 파일 샘플 (처음 10개):")
            for i, (idx, img_path, full_path) in enumerate(missing_files[:10]):
                print(f"   {i+1}. 행 {idx}: {img_path}")
                print(f"      → 찾은 경로: {full_path}")
            
            if len(missing_files) > 10:
                print(f"   ... 및 {len(missing_files) - 10}개 더")
    
    # 6. 실제 이미지 디렉토리 구조 확인
    print(f"\n📁 실제 이미지 디렉토리 구조:")
    for root, dirs, files in os.walk(pics_dir):
        level = root.replace(pics_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        # 파일 개수만 표시 (너무 많으면)
        if len(files) > 5:
            img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
            print(f"{indent}  📷 이미지 파일: {len(img_files)}개")
            if img_files:
                print(f"{indent}     예시: {img_files[0]}")
        else:
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    print(f"{subindent}📷 {file}")
    
    # 7. 전체 요약
    print(f"\n" + "=" * 60)
    print(f"📊 전체 검사 요약")
    print(f"=" * 60)
    print(f"🔍 검사한 이미지 경로: {total_checked:,}개")
    print(f"✅ 존재하는 파일: {total_checked - total_missing:,}개")
    print(f"❌ 누락된 파일: {total_missing:,}개")
    print(f"📈 전체 무결성: {(total_checked - total_missing)/total_checked*100:.2f}%")
    
    if total_missing == 0:
        print(f"\n🎉 모든 이미지 파일이 정상적으로 존재합니다!")
    else:
        print(f"\n⚠️  {total_missing}개의 이미지 파일이 누락되었습니다.")
    
    return {
        'total_checked': total_checked,
        'total_missing': total_missing,
        'integrity_rate': (total_checked - total_missing)/total_checked*100 if total_checked > 0 else 0
    }

def analyze_dataset_distribution():
    """데이터셋의 라벨 분포를 분석합니다."""
    csv_path = "DATA/email_data/EDP.csv"
    
    if not os.path.exists(csv_path):
        return
    
    print(f"\n📊 데이터셋 라벨 분포 분석")
    print(f"=" * 40)
    
    df = pd.read_csv(csv_path)
    
    # 라벨 컬럼 찾기
    label_columns = []
    for col in df.columns:
        if 'label' in col.lower() or 'class' in col.lower() or 'spam' in col.lower():
            label_columns.append(col)
    
    for col in label_columns:
        print(f"\n🏷️  '{col}' 컬럼 분포:")
        value_counts = df[col].value_counts()
        for value, count in value_counts.items():
            percentage = count / len(df) * 100
            print(f"   {value}: {count:,}개 ({percentage:.1f}%)")

if __name__ == "__main__":
    # 이미지 무결성 검사 실행
    result = check_image_integrity()
    
    # 데이터셋 분포 분석
    analyze_dataset_distribution()
    
    print(f"\n✅ 검사 완료!") 
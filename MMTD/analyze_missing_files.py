#!/usr/bin/env python3
"""
누락된 이미지 파일들의 패턴 분석 스크립트
- 파일명 인코딩 문제 진단
- 특수문자 및 길이 문제 분석
- 실제 파일과 CSV 경로 매칭 시도
"""

import os
import pandas as pd
import re
from pathlib import Path
import unicodedata

def analyze_missing_files():
    """누락된 파일들의 패턴을 분석합니다."""
    
    csv_path = "DATA/email_data/EDP.csv"
    pics_dir = "DATA/email_data/pics"
    
    print("=" * 60)
    print("🔍 누락된 파일 패턴 분석")
    print("=" * 60)
    
    # CSV 로드
    df = pd.read_csv(csv_path)
    image_paths = df['pics'].dropna()
    
    # 실제 파일 목록 생성
    actual_files = set()
    for root, dirs, files in os.walk(pics_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                rel_path = os.path.relpath(os.path.join(root, file), pics_dir)
                actual_files.add(rel_path.replace('\\', '/'))  # Windows 호환성
    
    print(f"📊 실제 이미지 파일 수: {len(actual_files):,}개")
    print(f"📊 CSV 이미지 경로 수: {len(image_paths):,}개")
    
    # 누락된 파일 찾기
    missing_files = []
    existing_files = []
    
    for idx, img_path in enumerate(image_paths):
        img_path = str(img_path).strip()
        
        # 경로에서 폴더 부분 제거 (spam/, ham/)
        if '/' in img_path:
            folder, filename = img_path.split('/', 1)
            normalized_path = f"{folder}/{filename}"
        else:
            normalized_path = img_path
        
        if normalized_path in actual_files:
            existing_files.append(img_path)
        else:
            missing_files.append((idx, img_path))
    
    print(f"❌ 누락된 파일: {len(missing_files):,}개")
    print(f"✅ 존재하는 파일: {len(existing_files):,}개")
    
    # 누락된 파일 패턴 분석
    print(f"\n🔍 누락된 파일 패턴 분석:")
    
    # 1. 파일명 길이 분석
    missing_lengths = [len(os.path.basename(path)) for _, path in missing_files]
    if missing_lengths:
        print(f"📏 누락된 파일명 길이:")
        print(f"   평균: {sum(missing_lengths)/len(missing_lengths):.1f}자")
        print(f"   최대: {max(missing_lengths)}자")
        print(f"   최소: {min(missing_lengths)}자")
    
    # 2. 특수문자 분석
    special_char_count = 0
    unicode_count = 0
    long_filename_count = 0
    
    for _, path in missing_files:
        filename = os.path.basename(path)
        
        # 특수문자 검사
        if re.search(r'[^\w\-_.]', filename):
            special_char_count += 1
        
        # 유니코드 문자 검사
        if any(ord(char) > 127 for char in filename):
            unicode_count += 1
        
        # 긴 파일명 검사 (255자 제한)
        if len(filename) > 200:
            long_filename_count += 1
    
    print(f"\n📊 누락된 파일 특성:")
    print(f"   특수문자 포함: {special_char_count}개 ({special_char_count/len(missing_files)*100:.1f}%)")
    print(f"   유니코드 문자 포함: {unicode_count}개 ({unicode_count/len(missing_files)*100:.1f}%)")
    print(f"   긴 파일명 (200자+): {long_filename_count}개 ({long_filename_count/len(missing_files)*100:.1f}%)")
    
    # 3. 누락된 파일 샘플 (다양한 패턴)
    print(f"\n❌ 누락된 파일 샘플 (다양한 패턴):")
    
    # 유니코드 문자가 있는 파일들
    unicode_files = [(idx, path) for idx, path in missing_files 
                    if any(ord(char) > 127 for char in os.path.basename(path))]
    if unicode_files:
        print(f"\n🌐 유니코드 문자 포함 파일들 (처음 5개):")
        for i, (idx, path) in enumerate(unicode_files[:5]):
            filename = os.path.basename(path)
            print(f"   {i+1}. 행 {idx}: {filename}")
            print(f"      길이: {len(filename)}자")
            # 유니코드 문자 분석
            unicode_chars = [char for char in filename if ord(char) > 127]
            print(f"      유니코드 문자: {len(unicode_chars)}개")
    
    # 긴 파일명들
    long_files = [(idx, path) for idx, path in missing_files 
                 if len(os.path.basename(path)) > 100]
    if long_files:
        print(f"\n📏 긴 파일명들 (처음 3개):")
        for i, (idx, path) in enumerate(long_files[:3]):
            filename = os.path.basename(path)
            print(f"   {i+1}. 행 {idx}: {filename[:50]}...")
            print(f"      전체 길이: {len(filename)}자")
    
    # 4. 실제 파일과 유사한 이름 찾기 시도
    print(f"\n🔍 유사한 파일명 매칭 시도:")
    
    # 실제 파일명들을 간단한 형태로 변환
    actual_simple_names = {}
    for file_path in actual_files:
        filename = os.path.basename(file_path)
        # 확장자 제거하고 간단한 형태로
        simple_name = re.sub(r'[^\w]', '', filename.lower().split('.')[0])
        if simple_name:
            actual_simple_names[simple_name] = file_path
    
    matches_found = 0
    for idx, missing_path in missing_files[:20]:  # 처음 20개만 시도
        missing_filename = os.path.basename(missing_path)
        missing_simple = re.sub(r'[^\w]', '', missing_filename.lower().split('.')[0])
        
        if missing_simple in actual_simple_names:
            matches_found += 1
            print(f"   🎯 매칭 발견:")
            print(f"      누락: {missing_filename}")
            print(f"      실제: {os.path.basename(actual_simple_names[missing_simple])}")
    
    if matches_found == 0:
        print(f"   ❌ 유사한 파일명을 찾을 수 없습니다.")
    else:
        print(f"   ✅ {matches_found}개의 유사한 파일명을 찾았습니다.")
    
    return {
        'total_missing': len(missing_files),
        'special_char_count': special_char_count,
        'unicode_count': unicode_count,
        'long_filename_count': long_filename_count,
        'matches_found': matches_found
    }

def check_filesystem_encoding():
    """파일시스템 인코딩 문제를 확인합니다."""
    
    print(f"\n🔧 파일시스템 인코딩 검사:")
    
    pics_dir = "DATA/email_data/pics/spam"
    
    # 실제 파일들 중 유니코드 문자가 있는 것들 찾기
    unicode_files = []
    
    try:
        for filename in os.listdir(pics_dir):
            if any(ord(char) > 127 for char in filename):
                unicode_files.append(filename)
    except Exception as e:
        print(f"❌ 디렉토리 읽기 실패: {e}")
        return
    
    print(f"📊 실제 유니코드 파일명: {len(unicode_files)}개")
    
    if unicode_files:
        print(f"📝 유니코드 파일명 샘플 (처음 5개):")
        for i, filename in enumerate(unicode_files[:5]):
            print(f"   {i+1}. {filename}")
            print(f"      길이: {len(filename)}자")
            
            # 파일 존재 확인
            full_path = os.path.join(pics_dir, filename)
            exists = os.path.exists(full_path)
            print(f"      존재: {'✅' if exists else '❌'}")

if __name__ == "__main__":
    result = analyze_missing_files()
    check_filesystem_encoding()
    
    print(f"\n" + "=" * 60)
    print(f"📊 분석 완료")
    print(f"=" * 60)
    print(f"❌ 총 누락 파일: {result['total_missing']:,}개")
    print(f"🌐 유니코드 문제: {result['unicode_count']:,}개")
    print(f"📏 긴 파일명 문제: {result['long_filename_count']:,}개")
    print(f"🎯 매칭 가능: {result['matches_found']:,}개") 
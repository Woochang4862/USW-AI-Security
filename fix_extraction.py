#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
올바른 디렉토리 구조로 이메일 데이터 재압축 해제
"""

import zipfile
import os
import sys
from pathlib import Path

def extract_with_proper_structure(zip_path, extract_to):
    """올바른 디렉토리 구조로 ZIP 파일을 압축 해제합니다."""
    print(f"압축 해제 시작: {zip_path}")
    print(f"대상 디렉토리: {extract_to}")
    
    # 기존 디렉토리 삭제
    if os.path.exists(extract_to):
        import shutil
        shutil.rmtree(extract_to)
    
    # 추출 디렉토리 생성
    os.makedirs(extract_to, exist_ok=True)
    
    success_count = 0
    error_count = 0
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.infolist()
        total_files = len(file_list)
        
        print(f"총 {total_files}개 파일 처리 중...")
        
        for i, file_info in enumerate(file_list):
            if i % 1000 == 0:
                print(f"진행률: {i}/{total_files} ({i/total_files*100:.1f}%)")
            
            try:
                # 원본 파일명
                original_filename = file_info.filename
                
                # 파일명 디코딩 시도 (여러 인코딩 방식)
                decoded_filename = None
                
                # 인코딩 시도 순서
                encodings = ['utf-8', 'cp949', 'euc-kr', 'gbk', 'gb2312', 'big5', 'shift-jis', 'latin-1']
                
                for encoding in encodings:
                    try:
                        # ZIP 파일 내부에서 파일명은 보통 CP437로 인코딩되어 있음
                        decoded_filename = original_filename.encode('cp437').decode(encoding)
                        break
                    except:
                        try:
                            # 직접 디코딩 시도
                            decoded_filename = original_filename.encode('latin-1').decode(encoding)
                            break
                        except:
                            continue
                
                # 디코딩 실패시 원본 사용
                if decoded_filename is None:
                    decoded_filename = original_filename
                
                # 안전한 파일 경로 생성
                safe_path = os.path.join(extract_to, decoded_filename)
                
                # 디렉토리 생성
                os.makedirs(os.path.dirname(safe_path), exist_ok=True)
                
                # 파일이 디렉토리가 아닌 경우에만 추출
                if not file_info.is_dir():
                    with zip_ref.open(file_info) as source:
                        with open(safe_path, 'wb') as target:
                            target.write(source.read())
                    
                    success_count += 1
                    
            except Exception as e:
                error_count += 1
                if error_count <= 10:  # 처음 10개 오류만 출력
                    print(f"파일 처리 오류: {file_info.filename} - {str(e)}")
                continue
    
    print(f"\n압축 해제 완료!")
    print(f"성공: {success_count}개 파일")
    print(f"오류: {error_count}개 파일")
    
    return success_count, error_count

def analyze_structure(extract_to):
    """압축 해제된 디렉토리 구조를 분석합니다."""
    print(f"\n디렉토리 구조 분석: {extract_to}")
    
    if not os.path.exists(extract_to):
        print("추출 디렉토리가 존재하지 않습니다.")
        return
    
    # 디렉토리 구조 출력
    for root, dirs, files in os.walk(extract_to):
        level = root.replace(extract_to, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # 처음 5개 파일만 표시
            print(f"{subindent}{file}")
        
        if len(files) > 5:
            print(f"{subindent}... 및 {len(files) - 5}개 파일 더")

def main():
    """메인 함수"""
    zip_file = "MMTD/email_data.zip"
    extract_dir = "email_data"
    
    if not os.path.exists(zip_file):
        print(f"오류: {zip_file} 파일을 찾을 수 없습니다.")
        return 1
    
    print("=" * 60)
    print("올바른 디렉토리 구조로 이메일 데이터 압축 해제")
    print("=" * 60)
    
    try:
        # 압축 해제
        success, errors = extract_with_proper_structure(zip_file, extract_dir)
        
        # 구조 분석
        analyze_structure(extract_dir)
        
        print(f"\n압축 해제가 완료되었습니다!")
        print(f"추출된 파일들은 '{extract_dir}' 디렉토리에 있습니다.")
        
        if errors > 0:
            print(f"\n주의: {errors}개 파일에서 오류가 발생했습니다.")
        
        # 예상 구조 확인
        expected_paths = [
            "email_data/pics/spam/",
            "email_data/pics/ham/",
            "email_data/email_data_EDP.csv"
        ]
        
        print(f"\n예상 구조 확인:")
        for path in expected_paths:
            if os.path.exists(path):
                print(f"✅ {path}")
            else:
                print(f"❌ {path}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이메일 데이터 압축 해제 및 인코딩 문제 해결 스크립트
다양한 인코딩 방식을 시도하여 올바른 파일명으로 압축을 해제합니다.
"""

import zipfile
import os
import sys
import chardet
from pathlib import Path
import shutil

def detect_encoding(data):
    """데이터의 인코딩을 감지합니다."""
    result = chardet.detect(data)
    return result['encoding'], result['confidence']

def try_decode_filename(filename_bytes, encodings=None):
    """다양한 인코딩으로 파일명을 디코딩 시도합니다."""
    if encodings is None:
        encodings = [
            'utf-8', 'cp949', 'euc-kr', 'iso-8859-1', 'windows-1252',
            'gbk', 'gb2312', 'big5', 'shift_jis', 'cp932', 'utf-16',
            'latin1', 'ascii'
        ]
    
    decoded_names = []
    
    for encoding in encodings:
        try:
            decoded = filename_bytes.decode(encoding)
            decoded_names.append((encoding, decoded))
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    return decoded_names

def safe_extract_zip(zip_path, extract_to, max_attempts=5):
    """안전하게 zip 파일을 압축 해제합니다."""
    print(f"압축 해제 시작: {zip_path}")
    print(f"대상 디렉토리: {extract_to}")
    
    # 추출 디렉토리 생성
    os.makedirs(extract_to, exist_ok=True)
    
    success_count = 0
    error_count = 0
    encoding_stats = {}
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.infolist()
        total_files = len(file_list)
        
        print(f"총 {total_files}개 파일 처리 중...")
        
        for i, file_info in enumerate(file_list):
            if i % 1000 == 0:
                print(f"진행률: {i}/{total_files} ({i/total_files*100:.1f}%)")
            
            try:
                # 원본 파일명 (바이트)
                original_filename = file_info.filename
                
                # 파일명 디코딩 시도
                decoded_names = try_decode_filename(original_filename.encode('latin1'))
                
                if decoded_names:
                    # 가장 가능성 높은 인코딩 선택 (UTF-8 우선)
                    best_encoding = None
                    best_name = None
                    
                    for encoding, decoded_name in decoded_names:
                        if encoding == 'utf-8':
                            best_encoding = encoding
                            best_name = decoded_name
                            break
                    
                    if not best_encoding:
                        best_encoding, best_name = decoded_names[0]
                    
                    # 인코딩 통계 업데이트
                    encoding_stats[best_encoding] = encoding_stats.get(best_encoding, 0) + 1
                    
                    # 안전한 파일명 생성
                    safe_filename = create_safe_filename(best_name)
                    safe_path = os.path.join(extract_to, safe_filename)
                    
                    # 디렉토리 생성
                    os.makedirs(os.path.dirname(safe_path), exist_ok=True)
                    
                    # 파일 추출
                    with zip_ref.open(file_info) as source:
                        with open(safe_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                    
                    success_count += 1
                    
                else:
                    # 디코딩 실패 시 원본 이름 사용
                    safe_filename = create_safe_filename(original_filename)
                    safe_path = os.path.join(extract_to, safe_filename)
                    
                    os.makedirs(os.path.dirname(safe_path), exist_ok=True)
                    
                    with zip_ref.open(file_info) as source:
                        with open(safe_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                    
                    success_count += 1
                    
            except Exception as e:
                error_count += 1
                print(f"파일 처리 오류: {file_info.filename} - {str(e)}")
                continue
    
    print(f"\n압축 해제 완료!")
    print(f"성공: {success_count}개 파일")
    print(f"오류: {error_count}개 파일")
    print(f"\n인코딩 통계:")
    for encoding, count in sorted(encoding_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {encoding}: {count}개 파일")
    
    return success_count, error_count, encoding_stats

def create_safe_filename(filename):
    """안전한 파일명을 생성합니다."""
    # 위험한 문자들을 안전한 문자로 대체
    unsafe_chars = '<>:"/\\|?*'
    safe_filename = filename
    
    for char in unsafe_chars:
        safe_filename = safe_filename.replace(char, '_')
    
    # 연속된 언더스코어 정리
    while '__' in safe_filename:
        safe_filename = safe_filename.replace('__', '_')
    
    # 파일명 길이 제한 (Windows 호환성)
    if len(safe_filename) > 200:
        name, ext = os.path.splitext(safe_filename)
        safe_filename = name[:200-len(ext)] + ext
    
    return safe_filename

def analyze_extracted_files(extract_to):
    """압축 해제된 파일들을 분석합니다."""
    print(f"\n압축 해제된 파일 분석 중: {extract_to}")
    
    file_types = {}
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_size = os.path.getsize(file_path)
                total_size += file_size
                file_count += 1
                
                # 파일 확장자 통계
                ext = os.path.splitext(file)[1].lower()
                file_types[ext] = file_types.get(ext, 0) + 1
                
            except OSError:
                continue
    
    print(f"총 파일 수: {file_count}")
    print(f"총 크기: {total_size / (1024*1024):.2f} MB")
    print(f"\n파일 타입별 통계:")
    for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ext or '(확장자 없음)'}: {count}개")

def main():
    """메인 함수"""
    zip_file = "MMTD/email_data.zip"
    extract_dir = "extracted_email_data"
    
    if not os.path.exists(zip_file):
        print(f"오류: {zip_file} 파일을 찾을 수 없습니다.")
        return
    
    print("=" * 60)
    print("이메일 데이터 압축 해제 및 인코딩 문제 해결")
    print("=" * 60)
    
    try:
        # 압축 해제
        success, errors, encoding_stats = safe_extract_zip(zip_file, extract_dir)
        
        # 결과 분석
        analyze_extracted_files(extract_dir)
        
        print(f"\n압축 해제가 완료되었습니다!")
        print(f"추출된 파일들은 '{extract_dir}' 디렉토리에 있습니다.")
        
        if errors > 0:
            print(f"\n주의: {errors}개 파일에서 오류가 발생했습니다.")
            print("일부 파일은 손상되었거나 접근할 수 없을 수 있습니다.")
        
        # CSV 파일 인코딩 확인 및 변환
        csv_file = os.path.join(extract_dir, "email_data_EDP.csv")
        if os.path.exists(csv_file):
            print(f"\nCSV 파일 인코딩 확인 중: {csv_file}")
            
            # 다양한 인코딩으로 시도
            encodings_to_try = ['utf-8', 'cp949', 'euc-kr', 'gbk', 'gb2312', 'big5', 'shift-jis', 'latin-1']
            
            for encoding in encodings_to_try:
                try:
                    with open(csv_file, 'r', encoding=encoding) as f:
                        content = f.read(1000)  # 처음 1000자만 읽어서 테스트
                        print(f"✅ {encoding} 인코딩으로 성공적으로 읽을 수 있습니다.")
                        print(f"처음 500자 미리보기:")
                        print("-" * 50)
                        print(content[:500])
                        print("-" * 50)
                        
                        # UTF-8로 변환된 파일 생성
                        utf8_file = csv_file.replace('.csv', '_utf8.csv')
                        with open(csv_file, 'r', encoding=encoding) as source:
                            with open(utf8_file, 'w', encoding='utf-8') as target:
                                target.write(source.read())
                        print(f"✅ UTF-8로 변환된 파일 생성: {utf8_file}")
                        break
                        
                except UnicodeDecodeError:
                    print(f"❌ {encoding} 인코딩으로 읽을 수 없습니다.")
                    continue
                except Exception as e:
                    print(f"❌ {encoding} 인코딩 시도 중 오류: {e}")
                    continue
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

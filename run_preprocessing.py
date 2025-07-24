#!/usr/bin/env python3
"""
HR 데이터 전처리 실행 스크립트

사용법:
    python run_preprocessing.py

기능:
    - 01. consultations/output 폴더의 모든 문서 파일 처리
    - Excel (.xlsx, .xls), CSV, PDF, Word (.docx) 파일 지원
    - 처리된 결과를 data/processed/documents.pkl로 저장
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 메인 전처리 모듈 임포트 및 실행
if __name__ == "__main__":
    try:
        from src.preprocessing.main_preprocessor import (
            run_preprocessing_pipeline,
            save_documents_to_pickle,
            preview_documents
        )
        
        # 경로 설정
        raw_data_path = os.path.join(current_dir, '01. consultations', 'output')
        output_path = os.path.join(current_dir, 'data', 'processed', 'documents.pkl')
        
        print("🚀 HR 데이터 전처리 시작!")
        print(f"📂 입력 경로: {raw_data_path}")
        print(f"💾 출력 경로: {output_path}")
        print("-" * 60)
        
        # 경로 존재 확인
        if not os.path.exists(raw_data_path):
            print(f"❌ 오류: 입력 경로가 존재하지 않습니다: {raw_data_path}")
            print("'01. consultations/output' 폴더가 있는지 확인해주세요.")
            sys.exit(1)
        
        # 전처리 실행
        processed_docs = run_preprocessing_pipeline(raw_data_dir=raw_data_path)
        
        if processed_docs:
            # 결과 저장
            save_documents_to_pickle(processed_docs, output_path)
            
            # 미리보기
            preview_documents(processed_docs)
            
            print(f"\n✅ 전처리 완료!")
            print(f"📄 총 {len(processed_docs)}개의 문서가 처리되었습니다.")
            print(f"💾 결과 파일: {output_path}")
            
            # 파일 크기 확인
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"📊 파일 크기: {file_size:.2f} MB")
        else:
            print("❌ 처리된 문서가 없습니다.")
            print("입력 폴더에 지원되는 파일 형식(.xlsx, .xls, .csv, .pdf, .docx)이 있는지 확인해주세요.")
    
    except ImportError as e:
        print(f"❌ 모듈 임포트 오류: {e}")
        print("필요한 라이브러리들이 설치되어 있는지 확인해주세요:")
        print("pip install pypdf docx2txt unstructured openpyxl pandas")
    
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc() 
import pandas as pd
from src.preprocessing.parsers import parse_excel_for_hr_data
from src.preprocessing.normalizers import clean_text
from langchain_core.documents import Document
import os
import glob
import pickle
from typing import List, Dict, Any

# 새로운 로더 임포트 (설치 필요: pip install pypdf docx2txt unstructured)
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredWordDocumentLoader

def run_preprocessing_pipeline(raw_data_dir: str) -> List[Document]:
    """
    모든 원시 엑셀, PDF, Word 파일을 읽고, 파싱 및 정규화하여 Document 객체 리스트를 반환합니다.
    raw_data_dir은 이 함수를 호출하는 스크립트의 현재 작업 디렉토리를 기준으로 한 상대 경로여야 합니다.
    """
    all_documents = []
    
    print(f"전처리 시작: '{raw_data_dir}' 디렉토리 스캔 중...")
    
    # 1. 엑셀/CSV 파일 찾기 및 처리
    excel_csv_files = glob.glob(os.path.join(raw_data_dir, "**", "*.xlsx"), recursive=True)
    excel_csv_files.extend(glob.glob(os.path.join(raw_data_dir, "**", "*.xls"), recursive=True))
    excel_csv_files.extend(glob.glob(os.path.join(raw_data_dir, "**", "*.csv"), recursive=True))
    
    print(f"'{raw_data_dir}'에서 {len(excel_csv_files)}개의 엑셀/CSV 파일을 찾았습니다.")
    for file_path in excel_csv_files:
        print(f"처리 중 (Excel/CSV): {file_path}")
        try:
            parsed_data_list = parse_excel_for_hr_data(file_path) # parsers.py의 Excel 처리 함수 사용

            for item in parsed_data_list:
                cleaned_text = clean_text(item['text'])
                if cleaned_text:
                    doc = Document(page_content=cleaned_text, metadata=item['metadata'])
                    all_documents.append(doc)
                else:
                    print(f"경고: {file_path}의 한 행에서 정규화 후 유효한 텍스트가 없습니다. 이 데이터를 건너뛰었습니다.")
        except Exception as e:
            print(f"오류: Excel/CSV 파일 '{file_path}' 처리 중 오류 발생: {e}")

    # 2. PDF 파일 찾기 및 처리
    pdf_files = glob.glob(os.path.join(raw_data_dir, "**", "*.pdf"), recursive=True)
    print(f"'{raw_data_dir}'에서 {len(pdf_files)}개의 PDF 파일을 찾았습니다.")
    for file_path in pdf_files:
        print(f"처리 중 (PDF): {file_path}")
        try:
            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()
            for doc in pdf_docs:
                cleaned_text = clean_text(doc.page_content)
                if cleaned_text:
                    # PDF는 기본적으로 page_content와 metadata를 가집니다.
                    # source, page 등의 metadata는 loader가 자동으로 추가합니다.
                    # 필요한 경우 추가적인 metadata를 여기에 더할 수 있습니다.
                    doc.page_content = cleaned_text
                    doc.metadata["source_file"] = os.path.basename(file_path)
                    doc.metadata["file_path"] = file_path
                    doc.metadata["source_type"] = "pdf"
                    all_documents.append(doc)
                else:
                    print(f"경고: {file_path}의 한 페이지에서 유효한 텍스트가 없습니다. 이 페이지를 건너뛰었습니다.")
        except Exception as e:
            print(f"오류: PDF 파일 '{file_path}' 처리 중 오류 발생: {e}")

    # 3. Word (docx) 파일 찾기 및 처리
    docx_files = glob.glob(os.path.join(raw_data_dir, "**", "*.docx"), recursive=True)
    print(f"'{raw_data_dir}'에서 {len(docx_files)}개의 Word (docx) 파일을 찾았습니다.")
    for file_path in docx_files:
        print(f"처리 중 (Word): {file_path}")
        try:
            # Docx2txtLoader는 간단하지만, 더 강력한 파싱을 위해 UnstructuredWordDocumentLoader 권장
            # loader = Docx2txtLoader(file_path) # 필요시 사용
            loader = UnstructuredWordDocumentLoader(file_path) # Unstructured 라이브러리 필요
            word_docs = loader.load()
            for doc in word_docs:
                cleaned_text = clean_text(doc.page_content)
                if cleaned_text:
                    doc.page_content = cleaned_text
                    doc.metadata["source_file"] = os.path.basename(file_path)
                    doc.metadata["file_path"] = file_path
                    doc.metadata["source_type"] = "docx"
                    all_documents.append(doc)
                else:
                    print(f"경고: {file_path}의 한 부분에서 유효한 텍스트가 없습니다. 이 부분을 건너뛰었습니다.")
        except Exception as e:
            print(f"오류: Word 파일 '{file_path}' 처리 중 오류 발생: {e}")
            # Fallback: Docx2txtLoader 시도
            print(f"UnstructuredWordDocumentLoader 실패, Docx2txtLoader로 재시도...")
            try:
                loader = Docx2txtLoader(file_path)
                word_docs = loader.load()
                for doc in word_docs:
                    cleaned_text = clean_text(doc.page_content)
                    if cleaned_text:
                        doc.page_content = cleaned_text
                        doc.metadata["source_file"] = os.path.basename(file_path)
                        doc.metadata["file_path"] = file_path
                        doc.metadata["source_type"] = "docx"
                        all_documents.append(doc)
            except Exception as e2:
                print(f"오류: Docx2txtLoader도 실패 '{file_path}': {e2}")

    print(f"전처리 완료. 총 {len(all_documents)}개의 Document 객체 생성.")
    return all_documents

def save_documents_to_pickle(documents: List[Document], output_path: str):
    """
    생성된 Document 객체 리스트를 pickle 파일로 저장합니다.
    output_path는 이 함수를 호출하는 스크립트의 현재 작업 디렉토리를 기준으로 한 상대 경로여야 합니다.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(documents, f)
    print(f"Document 객체 {len(documents)}개를 '{output_path}'에 저장했습니다.")

def load_documents_from_pickle(input_path: str) -> List[Document]:
    """
    저장된 pickle 파일에서 Document 객체 리스트를 로드합니다.
    
    Args:
        input_path (str): pickle 파일 경로
        
    Returns:
        List[Document]: 로드된 Document 객체 리스트
    """
    try:
        with open(input_path, 'rb') as f:
            documents = pickle.load(f)
        print(f"'{input_path}'에서 {len(documents)}개의 Document 객체를 로드했습니다.")
        return documents
    except Exception as e:
        print(f"오류: pickle 파일 로드 실패 '{input_path}': {e}")
        return []

def preview_documents(documents: List[Document], num_samples: int = 3):
    """
    처리된 Document 객체들의 샘플을 미리보기합니다.
    
    Args:
        documents (List[Document]): 미리볼 Document 리스트
        num_samples (int): 미리볼 샘플 수
    """
    print(f"\n=== Document 미리보기 (총 {len(documents)}개 중 {min(num_samples, len(documents))}개) ===")
    
    for i, doc in enumerate(documents[:num_samples]):
        print(f"\n--- Document {i+1} ---")
        print(f"Content 길이: {len(doc.page_content)} 문자")
        print(f"Content 미리보기: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)

if __name__ == "__main__":
    # 프로젝트 루트 경로 계산
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # 실제 데이터 경로: "01. consultations/output"
    raw_data_path = os.path.join(project_root, '01. consultations', 'output')
    output_path = os.path.join(project_root, 'data', 'processed', 'documents.pkl')
    
    print(f"프로젝트 루트: {project_root}")
    print(f"원시 데이터 경로: {raw_data_path}")
    print(f"출력 파일 경로: {output_path}")
    
    # 경로 존재 확인
    if not os.path.exists(raw_data_path):
        print(f"⚠️  경고: 원시 데이터 경로가 존재하지 않습니다: {raw_data_path}")
        print("다음 중 하나를 확인해주세요:")
        print("1. '01. consultations/output' 폴더가 존재하는지")
        print("2. 파일 경로가 올바른지")
        exit(1)
    
    # 전처리 실행
    print("🚀 전처리 파이프라인 시작!")
    processed_docs = run_preprocessing_pipeline(raw_data_dir=raw_data_path)
    
    if processed_docs:
        # 결과 저장
        save_documents_to_pickle(processed_docs, output_path)
        
        # 미리보기
        preview_documents(processed_docs)
        
        print(f"\n✅ 전처리 완료!")
        print(f"📄 총 {len(processed_docs)}개의 문서가 처리되었습니다.")
        print(f"💾 결과 파일: {output_path}")
    else:
        print("❌ 처리된 문서가 없습니다. 입력 폴더와 파일들을 확인해주세요.") 
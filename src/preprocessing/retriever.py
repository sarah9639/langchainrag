# src/preprocessing/retriever.py (이 코드로 파일을 업데이트 해주세요)
print("DEBUG: Script started.")

import os
import pickle
from typing import List
from langchain_core.documents import Document
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
print("DEBUG: Environment variables loaded.")

def load_documents_from_pickle(input_path: str) -> List[Document]:
    """
    저장된 pickle 파일에서 Document 객체 리스트를 로드합니다.
    """
    try:
        with open(input_path, 'rb') as f:
            documents = pickle.load(f)
        print(f"'{input_path}'에서 {len(documents)}개의 Document 객체를 로드했습니다.")
        return documents
    except Exception as e:
        print(f"오류: pickle 파일 로드 실패 '{input_path}': {e}")
        return []

def initialize_retriever(processed_data_path: str):
    """
    Document를 로드하고, 텍스트 분할, 임베딩 모델을 사용하여 FAISS 벡터 저장소를 구축한 후, retriever를 반환합니다.
    """
    print("🚀 Retriever 초기화 시작...")

    documents = load_documents_from_pickle(processed_data_path)

    if not documents:
        print("❌ 로드된 문서가 없습니다. retriever를 초기화할 수 없습니다.")
        return None

    print("텍스트 분할(Text Splitting) 중...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    
    split_documents = text_splitter.split_documents(documents)
    print(f"✅ 원본 {len(documents)}개의 문서가 {len(split_documents)}개의 청크로 분할되었습니다.")

    # 2. 임베딩 모델 로드
    # UpstageEmbeddings에 'model' 인자를 추가하여 사용할 모델 이름을 명시합니다.
    model_name = "solar-embedding-1-large-passage" # <-- 모델 이름 변경
    print(f"임베딩 모델 로드 중: UpstageEmbeddings (모델: {model_name})")

    try:
        embeddings = UpstageEmbeddings(model=model_name)
    except Exception as e:
        print(f"오류: UpstageEmbeddings 모델 로드 실패. UPSTAGE_API_KEY가 올바르게 설정되었는지 확인하거나, 모델 이름('{model_name}')을 확인하세요. {e}")
        return None

    # 3. FAISS 벡터 저장소 구축 (분할된 문서 사용)
    print("FAISS 벡터 저장소 구축 중...")
    try:
        vectorstore = FAISS.from_documents(split_documents, embeddings)
        print("✅ FAISS 벡터 저장소 구축 완료.")
    except Exception as e:
        print(f"오류: FAISS 벡터 저장소 구축 실패: {e}")
        return None

    # 4. retriever 반환
    retriever = vectorstore.as_retriever()
    print("✅ Retriever 초기화 완료!")
    return retriever

if __name__ == "__main__":
    print("DEBUG: Inside __main__ block.")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

    processed_data_path = os.path.join(project_root, 'data', 'processed', 'documents.pkl')

    print(f"프로젝트 루트: {project_root}")
    print(f"처리된 데이터 경로: {processed_data_path}")

    retriever = initialize_retriever(processed_data_path)

    if retriever:
        print("\n=== Retriever 테스트 ===")
        query = "권고사직을 당했을 때 실업급여를 받을 수 있나요?"
        docs = retriever.invoke(query)
        print(f"쿼리: '{query}'")
        print(f"검색된 문서 수: {len(docs)}개")
        for i, doc in enumerate(docs[:3]):
            print(f"\n--- 문서 {i+1} ---")
            print(f"Content 미리보기: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")

# src/preprocessing/qa_chain.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA # 이 라인은 더 이상 직접 사용하지 않지만, 임시로 남겨둡니다.
from langchain_community.callbacks import get_openai_callback
import yaml
from langchain_teddynote import logging

# LCEL 구축을 위한 추가 임포트
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter # 딕셔너리에서 특정 키의 값을 추출하기 위해 사용

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# LangSmith 추적 설정
logging.langsmith("RAG-SAMPLE", set_enable=True)

# retriever.py 파일에서 initialize_retriever 함수를 임포트합니다.
from retriever import initialize_retriever

def load_prompt_from_yaml(file_path: str):
    """YAML 파일에서 프롬프트 템플릿을 로드합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get("qa_template", "")
    except Exception as e:
        print(f"오류: YAML 파일에서 프롬프트 로드 실패 '{file_path}': {e}")
        return ""

def build_qa_chain(retriever, llm_model_name: str = "gpt-4o-mini", llm_options: dict = None):
    """
    주어진 retriever와 LLM을 사용하여 질의응답 LCEL 체인을 구축합니다.
    llm_options를 통해 LLM의 추가 매개변수 (예: temperature)를 설정할 수 있습니다.
    """
    print("🚀 질의응답(QA) LCEL 체인 구축 시작...")

    if llm_options is None:
        llm_options = {}

    # ChatOpenAI LLM 초기화
    print(f"LLM 로드 중: ChatOpenAI (모델: {llm_model_name}, 옵션: {llm_options})")
    try:
        llm = ChatOpenAI(model=llm_model_name, **llm_options)
    except Exception as e:
        print(f"오류: ChatOpenAI 모델 로드 실패. OPENAI_API_KEY가 올바르게 설정되었는지, 모델 이름('{llm_model_name}')이 유효한지 확인하세요. {e}")
        return None

    # YAML 파일에서 프롬프트 템플릿 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    prompt_file_path = os.path.join(project_root, 'src', 'prompt', 'qa_prompt.yaml')
    
    qa_template_string = load_prompt_from_yaml(prompt_file_path)

    if not qa_template_string:
        print("❌ 프롬프트 템플릿 로드 실패. QA 체인을 구축할 수 없습니다.")
        return None

    QA_CHAIN_PROMPT = PromptTemplate.from_template(qa_template_string)

    # LCEL 체인 구축 시작
    retrieval_and_pass_through_chain = RunnableParallel({
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question")
    }).with_config(run_name="RetrieveAndPassQuestion")

    answer_generation_chain = (
        QA_CHAIN_PROMPT | llm | StrOutputParser()
    ).with_config(run_name="GenerateAnswer")

    final_rag_chain = (
        retrieval_and_pass_through_chain
        | RunnableParallel({
            "answer": answer_generation_chain,
            "source_documents": itemgetter("context")
        })
    ).with_config(run_name="FinalRAGChain")

    print("✅ 질의응답(QA) LCEL 체인 구축 완료!")
    return final_rag_chain

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

    processed_data_path = os.path.join(project_root, 'data', 'processed', 'documents.pkl')

    print("\n--- 검색기(Retriever) 초기화 중 ---")
    retriever_instance = initialize_retriever(processed_data_path)

    if retriever_instance:
        print("\n--- 질의응답(QA) LCEL 체인 구축 중 ---")
        
        # LLM 옵션 설정 (temperature를 0.0으로 설정)
        llm_custom_options = {
            "temperature": 0.0, # <-- 이 값을 0.0으로 변경했습니다.
        }
        
        qa_chain_instance = build_qa_chain(
            retriever_instance,
            llm_model_name="gpt-4o-mini",
            llm_options=llm_custom_options
        )

        if qa_chain_instance:
            print("\n=== 질의응답 체인 테스트 ===")
            query = "권고사직을 당했을 때 실업급여를 받을 수 있나요?"
            print(f"질문: {query}")

            with get_openai_callback() as cb:
                result = qa_chain_instance.invoke({"question": query})

                print(f"총 토큰 사용량: {cb.total_tokens}")
                print(f"총 비용: ${cb.total_cost:.4f}")

            print("\n--- 답변 ---")
            print(result["answer"])

            print("\n--- 참조 문서 ---")
            for i, doc in enumerate(result["source_documents"][:3]):
                print(f"문서 {i+1} (Source: {doc.metadata.get('source', 'N/A')}, ID: {doc.metadata.get('column_ID', 'N/A')}):")
                print(f"  {doc.page_content[:200]}...")
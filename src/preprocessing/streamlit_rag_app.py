#!/usr/bin/env python3
"""
🏢 HR RAG 채팅 앱

기존 RAG 시스템을 활용한 Streamlit 웹 애플리케이션
- ChatGPT 스타일 인터페이스
- 스트리밍 답변 출력
- 채팅 기록 유지
- 사이드바 설정 옵션
- 완전한 에러 처리
"""

import streamlit as st
import sys
import os
import time
from typing import Generator, Optional

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="🏢 HR RAG 채팅봇",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "system_ready" not in st.session_state:
        st.session_state.system_ready = False

def load_rag_system(temperature: float = 0.1, k: int = 3):
    """RAG 시스템 로드 및 초기화"""
    try:
        # 캐싱을 위해 이미 로드된 경우 재사용
        if st.session_state.system_ready and st.session_state.retriever is not None:
            # temperature 변경 시에만 QA 체인 재생성
            if hasattr(st.session_state, 'last_temperature') and st.session_state.last_temperature != temperature:
                st.session_state.qa_chain = create_qa_chain_with_temperature(st.session_state.retriever, temperature, k)
                st.session_state.last_temperature = temperature
            return True
        
        # 처음 로드하는 경우
        with st.spinner("🚀 RAG 시스템 초기화 중..."):
            # retriever 로드
            from src.preprocessing.retriever import initialize_retriever
            
            processed_data_path = os.path.join(current_dir, 'data', 'processed', 'documents.pkl')
            retriever = initialize_retriever(processed_data_path)
            
            if not retriever:
                st.error("❌ 검색기 초기화 실패")
                return False
            
            st.session_state.retriever = retriever
            
            # QA 체인 생성
            qa_chain = create_qa_chain_with_temperature(retriever, temperature, k)
            if not qa_chain:
                st.error("❌ QA 체인 생성 실패")
                return False
                
            st.session_state.qa_chain = qa_chain
            st.session_state.system_ready = True
            st.session_state.last_temperature = temperature
            
            st.success("✅ RAG 시스템 초기화 완료!")
            return True
            
    except Exception as e:
        st.error(f"❌ RAG 시스템 로드 중 오류: {str(e)}")
        return False

def create_qa_chain_with_temperature(retriever, temperature: float, k: int):
    """지정된 temperature로 QA 체인 생성"""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_upstage import ChatUpstage
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        
        # LLM 초기화
        openai_api_key = os.getenv("OPENAI_API_KEY")
        upstage_api_key = os.getenv("UPSTAGE_API_KEY")
        
        if openai_api_key:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=temperature,
                max_tokens=1000,
                streaming=True  # 스트리밍 활성화
            )
        elif upstage_api_key:
            llm = ChatUpstage(
                model="solar-1-mini-chat",
                temperature=temperature,
                max_tokens=1000,
                streaming=True  # 스트리밍 활성화
            )
        else:
            st.error("❌ API 키가 설정되지 않았습니다.")
            return None
        
        # 프롬프트 템플릿
        template = """당신은 대한민국의 노무 전문가입니다. 주어진 문서를 기반으로 정확하고 실용적인 답변을 제공해주세요.

**답변 원칙:**
1. 법적 근거를 명확히 제시해주세요
2. 실무적인 절차와 방법을 구체적으로 안내해주세요  
3. 주의사항이나 예외 상황도 함께 설명해주세요
4. 관련 기관이나 담당 부서 정보도 포함해주세요

**참고 문서:**
{context}

**질문:** {question}

**답변:**"""

        prompt = ChatPromptTemplate.from_template(template)
        
        # 문서 포맷팅 함수
        def format_docs(docs):
            if not docs:
                return "관련 문서를 찾을 수 없습니다."
            
            formatted_docs = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', '알 수 없음')
                source_type = doc.metadata.get('source_type', '알 수 없음')
                doc_info = f"[문서 {i}] ({source}, {source_type})\n{doc.page_content}\n"
                formatted_docs.append(doc_info)
            
            return "\n".join(formatted_docs)
        
        # retriever에 k 설정 적용
        retriever_with_k = retriever.with_config({"search_kwargs": {"k": k}})
        
        # QA 체인 구성
        qa_chain = (
            {
                "context": retriever_with_k | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt 
            | llm 
            | StrOutputParser()
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"❌ QA 체인 생성 실패: {str(e)}")
        return None

def stream_response(response_text: str) -> Generator[str, None, None]:
    """응답 텍스트를 스트리밍 방식으로 출력"""
    words = response_text.split()
    for i, word in enumerate(words):
        yield word + " "
        if i % 3 == 0:  # 3단어마다 잠시 대기
            time.sleep(0.05)

def get_rag_response(question: str) -> Optional[str]:
    """RAG 시스템에서 답변 생성"""
    try:
        if not st.session_state.qa_chain:
            return None
            
        # QA 체인 실행
        response = st.session_state.qa_chain.invoke(question)
        return response
        
    except Exception as e:
        st.error(f"❌ 답변 생성 중 오류: {str(e)}")
        return None

def main():
    """메인 앱"""
    
    # 세션 상태 초기화
    init_session_state()
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # Temperature 슬라이더
        temperature = st.slider(
            "🌡️ 창의성 조절 (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="낮을수록 일관된 답변, 높을수록 창의적 답변"
        )
        
        # 검색 문서 수 설정
        k = st.number_input(
            "📄 검색할 문서 수",
            min_value=1,
            max_value=10,
            value=3,
            help="관련 문서를 몇 개까지 검색할지 설정"
        )
        
        st.divider()
        
        # 시스템 상태 표시
        if st.session_state.system_ready:
            st.success("✅ 시스템 준비 완료")
        else:
            st.warning("⚠️ 시스템 초기화 필요")
        
        # 대화 초기화 버튼
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.success("대화 기록이 초기화되었습니다!")
            st.rerun()
        
        st.divider()
        
        # 앱 정보
        st.info("""
        **🏢 HR RAG 채팅봇**
        
        노무 관련 질문에 대해 정확하고 실용적인 답변을 제공합니다.
        
        **주요 기능:**
        - 📚 143개 문서 기반 검색
        - 🤖 AI 기반 답변 생성  
        - 💬 대화 기록 유지
        - ⚙️ 설정 커스터마이징
        """)
    
    # 메인 화면
    st.title("🏢 HR RAG 채팅봇")
    st.caption("노무 전문가 AI가 여러분의 질문에 답변해드립니다!")
    
    # RAG 시스템 로드
    if not load_rag_system(temperature, k):
        st.stop()
    
    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("노무 관련 질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI 응답 생성 및 표시
        with st.chat_message("assistant"):
            with st.spinner("🤔 답변 생성 중..."):
                response = get_rag_response(prompt)
                
            if response:
                # 스트리밍 방식으로 응답 출력
                response_placeholder = st.empty()
                streamed_response = ""
                
                for chunk in stream_response(response):
                    streamed_response += chunk
                    response_placeholder.markdown(streamed_response + "▌")
                
                # 최종 응답 표시
                response_placeholder.markdown(response)
                
                # 응답을 세션에 저장
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                error_msg = "죄송합니다. 답변을 생성할 수 없습니다. 다시 시도해주세요."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main() 
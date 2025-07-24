import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import load_prompt
from langchain_teddynote.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote import logging
from dotenv import load_dotenv
import os
import pickle
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# 1. 환경 변수 및 로깅 설정
# ─────────────────────────────────────────────────────────────────────────────
# .env 파일에서 OPENAI_API_KEY 등 환경 변수를 불러옵니다.
load_dotenv()

# LangSmith 로깅: 프로젝트 이름 지정 (로그 추적 시 사용)
logging.langsmith("[Project] HR RAG 채팅봇")

# ─────────────────────────────────────────────────────────────────────────────
# 2. 캐시 디렉토리 준비
# ─────────────────────────────────────────────────────────────────────────────
# 디스크에 결과물을 저장할 캐시 폴더 생성 (한 번 생성하면 재사용)

# 1) 최상위 캐시폴더 준비 
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 2) embeddings 저장용 폴더 준비 
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Streamlit UI 설정 및 세션 상태 초기화
# ─────────────────────────────────────────────────────────────────────────────
# 페이지 설정
st.set_page_config(
    page_title="HR RAG 채팅봇 | HR RAG 채팅봇",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SuperLawyer 스타일 CSS
st.markdown("""
<style>
    /* 전체 배경 */
    .main {
        background-color: #f8f9fa;
    }
    
    /* 사이드바 스타일 */
    .css-1d391kg {
        background-color: #1e1e1e;
    }
    
    /* 메인 헤더 */
    .main-header {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* 채팅 메시지 스타일 */
    .user-message {
        background: #f3f4f6;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        margin-left: 15%;
        border-left: 4px solid #6b7280;
    }
    
    .assistant-message {
        background: #eff6ff;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        margin-right: 15%;
        border-left: 4px solid #3b82f6;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
    
    /* 상태 배지 */
    .status-ready {
        background: #dcfce7;
        color: #166534;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .status-loading {
        background: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    /* 입력창 스타일 */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e5e7eb;
        padding: 0.75rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2rem;">⚖️ HR RAG 채팅봇</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">노무에 필요한 사항을 전달해드려요</p>
    </div>
""", unsafe_allow_html=True)

# 1. 세션 상태에 `"messages"`라는 키가 없으면(=최초 로드 시) 빈 리스트 `[]`를 할당
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 2. `st.session_state`에 `"chain"`이라는 키가 없으면, `None` 값을 할당
if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 3. retriever 초기화 상태 관리
if "retriever_ready" not in st.session_state:
    st.session_state["retriever_ready"] = False

# 사이드바 구성 (설정 옵션)
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #374151; margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0;">⚖️ HR RAG</h2>
            <p style="color: #9ca3af; margin: 0; font-size: 14px;">SuperLawyer</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.header("⚙️ 설정")
    
    # 모델 선택
    selected_model = st.selectbox(
        "🤖 LLM 선택", 
        ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20240620"], 
        index=1  # gpt-4o-mini가 기본값
    )
    
    # 검색 문서 수
    k = st.number_input(
        "📄 검색할 문서 수",
        min_value=1,
        max_value=10,
        value=3
    )
    
    # 메모리 상태 표시
    if "conversation_memory" in st.session_state:
        memory = st.session_state.conversation_memory
        chat_history_data = memory.load_memory_variables({})
        chat_history = chat_history_data.get("chat_history", [])
        
        if isinstance(chat_history, list):
            num_messages = len(chat_history)
            turns = num_messages // 2
            st.markdown(f'<span class="status-ready">💭 {turns}턴 대화 저장중</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-ready">💭 요약된 대화 이력 보존중</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-loading">💭 대화 없음</span>', unsafe_allow_html=True)
    
    st.divider()
    
    # 대화 초기화 버튼
    clear_btn = st.button("🗑️ 대화 초기화", use_container_width=True)
    
    st.divider()
    
    # 시스템 정보
    st.info("""
    **🏢 HR RAG 채팅봇**
    
    노무 관련 질문에 대해 정확하고 실용적인 답변을 제공합니다.
    
    **데이터 기반:**
    - 📚 143개 HR 전문 문서
    - 📄 권고사직, 해고, 실업급여
    - ⚖️ 법적 근거 및 실무 절차
    
    **메모리 시스템:**
    - 🧠 이전 대화 내용 기억
    - 📝 1000토큰 초과시 자동 요약
    - 🔄 연속적인 멀티턴 대화 지원
    """)

# ─────────────────────────────────────────────────────────────────────────────
# 4. 대화 기록 헬퍼 함수
# ─────────────────────────────────────────────────────────────────────────────
# 메시지 출력 (SuperLawyer 스타일로 향상)
def print_messages():
    for msg in st.session_state["messages"]:
        if msg.role == "user":
            st.markdown(f'<div class="user-message">👤 **사용자**<br/>{msg.content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">⚖️ **HR 전문가**<br/>{msg.content}</div>', unsafe_allow_html=True)

# 메시지 추가 
def add_message(role: str, content: str):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))

# ─────────────────────────────────────────────────────────────────────────────
# 5. HR Documents → FAISS Retriever 생성 (무거운 작업 캐시)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="HR 문서를 처리중입니다...")
def create_retriever():
    """기존에 전처리된 documents.pkl에서 FAISS 검색기를 생성합니다."""
    
    try:
        # 1) 전처리된 문서 로드
        documents_path = os.path.join("data", "processed", "documents.pkl")
        
        if not os.path.exists(documents_path):
            st.error(f"❌ 문서 파일을 찾을 수 없습니다: {documents_path}")
            return None
            
        with open(documents_path, 'rb') as f:
            documents = pickle.load(f)
        
        st.info(f"✅ {len(documents)}개의 HR 문서를 로드했습니다.")
        
        # 2) 텍스트 분할 (임베딩 최적화를 위한 청크 생성)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100
        )
        split_docs = splitter.split_documents(documents)
        
        st.info(f"✅ 문서를 {len(split_docs)}개의 청크로 분할했습니다.")
        
        # 3) 임베딩 생성 (OpenAI)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if openai_api_key:
            embeddings = OpenAIEmbeddings()
            st.info("✅ OpenAI 임베딩을 사용합니다.")
        else:
            st.error("❌ API 키가 설정되지 않았습니다.")
            return None
        
        # 4) FAISS 인덱스 생성
        vectorstore = FAISS.from_documents(
            documents=split_docs,
            embedding=embeddings
        )
        
        # 5) 검색기(retriever)로 변환
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        
        st.success("✅ FAISS 검색기 생성 완료!")
        return retriever
        
    except Exception as e:
        st.error(f"❌ 검색기 생성 중 오류: {str(e)}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 6. LangChain RAG 체인 생성 함수
# ─────────────────────────────────────────────────────────────────────────────
def create_chain(retriever, model="gpt-3.5-turbo"):
    """RAG 체인을 생성합니다."""
    
    try:
        # 1) 프롬프트 생성 (YAML 파일에서 로드)
        from langchain_core.prompts import ChatPromptTemplate
        
        # YAML 파일에서 프롬프트 템플릿 로드
        prompt_path = "src/prompt/qa_prompt.yaml"
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_config = yaml.safe_load(f)
                template = prompt_config['qa_template']
        except FileNotFoundError:
            st.error(f"❌ 프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
            return None
        except Exception as e:
            st.error(f"❌ 프롬프트 파일 로드 실패: {e}")
            return None

        prompt = ChatPromptTemplate.from_template(template)
        
        # 2) LLM 생성 (temperature 고정: 0)
        if model.startswith("gpt"):
            llm = ChatOpenAI(
                model=model, 
                temperature=0,
                streaming=True  # 스트리밍 활성화
            )
        elif model.startswith("claude"):
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model=model,
                temperature=0,
                streaming=True  # 스트리밍 활성화
            )
        else:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo", 
                temperature=0,
                streaming=True
            )
        
        # 3) 문서 포맷팅 함수
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
        
        # 4) Memory 생성 (토큰 제한으로 자동 요약)
        if "conversation_memory" not in st.session_state:
            st.session_state.conversation_memory = ConversationSummaryBufferMemory(
                llm=llm,  # 요약을 위한 LLM
                max_token_limit=1000,  # 1000 토큰 초과 시 요약
                memory_key="chat_history",
                return_messages=True
            )
        
        memory = st.session_state.conversation_memory
        
        # 5) Chain 생성 (메모리 + 스트리밍 지원)
        def run_chain_with_memory(question):
            """메모리를 활용한 체인 실행 (스트리밍 지원)"""
            
            # 현재 대화 히스토리 가져오기
            chat_history_data = memory.load_memory_variables({})
            chat_history = chat_history_data.get("chat_history", "이전 대화 내용이 없습니다.")
            
            # 메시지 리스트를 문자열로 변환
            if isinstance(chat_history, list):
                formatted_history = []
                for msg in chat_history:
                    if hasattr(msg, 'content'):
                        role = "사용자" if msg.__class__.__name__ == "HumanMessage" else "AI"
                        formatted_history.append(f"{role}: {msg.content}")
                chat_history = "\n".join(formatted_history) or "이전 대화 내용이 없습니다."
            
            # 컨텍스트 검색
            docs = retriever.get_relevant_documents(question)
            context = format_docs(docs)
            
            # 프롬프트 포맷팅 (chat_history 포함)
            formatted_prompt = prompt.format(
                chat_history=chat_history,
                context=context,
                question=question
            )
            
            return llm, formatted_prompt, memory, question
        
        return run_chain_with_memory
        
    except Exception as e:
        st.error(f"❌ 체인 생성 실패: {str(e)}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 7. 메인 로직: 시스템 초기화 & QA 수행
# ─────────────────────────────────────────────────────────────────────────────

# 시스템 초기화 (최초 1회만)
if not st.session_state["retriever_ready"]:
    with st.spinner("🚀 시스템 준비 중..."):
        retriever = create_retriever()
        if retriever:
            st.session_state["chain"] = create_chain(retriever, selected_model)
            st.session_state["retriever_ready"] = True
        else:
            st.error("❌ 시스템 초기화 실패")
            st.stop()

# 설정 변경 시 체인 재생성
if st.session_state["retriever_ready"]:
    # 새로운 체인 생성 (retriever는 재사용)
    if st.session_state["chain"] is None:
        retriever = create_retriever()  # 캐시된 것이 사용됨
        st.session_state["chain"] = create_chain(retriever, selected_model)

# 초기화 버튼 클릭 시 메시지 리스트와 메모리 초기화 (체인은 유지)
if clear_btn:
    st.session_state["messages"] = []
    # ConversationMemory 초기화
    if "conversation_memory" in st.session_state:
        st.session_state.conversation_memory.clear()
    st.rerun()

# 채팅 영역 (컨테이너로 감싸기)
with st.container():
    # 웰컴 메시지 (대화 기록이 없을 때만 표시)
    if not st.session_state["messages"] and st.session_state["retriever_ready"]:
        st.markdown("""
            <div style="text-align: center; padding: 3rem 1rem; color: #6b7280; background: white; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="color: #374151; margin-bottom: 1rem;">👋 안녕하세요!</h3>
                <p style="margin-bottom: 2rem;">노무 관련 질문을 자유롭게 물어보세요</p>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 2rem;">
                    <div style="background: #f9fafb; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6;">
                        <strong>권고사직과 해고의 차이</strong>
                    </div>
                    <div style="background: #f9fafb; padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981;">
                        <strong>실업급여 신청 방법</strong>
                    </div>
                    <div style="background: #f9fafb; padding: 1rem; border-radius: 8px; border-left: 4px solid #f59e0b;">
                        <strong>근로계약서 작성 방법</strong>
                    </div>
                    <div style="background: #f9fafb; padding: 1rem; border-radius: 8px; border-left: 4px solid #ef4444;">
                        <strong>부당해고 대응 방법</strong>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # 페이지 렌더링마다 이전 대화 기록 출력
    print_messages()

# 사용자 질문 입력
user_input = st.chat_input("노무 관련 질문을 입력하세요... (Shift + Enter로 줄바꿈)")
warning = st.empty()  # 빈 placeholder: 경고 메시지용 

if user_input:
    chain_func = st.session_state.get("chain")  # 저장된 chain 함수 가져오기
    if chain_func is None:
        # chain이 없으면 시스템 초기화 안내
        warning.error("시스템이 초기화되지 않았습니다. 페이지를 새로고침해주세요.")
    else:
        # 1) 사용자 메시지 화면에 쓰기
        st.chat_message("user").write(user_input)
        
        # 2) 메모리와 함께 스트리밍 실행
        try:
            with st.chat_message("assistant"):
                container = st.empty()
                ai_answer = ""
                
                # Chain 함수 실행하여 LLM, 프롬프트, 메모리 가져오기
                llm, formatted_prompt, memory, question = chain_func(user_input)
                
                # 스트리밍 응답
                for token in llm.stream(formatted_prompt):
                    ai_answer += token.content
                    container.markdown(ai_answer + "▌")
                
                # 최종 응답 표시
                container.markdown(ai_answer)
                
                # 메모리에 대화 저장 (토큰 초과 시 자동 요약)
                memory.save_context(
                    {"human": question},
                    {"ai": ai_answer}
                )
                
        except Exception as e:
            st.error(f"❌ 답변 생성 중 오류: {str(e)}")
            ai_answer = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
            
        # 3) 대화기록으로 저장 -> 다음 렌더링 때 출력
        add_message("user", user_input)
        add_message("assistant", ai_answer) 
import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# LCEL 관련 추가 임포트
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableMap
from langchain_core.output_parsers import StrOutputParser

# ConversationSummaryBufferMemory 사용
from langchain.memory import ConversationSummaryBufferMemory

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import logging
from dotenv import load_dotenv
import os
import pickle
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# 1. Streamlit 페이지 설정 (가장 먼저 실행되어야 함)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR RAG 채팅봇",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. 환경 변수 및 로깅 설정
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
logging.langsmith("[Project] HR RAG 채팅봇")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 캐시 디렉토리 준비
# ─────────────────────────────────────────────────────────────────────────────
for sub in [".cache", ".cache/files", ".cache/embeddings"]:
    if not os.path.exists(sub):
        os.mkdir(sub)

st.title("🏢 HR RAG 채팅봇")

# 커스텀 CSS 스타일
st.markdown(
    """
    <style>
    /* 전체 페이지 배경색 및 폰트 설정 */
    body {
        background-color: #1a1a1a;
        color: #e0e0e0;
        font-family: 'AppleSDGothicNeo-Regular', 'Noto Sans KR', sans-serif;
    }

    /* 사이드바 스타일 */
    .st-emotion-cache-jx6q2s {
        background-color: #2b2b2b;
        color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
    }

    /* 채팅 메시지 버블 스타일 */
    .st-chat-message-container {
        border-radius: 15px;
        padding: 10px 15px;
        margin-bottom: 10px; 
    } 
    
    .st-chat-message-container.st-chat-message-assistant {
        background-color: #333333;
        margin-right: auto;
        border-bottom-left-radius: 5px;
        animation: fadeInUp 0.5s ease-out;
    }
    .st-chat-message-container.st-chat-message-user {
        background-color: #004d40;
        margin-left: auto;
        border-bottom-right-radius: 5px;
        animation: fadeInUp 0.3s ease-out;
    }
    
    /* 웰컴메시지 부드러운 페이드 효과 */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeOut {
        from {
            opacity: 1;
            transform: translateY(0);
        }
        to {
            opacity: 0;
            transform: translateY(-10px);
        }
    }
    
    /* 입력창 스타일 */
    .st-emotion-cache-x1bvup {
        background-color: #2b2b2b;
        border-radius: 15px;
        padding: 10px;
        border: none;
    }
    .st-emotion-cache-1c7y2qn textarea {
        background-color: #2b2b2b;
        color: #e0e0e0;
        border: none;
    }
    .st-emotion-cache-sey4o0 {
        background-color: #007bff;
        border-radius: 8px;
        color: white;
    }

    /* 버튼 스타일 */
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0056b3;
        transform: translateY(-1px);
    }
    
    /* 중지 버튼 특별 스타일 */
    button[title="답변 중지"] {
        background-color: #dc3545 !important;
        color: white !important;
        font-size: 16px !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    button[title="답변 중지"]:hover {
        background-color: #c82333 !important;
        transform: scale(1.05) !important;
    }
    
    /* 비활성화된 입력창 스타일 */
    .stTextInput input:disabled {
        background-color: #3a3a3a !important;
        color: #888 !important;
        border: 1px solid #555 !important;
    }
    
    /* 정보 박스 스타일 */
    .st-emotion-cache-1r6dm7m {
        background-color: #3a3a3a;
        color: #b3e0ff;
        border-left: 5px solid #007bff;
        border-radius: 8px;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. 세션 상태 초기화
# ─────────────────────────────────────────────────────────────────────────────

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    st.session_state["chain"] = None

if "retriever_ready" not in st.session_state:
    st.session_state["retriever_ready"] = False

# 답변 중지 플래그 초기화
if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False

# LLM 생성 중인지를 나타내는 플래그 (UI 제어용)
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

# 사용자 입력 처리 중인지를 나타내는 플래그 (중복 방지용)
if "processing_user_input" not in st.session_state:
    st.session_state.processing_user_input = False

# 사용자 입력이 한 번이라도 있었는지 추적하는 플래그 (웰컴 메시지 제어용)
if "user_input_detected" not in st.session_state:
    st.session_state.user_input_detected = False

# ConversationSummaryBufferMemory 시스템 초기화
if "memory_store" not in st.session_state:
    st.session_state["memory_store"] = {}

# 멀티 대화 세션 관리
if "chat_sessions" not in st.session_state:
    st.session_state["chat_sessions"] = {}  # {session_id: {"title": str, "created_at": datetime}}
    
if "current_session_id" not in st.session_state:
    # 첫 세션 자동 생성
    import uuid
    from datetime import datetime
    first_session_id = str(uuid.uuid4())[:8]
    st.session_state["current_session_id"] = first_session_id
    st.session_state["chat_sessions"][first_session_id] = {
        "title": "새로운 대화",
        "created_at": datetime.now()
    }

if "session_counter" not in st.session_state:
    st.session_state["session_counter"] = 1

# 세션 ID를 기반으로 ConversationSummaryBufferMemory를 가져오는 함수
def get_session_memory(session_id: str):
    """세션 ID를 기반으로 ConversationSummaryBufferMemory를 반환"""
    if session_id not in st.session_state["memory_store"]:
        # 임시 LLM으로 메모리 초기화 (나중에 실제 LLM으로 교체)
        temp_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        st.session_state["memory_store"][session_id] = ConversationSummaryBufferMemory(
            llm=temp_llm,
            max_token_limit=1000,
            return_messages=True,
            memory_key="chat_history",
        )
    return st.session_state["memory_store"][session_id]

# 멀티 세션 관리 함수들
def create_new_session():
    """새로운 대화 세션 생성"""
    import uuid
    from datetime import datetime
    
    new_session_id = str(uuid.uuid4())[:8]
    st.session_state["chat_sessions"][new_session_id] = {
        "title": "새로운 대화",
        "created_at": datetime.now()
    }
    st.session_state["current_session_id"] = new_session_id
    st.session_state["messages"] = []  # 새 세션의 UI 메시지 초기화
    st.session_state.user_input_detected = False  # 웰컴 메시지 표시를 위해
    return new_session_id

def delete_session(session_id: str):
    """대화 세션 삭제"""
    if session_id in st.session_state["chat_sessions"]:
        del st.session_state["chat_sessions"][session_id]
    
    if session_id in st.session_state["memory_store"]:
        del st.session_state["memory_store"][session_id]
    
    # 현재 세션이 삭제된 경우 다른 세션으로 전환
    if st.session_state["current_session_id"] == session_id:
        remaining_sessions = list(st.session_state["chat_sessions"].keys())
        if remaining_sessions:
            switch_to_session(remaining_sessions[0])
        else:
            # 모든 세션이 삭제된 경우 새 세션 생성
            create_new_session()

def switch_to_session(session_id: str):
    """다른 세션으로 전환"""
    st.session_state["current_session_id"] = session_id
    
    # 해당 세션의 UI 메시지 로드 (메모리에서 복원)
    if session_id in st.session_state["memory_store"]:
        memory = st.session_state["memory_store"][session_id]
        messages = memory.chat_memory.messages
        
        # UI 메시지 리스트 재구성
        st.session_state["messages"] = []
        for msg in messages:
            role = "user" if msg.type == "human" else "assistant"
            st.session_state["messages"].append(ChatMessage(role=role, content=msg.content))
    else:
        st.session_state["messages"] = []
    
    # 웰컴 메시지 표시 여부 결정
    st.session_state.user_input_detected = len(st.session_state["messages"]) > 0

def generate_title_from_question(question: str):
    """첫 번째 질문을 기반으로 대화 제목 생성"""
    # 간단한 제목 생성 로직
    if len(question) > 30:
        return question[:27] + "..."
    return question

def update_session_title(session_id: str, new_title: str):
    """세션 제목 업데이트"""
    if session_id in st.session_state["chat_sessions"]:
        st.session_state["chat_sessions"][session_id]["title"] = new_title


# ─────────────────────────────────────────────────────────────────────────────
# 5. 사이드바 구성 (ChatGPT 스타일)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # 새 대화 버튼
    if st.button("➕ 새 대화", use_container_width=True, type="primary"):
        create_new_session()
        st.rerun()
    
    st.divider()
    
    # 대화 목록
    st.subheader("💬 대화 목록")
    
    current_session = st.session_state["current_session_id"]
    
    # 대화 세션들을 생성 시간 역순으로 정렬
    sessions = st.session_state["chat_sessions"]
    sorted_sessions = sorted(sessions.items(), 
                           key=lambda x: x[1]["created_at"], 
                           reverse=True)
    
    for session_id, session_data in sorted_sessions:
        title = session_data["title"]
        is_current = session_id == current_session
        
        # 현재 선택된 세션은 다른 스타일로 표시
        if is_current:
            st.markdown(f"**🟢 {title}**")
        else:
            # 대화 선택 버튼과 삭제 버튼을 한 줄에
            col1, col2 = st.columns([0.8, 0.2])
            
            with col1:
                if st.button(title, key=f"select_{session_id}", use_container_width=True):
                    switch_to_session(session_id)
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"delete_{session_id}", help="대화 삭제"):
                    delete_session(session_id)
                    st.rerun()
    
    # 대화가 없는 경우
    if not sessions:
        st.info("아직 대화가 없습니다.")
    
    st.divider()
    
    # 설정 섹션
    st.subheader("⚙️ 설정")

    # 모델 선택
    selected_model = st.selectbox(
        "🤖 LLM 모델",
        ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        index=1,
    )

    # 검색 문서 수
    k = st.number_input("📄 검색할 문서 수", min_value=1, max_value=10, value=3)
    
    st.divider()

    # 현재 세션 메모리 상태 표시
    if current_session in st.session_state.get("memory_store", {}):
        session_memory = st.session_state["memory_store"][current_session]
        # ConversationSummaryBufferMemory의 메시지는 chat_memory.messages에 있음
        message_count = len(session_memory.chat_memory.messages)
        turns = message_count // 2
        st.info(f"💭 현재 대화: {turns}턴 저장중")
    else:
        st.info(f"💭 현재 대화: 새로운 대화")

    # 현재 대화 초기화 버튼
    if st.button("🗑️ 현재 대화 초기화", use_container_width=True):
        if current_session in st.session_state.get("memory_store", {}):
            st.session_state["memory_store"][current_session].clear()
        st.session_state["messages"] = []
        st.session_state.user_input_detected = False
        st.rerun()

    st.divider()

    # 시스템 정보
    st.info(
        """
    **🏢 HR RAG 채팅봇**
    
    노무 관련 질문에 정확한 답변을 제공합니다.
    
    **특징:**
    - 🧠 이전 대화 기억
    - 📚 143개 전문 문서 기반
    - ⚖️ 법적 근거 제시
    - 🔄 멀티턴 대화 지원
    - 💬 다중 대화 세션
    """
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. 헬퍼 함수들
# ─────────────────────────────────────────────────────────────────────────────
def print_messages():
    """저장된 대화 기록을 화면에 출력"""
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def add_message(role: str, content: str):
    """새로운 메시지를 세션에 추가"""
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


def format_docs(docs):
    """검색된 문서를 프롬프트에 주입할 형식으로 포맷팅"""
    return "\n\n".join(doc.page_content for doc in docs)


# ─────────────────────────────────────────────────────────────────────────────
# 7. HR Documents → FAISS Retriever 생성
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def create_retriever():
    """기존에 전처리된 documents.pkl에서 FAISS 검색기를 생성합니다."""

    try:
        # 간단한 경로 해결
        # 1. 현재 파일 기준 경로
        current_dir = os.path.dirname(os.path.abspath(__file__))
        documents_path = os.path.join(current_dir, 'data', 'processed', 'documents.pkl')
        
        # 2. 안 되면 프로젝트 루트 기준
        if not os.path.exists(documents_path):
            project_root = os.path.join(current_dir, '..', '..')  # src/preprocessing에서 2단계 위로
            documents_path = os.path.join(project_root, 'data', 'processed', 'documents.pkl')
        
        # 3. 그래도 안 되면 작업 디렉토리 기준
        if not os.path.exists(documents_path):
            documents_path = os.path.join(os.getcwd(), 'data', 'processed', 'documents.pkl')
        
        if not os.path.exists(documents_path):
            st.error(f"❌ documents.pkl 파일을 찾을 수 없습니다.")
            st.info("💡 해결방법: 터미널에서 '000. Project_rag' 폴더로 이동 후 실행해주세요.")
            return None

        with open(documents_path, 'rb') as f:
            documents = pickle.load(f)

        # 2) 텍스트 분할
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(documents)

        # 3) 임베딩 생성 (OpenAI 우선 사용)
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if openai_api_key:
            embeddings = OpenAIEmbeddings()
        else:
            st.error("❌ API 키가 설정되지 않았습니다.")
            return None

        # 4) FAISS 인덱스 생성
        vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings)

        return vectorstore

    except Exception as e:
        st.error(f"❌ 검색기 생성 중 오류: {str(e)}")
        return None

# get_retriever 함수에서 score_threshold 매개변수 제거
def get_retriever(vectorstore, k=3):
    """Vectorstore에서 검색기를 생성합니다."""
    return vectorstore.as_retriever(search_kwargs={"k": k})


# ─────────────────────────────────────────────────────────────────────────────
# 8. 새로운 체인 생성 함수 (ConversationSummaryBufferMemory 방식)
# ─────────────────────────────────────────────────────────────────────────────
def create_chain(retriever, model="gpt-3.5-turbo"):
    """ConversationSummaryBufferMemory를 사용한 RAG 체인 생성"""
    
    try:
        # 1. 프롬프트 로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, 'prompt', 'qa_prompt.yaml')
        
        if not os.path.exists(prompt_path):
            st.error(f"❌ 프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
            return None

        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_config = yaml.safe_load(f)

        # 프롬프트 설정 (chat_history를 문자열로 처리)
        system_prompt = """당신은 대한민국의 친절하고 유능한 노무 전문가 AI입니다. 

**응답 지침:**
1. 이전 대화 내용을 참고하여 연속적이고 일관된 답변을 제공해주세요.
2. 법적 근거를 명확히 제시해주세요.
3. 실무적인 절차와 방법을 구체적으로 안내해주세요.
4. 주의사항이나 예외 상황도 함께 설명해주세요.

**이전 대화 기록:**
{chat_history}

**참고 문서:**
{context}"""

        # ChatPromptTemplate 생성 (MessagesPlaceholder 없이 단순한 문자열 방식)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])

        # 2. LLM 생성
        if model.startswith("gpt"):
            llm = ChatOpenAI(model=model, temperature=0, streaming=True)
        else:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)

        # 3. 체인 생성 함수 (메모리와 함께 실행)
        def run_chain_with_memory(question: str, session_id: str):
            """메모리를 포함하여 체인 실행"""
            # 현재 세션의 메모리 가져오기
            memory = get_session_memory(session_id)
            
            # 메모리 LLM 업데이트 (현재 선택된 모델로)
            memory.llm = llm
            
            # 메모리에서 대화 기록 가져오기
            memory_variables = memory.load_memory_variables({})
            chat_history = memory_variables.get("chat_history", "")
            
            # 관련 문서 검색
            docs = retriever.get_relevant_documents(question)
            context = format_docs(docs)
            
            # 프롬프트에 변수 전달
            formatted_prompt = prompt.format_messages(
                chat_history=chat_history,
                context=context,
                question=question
            )
            
            # 스트리밍을 위해 필요한 정보 반환
            return {
                "llm": llm,
                "formatted_prompt": formatted_prompt,
                "memory": memory,
                "question": question
            }
        
        return run_chain_with_memory

    except Exception as e:
        st.error(f"❌ 체인 생성 실패: {str(e)}")
        import traceback
        print(f"❌ 체인 생성 오류: {e}")
        print(f"❌ 오류 상세: {traceback.format_exc()}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 9. 메인 로직
# ─────────────────────────────────────────────────────────────────────────────
# 시스템 초기화 (최초 1회만)
if not st.session_state["retriever_ready"]:
    with st.spinner("🚀 시스템 준비 중..."):
        # 캐시 클리어 (임베딩 변경으로 인해)
        st.cache_resource.clear()
        
        vectorstore = create_retriever()
        if vectorstore:
            retriever = get_retriever(vectorstore, k)
            st.session_state["vectorstore"] = vectorstore
            st.session_state["chain"] = create_chain(retriever, selected_model)
            st.session_state["retriever_ready"] = True
        else:
            st.error("❌ 시스템 초기화 실패")
            st.stop()

# 설정 변경 시 체인 재생성
# 사이드바 설정 (k, selected_model) 변경 감지
# 변경이 감지되면 체인을 재생성합니다.
if st.session_state["retriever_ready"]:
    # 이전 설정값들 저장 및 비교
    prev_k = st.session_state.get("prev_k", None)
    prev_model = st.session_state.get("prev_model", None)
    
    # 설정이 변경된 경우에만 체인 재생성
    if prev_k != k or prev_model != selected_model:
        # 🔧 모델 변경 시 UI 상태 플래그 초기화
        st.session_state.is_generating = False
        st.session_state.stop_generation = False
        st.session_state.processing_user_input = False
        
        try:
            with st.spinner("🔄 설정 변경 중..."):
                current_retriever = get_retriever(st.session_state["vectorstore"], k)
                new_chain = create_chain(current_retriever, selected_model)
                
                if new_chain is not None:
                    st.session_state["chain"] = new_chain
                    # 현재 설정값 저장
                    st.session_state["prev_k"] = k
                    st.session_state["prev_model"] = selected_model
                    
                    # 설정 변경 알림 (잠깐 표시)
                    if prev_model is not None and prev_model != selected_model:
                        st.success(f"✅ 모델이 {selected_model}로 변경되었습니다!")
                        # 페이지 새로고침으로 UI 상태 완전 초기화
                        st.rerun()
                else:
                    st.error("❌ 새로운 설정으로 체인 생성에 실패했습니다.")
                    # 이전 모델로 롤백 시도
                    if prev_model and prev_model != selected_model:
                        st.info(f"🔄 이전 모델({prev_model})로 롤백을 시도합니다...")
                        try:
                            rollback_chain = create_chain(current_retriever, prev_model)
                            if rollback_chain is not None:
                                st.session_state["chain"] = rollback_chain
                                st.warning(f"⚠️ {prev_model} 모델로 롤백되었습니다.")
                            else:
                                st.error("❌ 롤백도 실패했습니다. 페이지를 새로고침해주세요.")
                        except Exception as rollback_error:
                            st.error(f"❌ 롤백 중 오류: {str(rollback_error)}")
        except Exception as e:
            st.error(f"❌ 설정 변경 중 오류: {str(e)}")
            # 특정 오류 메시지에 따른 안내
            if "duplicate validator" in str(e):
                st.info("💡 라이브러리 충돌이 발생했습니다. 페이지를 새로고침하거나 다른 모델을 선택해주세요.")
            
            # 오류 시에도 상태 초기화
            st.session_state.is_generating = False
            st.session_state.stop_generation = False
            st.session_state.processing_user_input = False
    else:
        # 첫 실행 시 설정값 저장
        if prev_k is None:
            st.session_state["prev_k"] = k
            st.session_state["prev_model"] = selected_model


# 이전 대화 기록 출력
def print_previous_messages():
    """현재 세션의 이전 메시지들을 출력"""
    messages_to_show = st.session_state["messages"]
    
    # 사용자 입력 처리 중이면 마지막 사용자 메시지는 제외 (중복 방지)
    # 왜냐하면 이 메시지는 사용자 입력 시점에 이미 직접 출력되기 때문
    if st.session_state.get('processing_user_input', False):
        if len(messages_to_show) > 0 and messages_to_show[-1].role == "user":
            messages_to_show = messages_to_show[:-1]
    
    for chat_message in messages_to_show:
        st.chat_message(chat_message.role).write(chat_message.content)

# 이전 대화 기록 표시
print_previous_messages()

# 웰컴 메시지 표시 여부 결정 (더 즉각적인 반응을 위해)
show_welcome = (len(st.session_state["messages"]) == 0 and 
                st.session_state.get("retriever_ready", False) and 
                not st.session_state.get("user_input_detected", False))

# 첫 방문 시 웰컴 메시지
if show_welcome:
    with st.chat_message("assistant"):
        st.markdown(
            """
            <div style="animation: fadeInUp 0.8s ease-out;">
            
            안녕하세요! 🏢 **HR RAG 채팅봇**입니다.
            
            노무 관련 질문을 자유롭게 물어보세요:
            - 권고사직과 해고의 차이
            - 실업급여 신청 방법  
            - 근로계약서 작성 방법
            - 기타 노무 관련 법률 문의
            
            언제든지 도움이 필요하시면 질문해주세요! 💬
            
            </div>
            """,
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────────────────────
# 동적 사용자 입력 (Send ↔ Stop 버튼 전환)
# ─────────────────────────────────────────────────────────────
user_input = None

if st.session_state.get('is_generating', False):
    # LLM 답변 중: 입력창을 비활성화하고 중지 버튼으로 변경
    col1, col2 = st.columns([0.85, 0.15])
    
    with col1:
        st.text_input(
            "노무 관련 질문을 입력하세요...", 
            value="답변 생성 중...", 
            disabled=True,
            key="disabled_input",
            label_visibility="collapsed"
        )
    
    with col2:
        if st.button("⏹️", key="stop_button", help="답변 중지", use_container_width=True, type="primary"):
            st.session_state.stop_generation = True
            st.session_state.is_generating = False
            st.session_state.processing_user_input = False
            st.info("답변 생성을 중지했습니다.")
            st.rerun()
            
    # 🆘 추가 안전장치: 강제 초기화 버튼
    if st.button("🔄 상태 초기화", key="force_reset", help="UI 상태를 강제로 초기화합니다"):
        st.session_state.is_generating = False
        st.session_state.stop_generation = False
        st.session_state.processing_user_input = False
        st.success("✅ 상태가 초기화되었습니다!")
        st.rerun()
else:
    # 일반 상태: 표준 chat_input 사용
    user_input = st.chat_input("노무 관련 질문을 입력하세요...", key="chat_input")

# 사용자 입력 처리
if user_input:
    # 🔥 웰컴메시지 즉시 숨기기 (가장 먼저 실행)
    st.session_state.user_input_detected = True
    
    # 플래그 설정
    st.session_state.stop_generation = False
    st.session_state.is_generating = True
    st.session_state.processing_user_input = True

    # 체인 가져오기
    chain = st.session_state.get("chain")
    if chain is None:
        st.error("❌ 시스템이 초기화되지 않았습니다.")
        st.info("💡 모델을 다시 선택하거나 페이지를 새로고침해주세요.")
        
        # 자동 체인 재생성 시도
        if st.session_state.get("retriever_ready", False):
            try:
                with st.spinner("🔄 시스템 재초기화 중..."):
                    current_retriever = get_retriever(st.session_state["vectorstore"], k)
                    new_chain = create_chain(current_retriever, selected_model)
                    
                    if new_chain is not None:
                        st.session_state["chain"] = new_chain
                        st.success("✅ 시스템이 재초기화되었습니다. 다시 질문해주세요!")
                        st.rerun()
                    else:
                        st.error("❌ 시스템 재초기화에 실패했습니다.")
            except Exception as e:
                st.error(f"❌ 재초기화 중 오류: {str(e)}")
        
        st.stop()

    # 사이드바에서 설정된 current_session_id 사용
    current_session_id = st.session_state["current_session_id"]

    # 1. 사용자 메시지 저장 및 즉시 화면에 표시
    add_message("user", user_input)
    
    # 사용자 메시지를 즉시 화면에 표시 (AI 응답 기다리지 않음)
    with st.chat_message("user"):
        st.write(user_input)

    # 2. AI 응답 생성 및 스트리밍 표시
    ai_answer = ""
    with st.chat_message("assistant"):
        container = st.empty()
        generation_stopped = False # 실제로 중단되었는지 여부를 추적

        try:
            # ConversationSummaryBufferMemory 체인 실행 (스트리밍)
            # 사이드바에서 설정된 current_session_id 사용
            current_session_id = st.session_state["current_session_id"]

            # 체인 함수 호출하여 필요한 정보 가져오기
            chain_info = chain(user_input, current_session_id)
            llm = chain_info["llm"]
            formatted_prompt = chain_info["formatted_prompt"]
            memory = chain_info["memory"]
            question = chain_info["question"]
            
            # 스트리밍 응답
            for chunk in llm.stream(formatted_prompt):
                if st.session_state.stop_generation:
                    generation_stopped = True
                    break # 중지 플래그가 True이면 루프를 즉시 중단
                
                ai_answer += chunk.content
                
                # 스트리밍 UI 업데이트
                container.markdown(ai_answer + "▌") # 현재까지의 답변 + 커서 표시

            # 최종 응답 표시 또는 중지 메시지
            if generation_stopped:
                container.markdown(ai_answer + "\n\n**답변 생성이 중지되었습니다.**")
            else:
                container.markdown(ai_answer)  # 최종 답변 (커서 제거)

            # 메모리에 대화 저장 (완전한 답변만 저장)
            if not generation_stopped and ai_answer:
                memory.save_context(
                    inputs={"human": question},
                    outputs={"ai": ai_answer}
                )
                
                # 첫 번째 질문인 경우 대화 제목 자동 생성
                if len(memory.chat_memory.messages) == 2:  # human + ai = 2개 (첫 번째 대화)
                    current_title = st.session_state["chat_sessions"][current_session_id]["title"]
                    if current_title == "새로운 대화":  # 기본 제목인 경우만 업데이트
                        new_title = generate_title_from_question(question)
                        update_session_title(current_session_id, new_title)
                
                add_message("assistant", ai_answer)

        except Exception as e:
            st.error(f"❌ 답변 생성 중 오류: {str(e)}")
            ai_answer = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
            container.markdown(ai_answer)
            generation_stopped = True # 오류 발생 시에도 저장하지 않도록 처리

        finally:
            # LLM 생성 완료 또는 중단 시 is_generating 플래그를 False로 설정
            st.session_state.is_generating = False
            st.session_state.processing_user_input = False # 사용자 입력 처리 중 플래그 해제
            st.rerun() # UI를 업데이트하여 중지 버튼 숨기기
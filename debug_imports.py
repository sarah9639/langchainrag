#!/usr/bin/env python3
"""
qa_chain.py의 import 문제 진단 스크립트
"""

print("🔍 qa_chain.py import 문제 진단 시작!")
print("=" * 50)

# 1. 기본 패키지들 체크
basic_packages = [
    "os",
    "dotenv", 
    "langchain_openai",
    "langchain_core.prompts",
    "langchain_community.callbacks",
    "langchain_core.runnables",
    "langchain_core.output_parsers",
    "operator"
]

for pkg in basic_packages:
    try:
        __import__(pkg)
        print(f"✅ {pkg} - 정상")
    except ImportError as e:
        print(f"❌ {pkg} - 실패: {e}")

print("\n" + "=" * 50)

# 2. 문제될 수 있는 패키지들 체크
print("🚨 문제 가능성 있는 패키지들:")

try:
    from langchain_teddynote import logging
    print("✅ langchain_teddynote - 정상")
except ImportError as e:
    print(f"❌ langchain_teddynote - 실패: {e}")
    print("💡 해결방법: pip install langchain-teddynote")

try:
    import yaml
    print("✅ yaml - 정상")
except ImportError as e:
    print(f"❌ yaml - 실패: {e}")
    print("💡 해결방법: pip install pyyaml")

print("\n" + "=" * 50)

# 3. retriever 모듈 체크
print("🔍 retriever 모듈 import 체크:")

try:
    # 현재 디렉토리에서 직접 import 시도
    from retriever import initialize_retriever
    print("✅ retriever 모듈 - 정상")
except ImportError as e:
    print(f"❌ retriever 모듈 - 실패: {e}")
    print("💡 문제: qa_chain.py에서 'from retriever import'는 같은 폴더의 retriever.py를 찾음")

try:
    # 상대 경로로 import 시도  
    from .retriever import initialize_retriever
    print("✅ 상대 경로 retriever 모듈 - 정상")
except ImportError as e:
    print(f"❌ 상대 경로 retriever 모듈 - 실패: {e}")

print("\n" + "=" * 50)

# 4. 필요한 파일들 존재 확인
print("📁 필요한 파일/폴더 존재 확인:")

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

files_to_check = [
    os.path.join(current_dir, "retriever.py"),
    os.path.join(project_root, 'src', 'prompt', 'qa_prompt.yaml'),
    os.path.join(project_root, 'data', 'processed', 'documents.pkl'),
    ".env"
]

for file_path in files_to_check:
    if os.path.exists(file_path):
        print(f"✅ {file_path} - 존재")
    else:
        print(f"❌ {file_path} - 없음")

print("\n" + "=" * 50)

# 5. 환경변수 확인
print("🔑 환경변수 확인:")

from dotenv import load_dotenv
load_dotenv()

api_keys = ["OPENAI_API_KEY", "UPSTAGE_API_KEY"]
for key in api_keys:
    value = os.getenv(key)
    if value:
        print(f"✅ {key} - 설정됨 ({value[:8]}...)")
    else:
        print(f"❌ {key} - 없음")

print("\n🎯 진단 완료!") 
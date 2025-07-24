import re
import unicodedata

def clean_text(text: str) -> str:
    """
    텍스트를 정규화하고 불필요한 문자를 제거합니다.
    
    Args:
        text (str): 정규화할 텍스트
        
    Returns:
        str: 정규화된 텍스트
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 1. 유니코드 정규화 (NFKC: 호환성 분해 후 정준 결합)
    text = unicodedata.normalize('NFKC', text)
    
    # 2. HTML 엔티티 제거 (기본적인 것들)
    html_entities = {
        '&nbsp;': ' ',
        '&lt;': '<',
        '&gt;': '>',
        '&amp;': '&',
        '&quot;': '"',
        '&#39;': "'",
        '&apos;': "'"
    }
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)
    
    # 3. 특수 문자 및 제어 문자 정리
    # 제어 문자 제거 (탭, 개행 제외)
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    
    # 4. 공백 정리
    # 연속된 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    
    # 5. 불필요한 문자 패턴 제거
    # 연속된 특수문자 정리
    text = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ.,!?;:()\[\]{}"-]+', ' ', text)
    
    # 6. 양쪽 공백 제거
    text = text.strip()
    
    # 7. 최소 길이 체크 (너무 짧은 텍스트 제거)
    if len(text) < 2:
        return ""
    
    return text

def normalize_korean_text(text: str) -> str:
    """
    한국어 텍스트의 특수한 정규화를 수행합니다.
    
    Args:
        text (str): 정규화할 한국어 텍스트
        
    Returns:
        str: 정규화된 한국어 텍스트
    """
    if not text:
        return ""
    
    # 1. 한글 자모 분리 해결
    text = unicodedata.normalize('NFC', text)
    
    # 2. 한국어 특수 문자 처리
    # 전각 문자를 반각으로
    text = text.replace('：', ':')
    text = text.replace('（', '(')
    text = text.replace('）', ')')
    text = text.replace('［', '[')
    text = text.replace('］', ']')
    text = text.replace('｛', '{')
    text = text.replace('｝', '}')
    text = text.replace('「', '"')
    text = text.replace('」', '"')
    text = text.replace('『', '"')
    text = text.replace('』', '"')
    
    # 3. 한국어 조사/어미 정리를 위한 기본 처리
    # (고급 형태소 분석은 별도 라이브러리 필요)
    
    return text

def remove_personal_info(text: str) -> str:
    """
    개인정보로 보이는 패턴을 마스킹합니다.
    
    Args:
        text (str): 처리할 텍스트
        
    Returns:
        str: 개인정보가 마스킹된 텍스트
    """
    if not text:
        return ""
    
    # 주민등록번호 패턴 마스킹
    text = re.sub(r'\d{6}-\d{7}', '[주민등록번호]', text)
    text = re.sub(r'\d{13}', '[주민등록번호]', text)
    
    # 전화번호 패턴 마스킹
    text = re.sub(r'\d{2,3}-\d{3,4}-\d{4}', '[전화번호]', text)
    text = re.sub(r'\d{3}-\d{4}-\d{4}', '[전화번호]', text)
    
    # 이메일 패턴 마스킹
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[이메일]', text)
    
    # 계좌번호 패턴 마스킹 (기본적인 패턴)
    text = re.sub(r'\d{3}-\d{2}-\d{6}', '[계좌번호]', text)
    text = re.sub(r'\d{4}-\d{4}-\d{4}-\d{4}', '[계좌번호]', text)
    
    return text

def extract_numbers_and_dates(text: str) -> dict:
    """
    텍스트에서 숫자와 날짜 정보를 추출합니다.
    
    Args:
        text (str): 분석할 텍스트
        
    Returns:
        dict: 추출된 숫자와 날짜 정보
    """
    if not text:
        return {}
    
    result = {
        'numbers': [],
        'dates': [],
        'amounts': []
    }
    
    # 숫자 추출 (콤마 포함)
    numbers = re.findall(r'\d{1,3}(?:,\d{3})*', text)
    result['numbers'] = numbers
    
    # 날짜 패턴 추출
    date_patterns = [
        r'\d{4}[-./]\d{1,2}[-./]\d{1,2}',  # YYYY-MM-DD 형태
        r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일',  # YYYY년 MM월 DD일
        r'\d{1,2}[-./]\d{1,2}[-./]\d{4}',  # MM-DD-YYYY 형태
    ]
    
    for pattern in date_patterns:
        dates = re.findall(pattern, text)
        result['dates'].extend(dates)
    
    # 금액 패턴 추출 (원, 만원, 억원 등)
    amount_patterns = [
        r'\d{1,3}(?:,\d{3})*\s*원',
        r'\d+\s*만\s*원',
        r'\d+\s*억\s*원',
        r'\d+\s*천\s*원'
    ]
    
    for pattern in amount_patterns:
        amounts = re.findall(pattern, text)
        result['amounts'].extend(amounts)
    
    return result

def standardize_spacing(text: str) -> str:
    """
    한국어 텍스트의 띄어쓰기를 표준화합니다.
    
    Args:
        text (str): 처리할 텍스트
        
    Returns:
        str: 띄어쓰기가 표준화된 텍스트
    """
    if not text:
        return ""
    
    # 기본적인 띄어쓰기 규칙 적용
    # 조사 앞 띄어쓰기 제거
    text = re.sub(r'\s+(은|는|이|가|을|를|에|의|와|과|로|으로|부터|까지|에서)', r'\1', text)
    
    # 숫자와 단위 사이 띄어쓰기 정리
    text = re.sub(r'(\d+)\s*(원|만원|억원|개|명|년|월|일|시|분)', r'\1\2', text)
    
    # 문장부호 앞뒤 띄어쓰기 정리
    text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
    text = re.sub(r'\s*([()])\s*', r' \1', text)
    
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip() 
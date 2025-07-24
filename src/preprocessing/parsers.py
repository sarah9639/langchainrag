import pandas as pd
import os
from typing import List, Dict, Any

def parse_excel_for_hr_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Excel/CSV 파일을 읽어서 HR 관련 데이터를 파싱합니다.
    각 행을 하나의 딕셔너리로 변환하여 리스트로 반환합니다.
    
    Args:
        file_path (str): Excel/CSV 파일 경로
        
    Returns:
        List[Dict[str, Any]]: 파싱된 데이터 리스트
    """
    parsed_data_list = []
    
    try:
        # 파일 확장자에 따라 다른 방법으로 읽기
        if file_path.endswith('.csv'):
            # CSV 파일 읽기 (인코딩 자동 감지)
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='cp949')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='euc-kr')
        else:
            # Excel 파일 읽기
            df = pd.read_excel(file_path, sheet_name=None)  # 모든 시트 읽기
            
            # 여러 시트가 있는 경우 모든 시트 처리
            if isinstance(df, dict):
                all_sheets_data = []
                for sheet_name, sheet_df in df.items():
                    sheet_data = process_dataframe(sheet_df, file_path, sheet_name)
                    all_sheets_data.extend(sheet_data)
                return all_sheets_data
        
        # DataFrame 처리
        parsed_data_list = process_dataframe(df, file_path)
        
    except Exception as e:
        print(f"오류: 파일 '{file_path}' 파싱 중 오류 발생: {e}")
        return []
    
    return parsed_data_list

def process_dataframe(df: pd.DataFrame, file_path: str, sheet_name: str = None) -> List[Dict[str, Any]]:
    """
    DataFrame을 처리하여 Document 형태로 변환 가능한 데이터로 만듭니다.
    
    Args:
        df (pd.DataFrame): 처리할 DataFrame
        file_path (str): 원본 파일 경로
        sheet_name (str): 시트 이름 (선택사항)
        
    Returns:
        List[Dict[str, Any]]: 처리된 데이터 리스트
    """
    processed_data = []
    
    # NaN 값을 빈 문자열로 대체
    df = df.fillna('')
    
    # 각 행을 처리
    for index, row in df.iterrows():
        # 행의 모든 값을 텍스트로 결합
        row_text_parts = []
        for col, value in row.items():
            if pd.notna(value) and str(value).strip():
                # 컬럼명과 값을 함께 저장
                row_text_parts.append(f"{col}: {str(value).strip()}")
        
        # 유효한 텍스트가 있는 경우에만 추가
        if row_text_parts:
            row_text = " | ".join(row_text_parts)
            
            # 메타데이터 생성
            metadata = {
                'source': os.path.basename(file_path),
                'source_type': 'excel' if file_path.endswith(('.xlsx', '.xls')) else 'csv',
                'row_index': index,
                'file_path': file_path
            }
            
            if sheet_name:
                metadata['sheet_name'] = sheet_name
            
            # 각 컬럼의 개별 값도 메타데이터에 저장
            for col, value in row.items():
                if pd.notna(value) and str(value).strip():
                    metadata[f'column_{col}'] = str(value).strip()
            
            processed_data.append({
                'text': row_text,
                'metadata': metadata
            })
    
    return processed_data

def extract_hr_keywords(text: str) -> List[str]:
    """
    텍스트에서 HR 관련 키워드를 추출합니다.
    
    Args:
        text (str): 분석할 텍스트
        
    Returns:
        List[str]: 추출된 키워드 리스트
    """
    hr_keywords = [
        # 근로관계
        '근로계약', '고용계약', '채용', '입사', '퇴사', '해고', '정리해고', '권고사직',
        # 급여/복리후생
        '급여', '임금', '수당', '상여금', '퇴직금', '연차', '휴가', '복리후생',
        # 근무시간/휴게
        '근무시간', '연장근로', '야간근로', '휴일근로', '휴게시간', '유연근무',
        # 산업안전
        '산업안전', '안전교육', '산업재해', '업무상재해',
        # 노동관계
        '노동조합', '단체협약', '노사협의', '쟁의행위',
        # 사회보험
        '국민연금', '건강보험', '고용보험', '산재보험', '실업급여'
    ]
    
    found_keywords = []
    text_lower = text.lower()
    
    for keyword in hr_keywords:
        if keyword.lower() in text_lower:
            found_keywords.append(keyword)
    
    return found_keywords 
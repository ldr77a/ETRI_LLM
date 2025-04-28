import os
import sys
import warnings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai
import traceback
import glob
import json
import csv
import pandas as pd
from langchain_core.documents import Document
import random

# 모든 경고 메시지를 완전히 억제
warnings.filterwarnings("ignore")

# 환경 변수 설정
API_KEY = "AIzaSyAMIlzkwZyyiUoDbQg_bL8j_pFFVasB3II"
os.environ["GOOGLE_API_KEY"] = API_KEY
genai.configure(api_key=API_KEY)

# 간단한 구현을 위한 임포트
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 상태 업데이트를 위한 깔끔한 출력
def print_status(msg, is_error=False):
    """상태 메시지 출력"""
    prefix = "[오류]" if is_error else "[정보]"
    print(f"{prefix} {msg}")

def extract_json_content(json_data):
    """JSON 데이터에서 텍스트 내용 추출"""
    content = []
    
    def process_value(value, prefix=""):
        if isinstance(value, dict):
            for k, v in value.items():
                process_value(v, f"{prefix}{k}: ")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                process_value(item, f"{prefix}항목 {i+1}: ")
        else:
            content.append(f"{prefix}{value}")
    
    process_value(json_data)
    return "\n".join(content)

def extract_csv_content(file_path):
    """CSV 파일에서 텍스트 내용 추출"""
    try:
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        # 아이템 관련 CSV 파일 특별 처리 (item_name 컬럼이 있는 파일)
        if "items" in file_name.lower() or "item" in file_name.lower():
            try:
                print_status(f"아이템 정보 CSV 파일 처리 중: {file_name}")
                
                # pandas로 전체 데이터 로드
                df = pd.read_csv(file_path, encoding='utf-8')
                
                # 중요 컬럼 확인
                has_item_name = 'item_name' in df.columns
                has_source = 'source' in df.columns
                has_quality = 'quality' in df.columns
                
                if has_item_name:
                    # 아이템 정보 추출을 위한 컬럼 이름 리스트
                    detail_columns = []
                    stat_columns = []
                    
                    # 컬럼 분류
                    for col in df.columns:
                        if col.startswith('stat_') or col.startswith('armor_') or 'effect' in col.lower():
                            stat_columns.append(col)
                        elif col not in ['item_id', 'item_name', 'source']:
                            detail_columns.append(col)
                    
                    # 각 아이템 정보 추출 및 구조화
                    all_items = []
                    for i, row in df.iterrows():
                        item_info = {
                            "name": row['item_name'] if has_item_name else f"아이템 {i+1}",
                            "details": {},
                            "stats": {},
                            "source": row['source'] if has_source else "정보 없음"
                        }
                        
                        # 상세 정보 추가
                        for col in detail_columns:
                            if pd.notna(row[col]) and row[col] != '':
                                item_info["details"][col] = row[col]
                        
                        # 스탯 정보 추가
                        for col in stat_columns:
                            if pd.notna(row[col]) and row[col] != '' and row[col] != 0:
                                # 컬럼명에서 stat_ 접두사 제거하여 스탯 이름 추출
                                stat_name = col
                                if col.startswith('stat_'):
                                    stat_name = col[5:]  # 'stat_' 접두사 제거
                                item_info["stats"][stat_name] = row[col]
                        
                        all_items.append(item_info)
                    
                    # 결과를 문자열로 변환
                    content = [f"아이템 정보 파일: {file_name}"]
                    content.append(f"총 아이템 수: {len(all_items)}개")
                    content.append("")
                    
                    # 각 아이템 정보 포맷팅
                    for i, item in enumerate(all_items):
                        content.append(f"[아이템 {i+1}] {item['name']}")
                        
                        if has_quality and 'quality' in item['details']:
                            content.append(f"품질: {item['details'].get('quality', '정보 없음')}")
                            
                        content.append(f"획득 방법: {item['source'] if item['source'] else '정보 없음'}")
                        
                        # 아이템 레벨 및 요구 레벨 정보
                        if 'item_level' in item['details']:
                            content.append(f"아이템 레벨: {item['details'].get('item_level', '정보 없음')}")
                        if 'required_level' in item['details']:
                            content.append(f"요구 레벨: {item['details'].get('required_level', '정보 없음')}")
                        
                        # 아이템 종류 정보
                        item_types = []
                        if 'inventory_type' in item['details']:
                            item_types.append(str(item['details']['inventory_type']))
                        if 'item_class' in item['details']:
                            item_types.append(str(item['details']['item_class']))
                        if 'item_subclass' in item['details']:
                            item_types.append(str(item['details']['item_subclass']))
                        
                        if item_types:
                            content.append(f"아이템 종류: {' / '.join(item_types)}")
                        
                        # 스탯 정보 추가
                        if item['stats']:
                            content.append("스탯:")
                            for stat_name, stat_value in item['stats'].items():
                                # 스탯 이름을 한글로 변환
                                kr_stat_name = stat_name
                                if stat_name == 'Strength':
                                    kr_stat_name = '힘'
                                elif stat_name == 'Agility':
                                    kr_stat_name = '민첩성'
                                elif stat_name == 'Intellect':
                                    kr_stat_name = '지능'
                                elif stat_name == 'Stamina':
                                    kr_stat_name = '체력'
                                elif stat_name == 'Haste':
                                    kr_stat_name = '가속'
                                elif stat_name == 'Critical Strike':
                                    kr_stat_name = '치명타'
                                elif stat_name == 'Mastery':
                                    kr_stat_name = '특화'
                                elif stat_name == 'Versatility':
                                    kr_stat_name = '유연성'
                                elif stat_name == 'Dodge':
                                    kr_stat_name = '회피'
                                elif 'armor_value' in stat_name:
                                    kr_stat_name = '방어도'
                                
                                content.append(f"  - {kr_stat_name}: {stat_value}")
                        
                        # 효과 정보 추가
                        for key, value in item['details'].items():
                            if 'effect' in key.lower() and pd.notna(value) and value != '':
                                content.append(f"효과: {value}")
                        
                        content.append("-" * 40)
                    
                    return "\n".join(content)
                else:
                    # item_name 컬럼이 없는 경우 일반 처리
                    return process_generic_csv(file_path, file_name, file_size, df)
            except Exception as e:
                print_status(f"아이템 CSV 처리 중 오류: {str(e)}", True)
                traceback.print_exc()
                # 일반 CSV 처리로 폴백
                return process_generic_csv(file_path, file_name, file_size)
        
        # PlayerState 및 기타 대용량 CSV 파일 처리
        return process_generic_csv(file_path, file_name, file_size)
        
    except Exception as e:
        print_status(f"CSV 파싱 오류: {str(e)}", True)
        traceback.print_exc()
        
        # 대안: csv 모듈 사용
        try:
            content = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                
                content.append(f"파일명: {file_name}")
                content.append(f"컬럼: {', '.join(headers)}")
                
                # 파일 크기에 따라 처리 방법 결정
                if file_size > 1 * 1024 * 1024:  # 1MB
                    # 대용량 파일은 샘플링
                    lines = []
                    for i, row in enumerate(reader):
                        if i < 50:  # 처음 50개 행 저장
                            lines.append((i+1, row))
                        elif i >= 50 and random.random() < 0.01:  # 그 이후 1% 확률로 샘플링
                            lines.append((i+1, row))
                        if i > 5000:  # 최대 5000행까지만 읽기
                            break
                    
                    content.append(f"데이터 크기: {file_size/1024/1024:.2f}MB")
                    content.append(f"샘플링된 행: {len(lines)}개")
                    content.append("")
                    
                    # 샘플 데이터 표시
                    for i, row in lines[:100]:  # 최대 100개만 표시
                        cols = []
                        for j, val in enumerate(row):
                            if j < len(headers):
                                cols.append(f"{headers[j]}={val}")
                        content.append(f"행 {i}: {' | '.join(cols[:10])}...")
                else:
                    # 작은 파일은 전체 처리
                    for i, row in enumerate(reader):
                        if i >= 100:  # 최대 100행까지만
                            content.append(f"... 그 외 행 생략 ...")
                            break
                        cols = []
                        for j, val in enumerate(row):
                            if j < len(headers):
                                cols.append(f"{headers[j]}={val}")
                        content.append(f"행 {i+1}: {' | '.join(cols)}")
            
            return "\n".join(content)
        except Exception as e2:
            print_status(f"기본 CSV 파싱 오류: {str(e2)}", True)
            # 최종 대안: 파일을 텍스트로 읽기
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                head = f.readline()  # 헤더 읽기
                sample = f.read(50000)  # 샘플 데이터 읽기 (50KB)
                return f"파일명: {file_name}\n헤더: {head}\n\n샘플 데이터:\n{sample}..."

def process_generic_csv(file_path, file_name, file_size, df=None):
    """일반 CSV 파일 처리"""
    try:
        # 이미 로드된 DataFrame이 있으면 사용, 없으면 새로 로드
        if df is None:
            if file_size > 5 * 1024 * 1024:  # 5MB 이상은 청크로 처리
                # 대용량 파일은 청크 단위로 처리
                try:
                    # 헤더만 먼저 읽기
                    with open(file_path, 'r', encoding='utf-8') as f:
                        header = f.readline().strip()
                    
                    # 청크 크기 설정 (파일 크기에 따라 조정)
                    chunk_size = 500 if file_size < 10 * 1024 * 1024 else 100
                    content = []
                    
                    # 헤더 추가
                    header_columns = [col.strip() for col in header.split(',')]
                    content.append(f"파일명: {file_name}")
                    content.append(f"컬럼: {', '.join(header_columns)}")
                    content.append(f"데이터 크기: {file_size/1024/1024:.2f}MB")
                    content.append("")
                    
                    # 파일의 처음, 중간, 끝부분에서 각각 샘플 추출
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # 헤더 건너뛰기
                        f.readline()
                        
                        # 처음 부분 데이터 (최대 100행)
                        content.append("--- 처음 데이터 샘플 ---")
                        for i in range(min(100, chunk_size)):
                            line = f.readline().strip()
                            if not line:
                                break
                            content.append(f"행 {i+1}: {line}")
                        
                        # 중간 부분 데이터를 찾기 위해 파일 크기의 중간 지점으로 이동
                        if file_size > 500 * 1024:  # 500KB 이상인 경우만
                            f.seek(file_size // 2)
                            # 다음 라인의 시작으로 이동
                            f.readline()
                            
                            content.append("")
                            content.append("--- 중간 데이터 샘플 ---")
                            for i in range(min(50, chunk_size // 2)):
                                line = f.readline().strip()
                                if not line:
                                    break
                                content.append(f"중간 행: {line}")
                        
                        # 끝 부분 데이터
                        if file_size > 200 * 1024:  # 200KB 이상인 경우만
                            # 파일 끝부분으로 이동 (대략적으로 마지막 10KB)
                            f.seek(max(0, file_size - 10 * 1024))
                            # 다음 라인의 시작으로 이동
                            f.readline()
                            
                            content.append("")
                            content.append("--- 끝 부분 데이터 샘플 ---")
                            lines = []
                            for i in range(min(50, chunk_size // 2)):
                                line = f.readline().strip()
                                if not line:
                                    continue
                                lines.append(line)
                            
                            # 마지막 몇 개 라인만 추가
                            for line in lines[-min(50, chunk_size // 2):]:
                                content.append(f"끝 부분 행: {line}")
                    
                    return "\n".join(content)
                    
                except Exception as e:
                    print_status(f"대용량 CSV 처리 중 오류: {str(e)}", True)
                    # pandas로 대체 시도
                    try:
                        # 큰 파일은 chunk로 읽기
                        chunks = pd.read_csv(file_path, encoding='utf-8', chunksize=chunk_size)
                        first_chunk = next(chunks)
                        
                        content = []
                        content.append(f"파일명: {file_name}")
                        content.append(f"컬럼: {', '.join(first_chunk.columns)}")
                        content.append(f"데이터 크기: {file_size/1024/1024:.2f}MB")
                        content.append("")
                        content.append("--- 처음 데이터 샘플 ---")
                        
                        # 샘플 데이터 추가 (처음 50행)
                        for i, row in enumerate(first_chunk.itertuples()):
                            if i >= 50:
                                break
                            values = []
                            for col in first_chunk.columns:
                                if hasattr(row, col):
                                    values.append(f"{col}={getattr(row, col)}")
                            content.append(f"행 {i+1}: {' | '.join(values)}")
                        
                        return "\n".join(content)
                    except Exception as e2:
                        print_status(f"Pandas CSV 처리 중 오류: {str(e2)}", True)
                        # 최종 대안: 텍스트로 읽기
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            return f"파일명: {file_name}\n크기: {file_size/1024/1024:.2f}MB\n\n헤더: {f.readline()}\n\n샘플 데이터:\n{f.read(50000)}"
            else:
                # 작은 CSV 파일은 pandas로 전체 처리
                df = pd.read_csv(file_path, encoding='utf-8')
        
        # DataFrame 처리
        content = []
        
        # 기본 정보 추가
        content.append(f"파일명: {file_name}")
        content.append(f"행 수: {len(df)}")
        content.append(f"컬럼: {', '.join(df.columns)}")
        content.append("")
        
        # 데이터 요약 정보 추가
        try:
            # 수치형 컬럼에 대한 기술 통계
            numeric_columns = df.select_dtypes(include=['number']).columns
            if len(numeric_columns) > 0:
                content.append("수치형 컬럼 통계:")
                for col in numeric_columns[:10]:  # 최대 10개 컬럼만
                    stats = df[col].describe()
                    content.append(f"  {col}: 평균={stats['mean']:.2f}, 최소={stats['min']:.2f}, 최대={stats['max']:.2f}")
                content.append("")
        except:
            pass
        
        # 각 행 데이터 추가 (최대 100행)
        content.append("데이터 샘플:")
        max_rows = min(100, len(df))
        for i, row in enumerate(df.itertuples()):
            if i >= max_rows:
                break
            values = []
            for col in df.columns:
                if hasattr(row, col):
                    values.append(f"{col}={getattr(row, col)}")
            content.append(f"행 {i+1}: {' | '.join(values[:10])}...")  # 컬럼 일부만 표시
        
        if len(df) > max_rows:
            content.append(f"... 그 외 {len(df) - max_rows}개 행 생략 ...")
        
        return "\n".join(content)
    except Exception as e:
        print_status(f"일반 CSV 처리 중 오류: {str(e)}", True)
        traceback.print_exc()
        return f"파일 처리 실패: {str(e)}"

def load_data():
    """모든 데이터 로드 (JSON 및 CSV)"""
    try:
        all_documents = []
        
        # 텍스트 분할 설정
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # 1. JSON 파일 로드 - Datajsonfile 디렉토리
        data_dir = "Datajsonfile"
        if os.path.exists(data_dir):
            json_files = glob.glob(os.path.join(data_dir, "*.json"))
            
            if json_files:
                print_status(f"발견된 JSON 파일: {len(json_files)}개")
                for file in json_files:
                    print_status(f"  - {os.path.basename(file)}")
                
                # 모든 JSON 파일 로드 및 처리
                for file_path in json_files:
                    try:
                        print_status(f"JSON 파일 로드 중: {os.path.basename(file_path)}")
                        
                        # 직접 JSON 파일 로드
                        with open(file_path, 'r', encoding='utf-8') as f:
                            try:
                                # JSON 파싱
                                json_data = json.load(f)
                                
                                # JSON 내용을 텍스트로 변환
                                content = extract_json_content(json_data)
                                
                                # Document 객체 생성
                                doc = Document(
                                    page_content=content,
                                    metadata={"source": file_path}
                                )
                                
                                # 텍스트 분할
                                splits = text_splitter.split_documents([doc])
                                print_status(f"  - 청크 생성: {len(splits)}개")
                                all_documents.extend(splits)
                            except json.JSONDecodeError as e:
                                print_status(f"JSON 파싱 오류: {str(e)}", True)
                                # 대안: 파일 내용 전체를 하나의 문자열로 로드
                                file_content = f.read()
                                doc = Document(
                                    page_content=file_content,
                                    metadata={"source": file_path}
                                )
                                splits = text_splitter.split_documents([doc])
                                print_status(f"  - 대체 방식으로 청크 생성: {len(splits)}개")
                                all_documents.extend(splits)
                    except Exception as e:
                        print_status(f"파일 '{os.path.basename(file_path)}' 처리 중 오류: {str(e)}", True)
                        continue
        else:
            print_status(f"'{data_dir}' 디렉토리가 존재하지 않습니다.")
        
        # 2. CSV 파일 로드 - Datafile 디렉토리
        data_dir = "Datafile"
        if os.path.exists(data_dir):
            csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
            
            if csv_files:
                print_status(f"발견된 CSV 파일: {len(csv_files)}개")
                for file in csv_files:
                    print_status(f"  - {os.path.basename(file)}")
                
                # 모든 CSV 파일 로드 및 처리
                for file_path in csv_files:
                    try:
                        print_status(f"CSV 파일 로드 중: {os.path.basename(file_path)}")
                        
                        # CSV 내용을 텍스트로 변환
                        content = extract_csv_content(file_path)
                        
                        # Document 객체 생성
                        doc = Document(
                            page_content=content,
                            metadata={"source": file_path}
                        )
                        
                        # 텍스트 분할
                        splits = text_splitter.split_documents([doc])
                        print_status(f"  - 청크 생성: {len(splits)}개")
                        all_documents.extend(splits)
                    except Exception as e:
                        print_status(f"파일 '{os.path.basename(file_path)}' 처리 중 오류: {str(e)}", True)
                        continue
            else:
                print_status(f"'{data_dir}' 디렉토리에 CSV 파일이 없습니다.")
        else:
            print_status(f"'{data_dir}' 디렉토리가 존재하지 않습니다.")
        
        if not all_documents:
            print_status("로드된 데이터가 없습니다.", True)
            return None
        
        print_status(f"총 데이터 청크: {len(all_documents)}개")
        return all_documents
    except Exception as e:
        print_status(f"데이터 로드 실패: {str(e)}", True)
        traceback.print_exc()
        return None

def init_retriever():
    """벡터 데이터베이스 및 검색기 초기화"""
    try:
        # 인덱스 저장 경로
        index_path = "faiss_index_wow_data"
        index_exists = os.path.exists(index_path) and os.listdir(index_path)
        
        # 임베딩 모델 로드 - 더 일반적인 다국어 모델로 변경
        print_status("임베딩 모델 로드 중...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        if index_exists:
            # 기존 인덱스 로드
            print_status(f"기존 인덱스 로드 중: {index_path}")
            try:
                vectorstore = FAISS.load_local(
                    index_path, 
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                print_status("인덱스 로드 완료")
            except Exception as e:
                print_status(f"기존 인덱스 로드 실패: {str(e)}", True)
                print_status("새 인덱스 생성으로 전환 중...")
                index_exists = False
        
        if not index_exists:
            # 새 인덱스 생성
            print_status("새 인덱스 생성 중...")
            splits = load_data()
            if not splits:
                raise ValueError("데이터 로드 실패")
                
            vectorstore = FAISS.from_documents(splits, embeddings)
            
            # 인덱스 저장
            os.makedirs(index_path, exist_ok=True)
            vectorstore.save_local(index_path)
            print_status(f"인덱스 저장 완료: {index_path}")
        
        # 검색기 생성
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        return retriever
    except Exception as e:
        print_status(f"검색기 초기화 실패: {str(e)}", True)
        traceback.print_exc()
        return None

def init_llm():
    """LLM 모델 초기화"""
    try:
        print_status("LLM 모델 초기화 중...")
        
        # LangChain 통합 - 안전 설정 제거
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.3
            # 안전 설정 제거 - 오류의 원인
        )
        
        return llm
    except Exception as e:
        print_status(f"LLM 모델 초기화 실패: {str(e)}", True)
        return None

def build_qa_chain(llm, retriever):
    """초간단 QA 체인 구축"""
    try:
        print_status("매우 간단한 QA 체인 구축 중...")
        
        # 템플릿 수정: 단어 수 제한 완화
        template = """시스템: 당신은 월드 오브 워크래프트의 전사 클래스 및 던전, 몬스터, 아이템, 퀘스트 등 전반적인 가이드에 대한 전문가입니다. 
        와우 게임에 대한 전문화에 대한 지식을 갖추고 있습니다.
        주어진 파일 내용을 최대한 참조해서 답변하세요.
        항상 다음 형식으로만 답변합니다:

<질문 내용>
* 1.
* 2.
* 3.

여기서 <질문 내용>은 사용자가 입력한 질문 내용을 10자 이내로 담고, 각 항목은 최대 30단어까지 사용할 수 있습니다. 충분히 정보를 담되 간결함을 유지하세요. 만약, 정보를 한 문장으로 답변할 수 있다면 2, 3번 항목은 제외해도 됩니다. 3개 항목 이하로 답변해주세요.

컨텍스트:
{context}

질문: {question}

답변:"""
        
        # 프롬프트 생성
        prompt = ChatPromptTemplate.from_template(template)
        
        # 문서 조회 및 결합 함수
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # 강제 포맷팅 함수 - 응답 형식화 수정
        def extreme_format(response_text):
            """응답을 적절한 길이로 포맷팅"""
            # 기본 응답 형식 준비
            formatted = "요약 항목 없음\n* 정보 없음\n* 정보 없음\n* 정보 없음"
            
            lines = response_text.strip().split('\n')
            summary = ""
            points = []
            
            # 요약 추출 시도
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                    
                # 첫 줄을 요약으로 간주
                if i == 0 and not line.startswith('*'):
                    # 최대 15단어로 제한
                    words = line.split()
                    if len(words) > 15:
                        words = words[:15]
                    summary = ' '.join(words)
                
                # * 또는 숫자로 시작하는 줄을 항목으로 간주
                elif line.startswith('*') or (line[0].isdigit() and line[1:2] in ['.', ')']):
                    # * 기호 및 숫자 제거
                    if line.startswith('*'):
                        content = line[1:].strip()
                    else:
                        content = line[2:].strip()
                    
                    # 최대 20단어로 제한
                    words = content.split()
                    if len(words) > 20:
                        words = words[:20]
                    points.append(' '.join(words))
                
                # 3개 항목만 사용
                if len(points) >= 3:
                    break
            
            # 결과 구성
            if summary:
                formatted = summary + "\n"
                for i, point in enumerate(points[:3]):
                    formatted += f"* {point}\n"
                    
                # 부족한 항목 채우기
                while len(points) < 3:
                    formatted += "* 정보 없음\n"
                    points.append("정보 없음")
            
            return formatted.strip()
        
        # 체인 구성
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
            | extreme_format
        )
        
        print_status("QA 체인 구축 완료")
        return rag_chain
    except Exception as e:
        print_status(f"QA 체인 구축 실패: {str(e)}", True)
        traceback.print_exc()
        return None

def main():
    """메인 함수"""
    # 명령줄 인수 처리
    import argparse
    
    parser = argparse.ArgumentParser(description="월드 오브 워크래프트 전사 RAG 도우미")
    parser.add_argument("--query", "-q", help="질문을 직접 지정합니다")
    parser.add_argument("--rebuild", "-r", action="store_true", help="FAISS 인덱스를 새로 생성합니다")
    args = parser.parse_args()
    
    print("="*50)
    print("월드 오브 워크래프트 RAG 도우미 시작")
    print("="*50)
    
    # 인덱스 강제 재생성
    if args.rebuild:
        index_path = "faiss_index_wow_data"
        if os.path.exists(index_path):
            import shutil
            print_status(f"기존 인덱스 삭제 중: {index_path}")
            try:
                shutil.rmtree(index_path)
                print_status("인덱스 삭제 완료")
            except Exception as e:
                print_status(f"인덱스 삭제 실패: {str(e)}", True)
    
    # 검색기 초기화
    retriever = init_retriever()
    if not retriever:
        print_status("검색기 초기화 실패. 프로그램을 종료합니다.", True)
        return
    
    # LLM 초기화
    llm = init_llm()
    if not llm:
        print_status("LLM 모델 초기화 실패. 프로그램을 종료합니다.", True)
        return
    
    # QA 체인 구축
    qa_chain = build_qa_chain(llm, retriever)
    if not qa_chain:
        print_status("QA 체인 구축 실패. 프로그램을 종료합니다.", True)
        return
        
    print_status("월드 오브 워크래프트 RAG 도우미가 준비되었습니다!")
    
    # 단일 쿼리 모드
    if args.query:
        query = args.query
        print(f"\n질문: {query}")
        
        try:
            # 응답 생성
            answer = qa_chain.invoke(query)
            
            # 출력
            print(f"\n[답변]\n{answer}")
            
        except Exception as e:
            print_status(f"처리 중 문제 발생: {str(e)}", True)
            traceback.print_exc()
        
        return
    
    # stdin에서 읽기 (파이프 입력 지원)
    if not sys.stdin.isatty():
        try:
            query = sys.stdin.read().strip()
            if query:
                print(f"\n질문: {query}")
                
                # 응답 생성
                answer = qa_chain.invoke(query)
                
                # 출력
                print(f"\n[답변]\n{answer}")
        except Exception as e:
            print_status(f"파이프 입력 처리 중 문제 발생: {str(e)}", True)
            traceback.print_exc()
        
        return
    
    # 대화형 모드 (기본 모드)
    while True:
        try:
            query = input("\n질문: ")
            
            if query.lower() in ["exit", "quit", "종료"]:
                print("프로그램을 종료합니다.")
                break
            
            # 응답 생성
            answer = qa_chain.invoke(query)
            
            # 출력
            print(f"\n[답변]\n{answer}")
            
        except KeyboardInterrupt:
            print("\n프로그램이 사용자에 의해 중단되었습니다.")
            break
        except Exception as e:
            print_status(f"처리 중 문제 발생: {str(e)}", True)
            traceback.print_exc()
    
    print("="*50)
    print("월드 오브 워크래프트 RAG 도우미 종료")
    print("="*50)

if __name__ == "__main__":
    main()
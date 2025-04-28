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
from langchain_core.documents import Document

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

def load_data():
    """모든 JSON 데이터 로드"""
    try:
        # Datajsonfile 디렉토리의 모든 JSON 파일 로드
        data_dir = "Datajsonfile"
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        
        if not json_files:
            print_status(f"'{data_dir}' 디렉토리에 JSON 파일이 없습니다.", True)
            print_status("현재 디렉토리: " + os.getcwd())
            print_status("확인된 파일 목록:")
            
            if os.path.exists(data_dir):
                for file in os.listdir(data_dir):
                    print(f"  - {file}")
            else:
                print_status(f"'{data_dir}' 디렉토리가 존재하지 않습니다.", True)
                # 현재 디렉토리에서 json 파일 확인
                for file in os.listdir():
                    if file.endswith(".json"):
                        print(f"  - {file}")
            return None
        
        print_status(f"발견된 데이터 파일: {len(json_files)}개")
        for file in json_files:
            print_status(f"  - {os.path.basename(file)}")
        
        all_documents = []
        
        # 텍스트 분할 설정
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # 모든 파일 로드 및 처리
        for file_path in json_files:
            try:
                print_status(f"파일 로드 중: {os.path.basename(file_path)}")
                
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
        index_path = "faiss_index_wow_json"
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
        index_path = "faiss_index_wow_json"
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

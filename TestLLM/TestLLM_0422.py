import os
import sys
import warnings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
import traceback
import glob
from langchain_core.documents import Document
from langchain_community.llms.ollama import Ollama

# 모든 경고 메시지를 완전히 억제
warnings.filterwarnings("ignore")

# 간단한 구현을 위한 임포트
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 상태 업데이트를 위한 깔끔한 출력
def print_status(msg, is_error=False):
    """상태 메시지 출력"""
    prefix = "[오류]" if is_error else "[정보]"
    print(f"{prefix} {msg}")

def load_data():
    """모든 텍스트 데이터 로드"""
    try:
        # Datatxtfile 디렉토리의 모든 텍스트 파일 로드
        data_dir = "Datatxtfile"
        txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
        
        if not txt_files:
            print_status(f"'{data_dir}' 디렉토리에 텍스트 파일이 없습니다.", True)
            print_status("현재 디렉토리: " + os.getcwd())
            print_status("확인된 파일 목록:")
            
            if os.path.exists(data_dir):
                for file in os.listdir(data_dir):
                    print(f"  - {file}")
            else:
                print_status(f"'{data_dir}' 디렉토리가 존재하지 않습니다.", True)
                # 현재 디렉토리에서 텍스트 파일 확인
                for file in os.listdir():
                    if file.endswith(".txt"):
                        print(f"  - {file}")
            return None
        
        print_status(f"발견된 데이터 파일: {len(txt_files)}개")
        for file in txt_files:
            print_status(f"  - {os.path.basename(file)}")
        
        all_documents = []
        
        # 텍스트 분할 설정 - 더 작고 겹치는 청크로 변경하여 세밀한 정보도 검색 가능하게 함
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,  # 더 작은 청크 크기로 정밀한 검색
            chunk_overlap=50,  # 적절한 겹침
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # 모든 파일 로드 및 처리
        for file_path in txt_files:
            try:
                filename = os.path.basename(file_path)
                print_status(f"파일 로드 중: {filename}")
                
                # fury_warrior 파일에 특별히 주의
                is_fury_warrior = "fury_warrior" in filename.lower()
                if is_fury_warrior:
                    print_status(f"  - 분노 전사 관련 파일 발견: {filename}")
                
                # 텍스트 파일 로드
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    
                    # 중요 파일은 더 작은 청크로 분할
                    if is_fury_warrior:
                        special_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=100,
                            chunk_overlap=50,
                            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                        )
                        
                        doc = Document(
                            page_content=file_content,
                            metadata={"source": file_path, "filename": filename, "is_important": True}
                        )
                        
                        splits = special_splitter.split_documents([doc])
                    else:
                        doc = Document(
                            page_content=file_content,
                            metadata={"source": file_path, "filename": filename, "is_important": False}
                        )
                        
                        splits = text_splitter.split_documents([doc])
                    
                    print_status(f"  - 청크 생성: {len(splits)}개")
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
        index_path = "faiss_index_wow_txt"
        index_exists = os.path.exists(index_path) and os.listdir(index_path)
        
        # 임베딩 모델 로드 - 다국어 지원 모델 사용
        print_status("임베딩 모델 로드 중...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # 다국어 지원 모델
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
        
        # 사용자 정의 검색 함수 생성
        def custom_retriever(query):
            # 기본 검색 실행 - 더 많은 문서를 가져옴
            docs_with_score = vectorstore.similarity_search_with_score(
                query, 
                k=10  # 충분히 많은 문서를 가져옴
            )
            
            # 검색 결과 확인 및 로깅
            print_status(f"검색 결과: {len(docs_with_score)}개 문서 찾음")
            
            # 중요 문서 (분노 전사 관련) 필터링
            fury_docs = [
                (doc, score) for doc, score in docs_with_score 
                if "fury_warrior" in doc.metadata.get("filename", "").lower()
            ]
            
            # 파악을 위해 문서 로깅
            for i, (doc, score) in enumerate(docs_with_score[:5]):
                filename = doc.metadata.get("filename", "알 수 없음")
                print_status(f"  - 검색 결과 {i+1}: {filename} (점수: {score:.4f})")
                # 분노 전사 관련 파일이면 내용 일부 출력
                if "fury_warrior" in filename.lower():
                    content_preview = doc.page_content[:100].replace("\n", " ")
                    print_status(f"    내용: {content_preview}...")
            
            # 중요 문서가 있으면 우선 반환
            if fury_docs:
                print_status(f"분노 전사 관련 문서 {len(fury_docs)}개 발견됨")
                # 스코어 기준 정렬 (낮을수록 더 관련성 높음)
                fury_docs.sort(key=lambda x: x[1])
                # 문서만 추출
                results = [doc for doc, _ in fury_docs]
                return results
            
            # 중요 문서가 없으면 일반 검색 결과 반환
            print_status("분노 전사 관련 문서가 없어 일반 검색 결과 반환")
            return [doc for doc, _ in docs_with_score[:7]]  # 최대 7개 문서 반환
        
        # 사용자 정의 검색기 반환
        return custom_retriever
    except Exception as e:
        print_status(f"검색기 초기화 실패: {str(e)}", True)
        traceback.print_exc()
        return None

def init_llm():
    """LLM 모델 초기화"""
    try:
        print_status("로컬 Ollama LLM 모델 초기화 중...")
        
        # Ollama로 Gemma3:4b 모델 사용 - 성능 최적화 매개변수 설정
        llm = Ollama(
            model="gemma3:4b",
            temperature=0.2,  # 낮은 온도로 더 정확한 응답 생성
            num_ctx=4096,     # 최대 컨텍스트 길이 증가
            top_p=0.9,        # 높은 확률 토큰만 샘플링
            repeat_penalty=1.1  # 반복 감소 페널티
        )
        
        print_status("Ollama 기반 Gemma3:4b 모델 초기화 완료")
        return llm
    except Exception as e:
        print_status(f"LLM 모델 초기화 실패: {str(e)}", True)
        traceback.print_exc()
        return None

def build_qa_chain(llm, retriever):
    """초간단 QA 체인 구축"""
    try:
        print_status("QA 체인 구축 중...")
        
        # 템플릿 수정: 문서 출처 정보 포함 및 정확한 인용 강조
        template = """시스템: 당신은 월드 오브 워크래프트(WoW) 전문가입니다. 전사 클래스, 던전, 몬스터, 아이템, 퀘스트 등 게임 전반에 대한 가이드 역할을 수행합니다.

**주요 임무:** 주어진 '컨텍스트' 문서를 기반으로 사용자의 질문에 정확하게 답변하십시오. 컨텍스트에 명시된 정보가 항상 우선합니다.

**답변 규칙:**
1. **컨텍스트 기반:** 답변은 반드시 제공된 '컨텍스트' 섹션의 정보만을 근거로 해야 합니다.
2. **정확한 인용:** 컨텍스트에 나타난 순서, 목록, 우선순위는 정확히 그대로 유지하세요. 순서를 변경하거나 정보를 추가하지 마세요.
3. **명확한 형식:** 핵심 답변을 먼저 제시하고, 필요시 글머리 기호(*)나 번호를 사용하여 세부 정보를 추가하세요.
4. **정보 부재 시:** 컨텍스트에서 답변을 찾을 수 없다면, "제공된 문서에는 해당 질문에 대한 정보가 없습니다."라고 명확히 응답하세요.

컨텍스트:
{context}

질문: {question}

답변:"""
        
        # 프롬프트 생성
        prompt = ChatPromptTemplate.from_template(template)
        
        # 문서 조회 및 결합 함수
        def format_docs(docs):
            formatted_docs = []
            
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "출처 없음")
                filename = doc.metadata.get("filename", os.path.basename(source))
                
                # 문서 내용과 출처 정보를 함께 표시
                formatted_doc = f"[문서 {i+1}] 출처: {filename}\n{doc.page_content}"
                formatted_docs.append(formatted_doc)
                
            return "\n\n" + "\n\n---\n\n".join(formatted_docs)
        
        # 체인 구성
        def create_context(question):
            docs = retriever(question)
            return format_docs(docs)
        
        rag_chain = (
            {"context": create_context, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
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
    
    parser = argparse.ArgumentParser(description="월드 오브 워크래프트 전사 RAG 도우미 (로컬 모델 버전)")
    parser.add_argument("--query", "-q", help="질문을 직접 지정합니다")
    parser.add_argument("--rebuild", "-r", action="store_true", help="FAISS 인덱스를 새로 생성합니다")
    args = parser.parse_args()
    
    print("="*50)
    print("월드 오브 워크래프트 RAG 도우미 시작 (로컬 Ollama 버전 - 텍스트 파일 기반)")
    print("="*50)
    
    # Ollama 서버 상태 확인
    try:
        print_status("Ollama 서버 확인 중...")
        # 간단한 요청으로 서버 상태 확인
        test_llm = Ollama(model="gemma3:4b")
        test_llm.invoke("Hello")
        print_status("Ollama 서버 연결 성공")
    except Exception as e:
        print_status(f"Ollama 서버 연결 실패: {str(e)}", True)
        print_status("Ollama 서버가 실행 중인지 확인하세요. (명령어: ollama serve)", True)
        print_status("또는 Gemma3:4b 모델이 설치되어 있는지 확인하세요. (명령어: ollama pull gemma3:4b)", True)
        return
    
    # 인덱스 강제 재생성
    if args.rebuild:
        index_path = "faiss_index_wow_txt"
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

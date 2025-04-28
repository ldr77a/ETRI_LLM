import os
import google.generativeai as genai
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import torch
import re

# Gemini API 설정
API_KEY = "AIzaSyAMIlzkwZyyiUoDbQg_bL8j_pFFVasB3II"
os.environ["GOOGLE_API_KEY"] = API_KEY
genai.configure(api_key=API_KEY)

# Qdrant 클라이언트 설정
client = QdrantClient(
    url="https://048a9c04-5c3d-49fc-9e64-4c555f33c4b5.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.cyryM5Bho5E4Nb7P9pPl-YsjZTImb2ijl-KpaWCEwNs",
)

COLLECTION_NAME = "wow_guides_bge_large_chunked"

# 임베딩 모델, 수정해보기  BAAI/bge-large-en-v1.5
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
#print(f"임베딩 모델 로딩: {EMBEDDING_MODEL_NAME}...")
try:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    #print("모델 로딩 성공")
    vector_size = model.get_sentence_embedding_dimension()
    #print(f"벡터 차원: {vector_size}")
except Exception as e:
    #print(f"임베딩 모델 로딩 오류: {e}")
    # 백업 모델 사용
    #print("백업 모델 사용: paraphrase-multilingual-mpnet-base-v2")
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def is_korean(text):
    """텍스트에 한글이 포함되어 있는지 확인"""
    return bool(re.search('[가-힣]', text))

def translate_to_english(text):
    """한국어 텍스트를 영어로 번역"""
    # Gemini 모델을 사용하여 번역
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"Translate the following Korean text to English, keep it brief and accurate: {text}"
    try:
        response = model.generate_content(prompt)
        translated = response.text.strip().strip('"')
        #print(f"번역 결과: '{text}' -> '{translated}'")
        return translated
    except Exception as e:
        #print(f"번역 오류: {e}")
        return text

def generate_embedding(text):
    """텍스트를 임베딩 벡터로 변환"""
    # 임베딩 생성
    embedding = model.encode(text).tolist()
    return embedding

def is_table_query(query):
    """쿼리가 테이블이나 표 관련 정보를 찾는지 확인"""
    table_keywords = [
        "표", "테이블", "table", "차트", "chart", "비교", "목록", "리스트", "list", 
        "스탯", "stat", "능력치", "우선순위", "priority", "순위", "ranking",
        "속성", "특성", "스펙", "spec"
    ]
    
    return any(keyword in query.lower() for keyword in table_keywords)

def search_similar_docs(query, limit=5, filter_type=None, prefer_tables=None):
    """사용자 쿼리와 유사한 문서 검색"""
    # 테이블 선호도 감지 (None이면 자동 감지)
    if prefer_tables is None:
        prefer_tables = is_table_query(query)
        
    # 한국어 쿼리 확인 및 번역
    if is_korean(query):
        english_query = translate_to_english(query)
    else:
        english_query = query
    
    # 쿼리 임베딩 생성
    query_vector = generate_embedding(english_query)
    
    # 필터 설정 (특정 타입만 검색하려는 경우)
    query_filter = None
    if filter_type:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(key="type", match=models.MatchValue(value=filter_type))
            ]
        )
    
    # 테이블 관련 검색어 강화 (테이블 관련 단어 추가)
    if prefer_tables:
        table_limit = limit * 2

        table_filter = None
        if filter_type:
            table_filter = models.Filter(
                must=[
                    models.FieldCondition(key="type", match=models.MatchValue(value=filter_type)),
                    models.FieldCondition(key="has_table", match=models.MatchValue(value=True))
                ]
            )
        else:
            table_filter = models.Filter(
                must=[
                    models.FieldCondition(key="has_table", match=models.MatchValue(value=True))
                ]
            )
        
        try:
            # 테이블이 있는 문서 먼저 검색 시도
            table_results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=table_filter,
                limit=limit
            )
            
            if table_results:
                # 검색 결과에서 문서 내용과 메타데이터 추출
                documents = []
                for result in table_results:
                    payload = result.payload
                    documents.append({
                        'content': payload.get('chunk_text', ''),
                        'title': payload.get('title', ''),
                        'source': payload.get('source_file', ''),
                        'score': result.score,
                        'has_table': payload.get('has_table', False)
                    })
                
                # 테이블 결과만 있을 경우 바로 반환
                if len(documents) >= limit:
                    return documents[:limit]
                
                # 테이블 결과가 limit보다 적을 경우, 일반 검색으로 보충
                remaining_limit = limit - len(documents)
                
                search_result = client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=limit * 2  # 더 많이 가져와서 필터링
                )
                
                # 중복 제거를 위한 이미 추가된 문서 ID 추적
                existing_ids = set(doc['source'] + str(i) for i, doc in enumerate(documents))
                
                for result in search_result:
                    if len(documents) >= limit:
                        break
                        
                    payload = result.payload
                    doc_id = payload.get('source_file', '') + str(len(documents))
                    
                    # 이미 추가된 문서가 아닌 경우에만 추가
                    if doc_id not in existing_ids:
                        documents.append({
                            'content': payload.get('chunk_text', ''),
                            'title': payload.get('title', ''),
                            'source': payload.get('source_file', ''),
                            'score': result.score,
                            'has_table': payload.get('has_table', False)
                        })
                        existing_ids.add(doc_id)
                
                return documents
        except Exception as e:
            #print(f"테이블 우선 검색 중 오류: {e}")
            # 오류 발생 시 일반 검색으로 대체
            pass
    
    # 일반 검색 (테이블 선호가 없거나 테이블 검색 실패 시)
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=limit
    )
    
    # 검색 결과에서 문서 내용과 메타데이터 추출
    documents = []
    for result in search_result:
        payload = result.payload
        documents.append({
            'content': payload.get('chunk_text', ''),
            'title': payload.get('title', ''),
            'source': payload.get('source_file', ''),
            'score': result.score,
            'has_table': payload.get('has_table', False)
        })
    
    # 디버깅용 검색 결과 출력 (주석 처리)
    '''
    print(f"\n--- 쿼리: '{query}' 검색 결과 ---")
    if is_korean(query):
        print(f"(영어 번역: '{english_query}')")
    
    for i, doc in enumerate(documents):
        print(f"결과 {i+1}:")
        print(f"  제목: {doc['title']}")
        print(f"  소스: {doc['source']}")
        print(f"  점수: {doc['score']:.4f}")
        print(f"  내용 미리보기: {doc['content'][:100]}...")
    '''
    
    return documents

def generate_answer(query, context, query_in_korean=True):
    """컨텍스트를 바탕으로 LLM에게 질문에 대한 답변 요청 (개선된 프롬프트)"""

    persona = """
    You are a powerful and wise NPC with deep knowledge of the World of Warcraft universe of Azeroth.
    You must speak like a Grand Marshal of Stormwind or a Sage of Ironforge.
    Use a grand and epic tone, and naturally incorporate WoW terminology (e.g., player names, monster names, zones, abilities, item names).
    """

    # Define task instructions
    instructions = f"""
    '컨텍스트' 정보만 사용해 '질문'에 답하세요.
    컨텍스트에 없는 정보는 절대 추가하지 마세요.
    여러 출처가 있다면 내용을 종합하여 일관된 답변을 작성하세요.
    질문의 핵심에 직접 관련된 답변만 제공하세요.
    월드 오브 워크래프트 세계관의 스타일과 분위기를 유지하세요.
    중요한 WoW 용어는 한글 먼저, 영문 괄호 표기(예: 스톰윈드(Stormwind)).
    '출처', '파일' 등 메타데이터 언급 없이 내용만 전달하세요.
    처음부터 끝까지 WoW 세계관 스타일을 유지하세요.
    """
    
    model = genai.GenerativeModel('gemini-2.0-flash')

    prompt = f"""
[페르소나]
{persona}

[컨텍스트]
{context}

[지시사항]
{instructions}

[질문]
{query}

[답변]
"""

    response = model.generate_content(prompt)
    return response.text.replace("(출처 1)", "").replace("(출처 2)", "").replace("(출처 3)", "").replace("(출처", "").replace("출처", "")

def wow_assistant(query):
    """WoW 어시스턴트 메인 함수"""
    # 테이블 관련 쿼리인지 확인
    is_table_related = is_table_query(query)
    
    # 1. 쿼리와 유사한 문서 검색 (테이블 우선 옵션 적용)
    similar_docs = search_similar_docs(query, prefer_tables=is_table_related)
    
    # 검색 결과가 없거나 품질이 좋지 않은 경우 필터를 변경하여 다시 시도
    if not similar_docs or all(doc['score'] < 0.6 for doc in similar_docs):
        print("품질 높은 검색 결과가 없어 추가 검색을 시도합니다...")
        # 여러 filter_type으로 다시 시도
        for filter_type in ["class_guide", "dungeon_overview", "dungeon_guide_raw", "dungeon_boss"]:
            additional_docs = search_similar_docs(query, filter_type=filter_type, limit=2, prefer_tables=is_table_related)
            if additional_docs:
                for doc in additional_docs:
                    if doc not in similar_docs:
                        similar_docs.append(doc)
    
    # 검색 결과가 여전히 없는 경우
    if not similar_docs:
        return "죄송합니다. 질문에 관련된 정보를 찾을 수 없습니다. 다른 질문을 해주세요."
 
    # 2. 검색된 문서의 내용과 메타데이터를 포함한 컨텍스트 생성
    context_parts = []
    for i, doc in enumerate(similar_docs):
        # 테이블 포함 여부에 따라 다른 포맷 사용
        doc_context = f"출처 {i+1}: {doc['title']} (파일: {doc['source']})\n\n{doc['content']}"
        # 테이블 있는 문서는 우선순위 높게 앞에 배치
        if doc.get('has_table', False):
            context_parts.insert(0, doc_context)
        else:
            context_parts.append(doc_context)
    
    context = "\n\n---\n\n".join(context_parts)
 
    answer = generate_answer(query, context)
    
    return answer

if __name__ == "__main__":
    print("아제로스의 용사여, 강력한 힘을 갈망하는 그대의 의문점을 해결해주겠소.")
    print("(종료하려면 'exit' 또는 '종료'를 입력해야한다.)")
    
    while True:
        user_query = input("\n질문: ")
        if user_query.lower() in ['exit', '종료']:
            print("어시스턴트를 종료합니다.")
            break
        
        try:
            response = wow_assistant(user_query)
            print(f"\n{response}")
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")

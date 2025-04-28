import os
import json
import argparse
import google.generativeai as genai
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
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

# 임베딩 모델 설정
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
try:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vector_size = model.get_sentence_embedding_dimension()
except Exception as e:
    # 백업 모델 사용
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
        return translated
    except Exception as e:
        return text

def generate_embedding(text):
    """텍스트를 임베딩 벡터로 변환"""
    embedding = model.encode(text).tolist()
    return embedding

def search_similar_docs(query, limit=10, filter_type=None, prefer_technical=True):
    """사용자 쿼리와 유사한 문서 검색 (기술 문서 중심)"""
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
    
    # 기술 문서 우선 검색 설정 
    if prefer_technical:
        try:
            tech_filters = [
                "class_guide", "dungeon_guide_raw", "dungeon_boss", 
                "dungeon_overview", "raid_guide"
            ]
            
            all_documents = []
            
            # 각 기술 문서 유형별로 검색
            for tech_filter in tech_filters:
                tech_query_filter = models.Filter(
                    must=[
                        models.FieldCondition(key="type", match=models.MatchValue(value=tech_filter))
                    ]
                )
                
                tech_results = client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,
                    query_filter=tech_query_filter,
                    limit=3  # 각 유형별로 3개씩만 검색
                )
                
                # 검색 결과 추가
                for result in tech_results:
                    payload = result.payload
                    all_documents.append({
                        'content': payload.get('chunk_text', ''),
                        'title': payload.get('title', ''),
                        'source': payload.get('source_file', ''),
                        'type': payload.get('type', ''),
                        'score': result.score,
                        'has_table': payload.get('has_table', False)
                    })
            
            # 점수에 따라 정렬 후 상위 결과만 반환
            sorted_documents = sorted(all_documents, key=lambda x: x['score'], reverse=True)
            return sorted_documents[:limit]
            
        except Exception as e:
            # 오류 발생 시 일반 검색으로 대체
            pass
    
    # 일반 검색 (기술 문서 우선 검색 실패 시)
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
            'type': payload.get('type', ''),
            'score': result.score,
            'has_table': payload.get('has_table', False)
        })
    
    return documents

def read_bug_report(file_path):
    """버그 리포트 파일을 읽어 내용을 반환"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"버그 리포트 파일을 읽는 중 오류 발생: {e}")
        return None

def generate_qa_checklist(bug_report, reference_docs, additional_instruction=""):
    """컨텍스트와 버그 리포트를 바탕으로 QA 체크리스트 생성"""
    
    # 컨텍스트 구성
    context_parts = []
    for i, doc in enumerate(reference_docs):
        doc_context = f"참조 자료 {i+1}: {doc['title']} (유형: {doc['type']})\n{doc['content']}"
        context_parts.append(doc_context)
    
    context = "\n\n---\n\n".join(context_parts)
    
    # QA 체크리스트 생성 프롬프트
    prompt = f"""
[작업 설명]
당신은 World of Warcraft 게임의 QA 전문가입니다. 버그 리포트와 참조 자료를 토대로 게임 개발팀이 확인해야 할 구체적인 QA 체크리스트를 생성해주세요.

[버그 리포트]
{bug_report}

[참조 자료]
{context}

[지시사항]
1. 위 버그 리포트와 참조 자료를 분석하여 게임 QA팀이 검증해야 할 항목들을 구체적으로 나열해주세요.
2. 각 체크 항목은 명확하고 구체적이어야 합니다. 
3. 각 항목은 QA 담당자가 명확하게 "예/아니오"로 답할 수 있는 형태로 작성해주세요.
4. 항목에는 구체적인 게임 요소(보스 이름, 아이템 이름, 능력 이름, 지역 이름 등)를 포함해야 합니다.
5. 최소 15개 이상의 구체적인 체크 항목을 작성해주세요.
6. 각 체크 항목에는 가능한 한 정확한 수치 정보를 포함하세요. 예: "피해량 294757", "지속시간 10초", "쿨다운 40초" 등.
7. 체크 항목은 번호를 매겨 순서대로 작성해주세요. 예: "1.", "2.", "3.", ...
8. 각 체크 항목에는 항목 내용에 대한 요약을 볼드체(**텍스트**)로 작성하고 콜론(:)으로 분리하세요.
9. 체크리스트 형식은 아래 예시와 동일하게 작성해주세요.
10. 카테고리 헤더를 사용하지 말고, 모든 항목을 연속적인 숫자 리스트로 작성하세요.
{additional_instruction}

[형식 예시]
다음은 버그 리포트와 참조 자료를 기반으로 생성된 World of Warcraft 게임 QA 체크리스트입니다.

1.  **키카탈 전투: 검은 피 밟기 시 이동 불가 상태 적용 여부 (탱커)**: 키카탈 전투에서 탱커 캐릭터로 검은 피(Black Blood)를 밟았을 때, Grasping Blood 효과로 인해 정상적으로 이동 불가 상태가 되는가?
2.  **키카탈 전투: 검은 피 밟기 시 이동 불가 상태 적용 여부 (딜러/힐러)**: 키카탈 전투에서 딜러/힐러 캐릭터로 검은 피(Black Blood)를 밟았을 때, Grasping Blood 효과로 인해 정상적으로 이동 불가 상태가 되는가?
3.  **키카탈 전투: 검은 피 밟기 시 이동 불가 상태 지속 시간**: 키카탈 전투에서 검은 피(Black Blood)를 밟아 이동 불가 상태가 되었을 때, 해당 상태가 1분간 지속되는가?
"""
    
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    return response.text

def save_checklist_to_file(checklist, output_file):
    """체크리스트를 파일로 저장"""
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(checklist)
        print(f"QA 체크리스트가 '{output_file}'에 저장되었습니다.")
        return True
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='버그 리포트 기반 QA 체크리스트 생성 도구')
    parser.add_argument('-b', '--bug_report', type=str, required=True, help='버그 리포트 파일 경로')
    parser.add_argument('-o', '--output', type=str, default='qa_checklist_결과.txt', help='출력 파일 경로')
    
    args = parser.parse_args()
    
    # 버그 리포트 파일 필수
    if not args.bug_report:
        print("버그 리포트 파일(-b)을 제공해주세요.")
        return
    
    # 버그 리포트 처리
    bug_report_content = read_bug_report(args.bug_report)
    if not bug_report_content:
        print("버그 리포트를 읽을 수 없습니다. 다시 확인해주세요.")
        return
    
    # 검색 쿼리 생성 (버그 리포트의 처음 일부를 사용)
    search_query = bug_report_content[:500]
    
    print("관련 자료 검색 중...")
    reference_docs = search_similar_docs(search_query, limit=15)
    
    if not reference_docs:
        print("검색 결과가 없습니다. 다른 키워드가 포함된 버그 리포트를 시도해주세요.")
        return
    
    print(f"검색 완료: {len(reference_docs)}개 문서 발견")
    
    print("QA 체크리스트 생성 중...")
    qa_checklist = generate_qa_checklist(bug_report_content, reference_docs)
    
    if qa_checklist:
        if save_checklist_to_file(qa_checklist, args.output):
            print("생성된 체크리스트:")
            print("-" * 50)
            print(qa_checklist)
            print("-" * 50)
    else:
        print("체크리스트 생성에 실패했습니다.")

if __name__ == "__main__":
    print("월드 오브 워크래프트 QA 체크리스트 생성기를 시작합니다.")
    main() 
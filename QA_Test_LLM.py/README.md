# 버그 리포트 QA 체크리스트 생성기

버그 리포트 파일을 업로드하면 게임 QA 테스트를 위한 체크리스트를 자동으로 생성하는 도구입니다. 이 도구는 버그 리포트를 분석하고 벡터 데이터베이스에서 관련 정보를 검색하여 검증해야 할 항목들을 추출합니다.

## 기능

- 버그 리포트 파일 분석
- 관련 게임 정보 자동 검색 (Qdrant 벡터 데이터베이스 활용)
- 구체적인 QA 체크리스트 자동 생성
- 생성된 체크리스트 파일로 저장 및 다운로드

## 설치 방법

1. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

2. API 키는 코드에 이미 포함되어 있습니다.

## 사용 방법

### 커맨드 라인 인터페이스 (CLI)

```bash
# 버그 리포트 파일을 지정하여 실행
cd QA_Test_LLM.py
python QA_Checklist_Generator.py -b 버그리포트파일.txt -o 체크리스트결과.txt
```

### 웹 인터페이스

```bash
# 웹 인터페이스 실행
cd QA_Test_LLM.py
streamlit run QA_Checklist_Web.py
```

브라우저에서 http://localhost:8501 로 접속하면 웹 인터페이스를 통해 파일을 업로드하고 체크리스트를 생성할 수 있습니다.

## 작동 원리

1. **버그 리포트 업로드**: 사용자가 버그 리포트 파일(.txt, .log, .md 등)을 업로드합니다.

2. **관련 데이터 검색**: 
   - 업로드된 버그 리포트를 분석하여 관련 키워드 추출
   - Qdrant 벡터 데이터베이스에서 관련 문서 검색
   - 한국어 내용은 자동으로 영어로 번역하여 검색 성능 향상

3. **체크리스트 생성**:
   - 검색된 문서와 버그 리포트를 종합적으로 분석
   - 구체적인 QA 체크 항목 자동 생성
   - 각 체크 항목은 "예/아니오"로 답변 가능한 형태로 작성

4. **결과 출력**:
   - 생성된 체크리스트를 파일로 저장
   - 웹 인터페이스에서는 바로 확인 및 다운로드 가능

## 파일 구조

- `QA_Checklist_Generator.py`: 메인 생성 알고리즘
- `QA_Checklist_Web.py`: 웹 인터페이스

## 요구사항

- Python 3.8 이상
- 인터넷 연결 (Qdrant 서버 및 Gemini API 사용)
- 필요 패키지: google-generativeai, qdrant-client, sentence-transformers, streamlit 등 
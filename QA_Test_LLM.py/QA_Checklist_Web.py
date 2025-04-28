import streamlit as st
import os
import tempfile
from QA_Checklist_Generator import search_similar_docs, generate_qa_checklist

st.set_page_config(
    page_title="버그 리포트 QA 체크리스트 생성기",
    page_icon="🎮",
    layout="wide"
)

st.title("버그 리포트 QA 체크리스트 생성기")
st.markdown("_게임 버그 리포트 파일을 업로드하면 QA 테스트 체크리스트가 자동으로 생성됩니다_")

# 사이드바 - 사용 방법
st.sidebar.header("사용 방법")
st.sidebar.markdown("""
1. 버그 리포트 파일(.txt, .log 등)을 업로드하세요
2. '체크리스트 생성' 버튼을 클릭하세요
3. 생성된 체크리스트를 파일로 다운로드할 수 있습니다
4. '추가 체크리스트 생성' 버튼으로 더 많은 항목을 생성할 수 있습니다
""")

# 상태 변수 초기화
if 'checklist' not in st.session_state:
    st.session_state.checklist = ""
if 'all_docs' not in st.session_state:
    st.session_state.all_docs = []
if 'file_content' not in st.session_state:
    st.session_state.file_content = ""
if 'generated_count' not in st.session_state:
    st.session_state.generated_count = 0

# 파일 업로드 섹션
st.header("버그 리포트 파일 업로드")
uploaded_file = st.file_uploader("버그 리포트 파일을 업로드하세요", type=["txt", "log", "md", "doc", "docx"])

if uploaded_file is not None:
    # 파일 내용 미리보기
    file_content = uploaded_file.read().decode("utf-8")
    st.session_state.file_content = file_content  # 세션에 저장
    st.subheader("파일 내용 미리보기")
    st.text_area("", file_content[:500] + "..." if len(file_content) > 500 else file_content, height=150)
    
    # 체크리스트 생성 버튼
    generate_button = st.button("체크리스트 생성")
    
    # 버튼 클릭 시 처리
    if generate_button:
        with st.spinner("체크리스트 생성 중... 잠시만 기다려주세요."):
            # 1. 유사 문서 검색 (Chunking)
            st.info("1/2 단계: 관련 문서 검색 중...")
            chunks = [file_content[i:i+500] for i in range(0, len(file_content), 500)]
            all_docs = []
            for chunk in chunks:
                docs = search_similar_docs(chunk, limit=5)  # 각 청크마다 5개씩
                all_docs.extend(docs)
            
            # 중복 제거 (source 및 title 기준)
            unique_docs = []
            seen = set()
            for doc in all_docs:
                key = doc['source'] + doc['title']
                if key not in seen:
                    seen.add(key)
                    unique_docs.append(doc)
            
            # 유사도 점수로 정렬 후 상위 15개 선택
            sorted_docs = sorted(unique_docs, key=lambda x: x['score'], reverse=True)[:15]
            st.session_state.all_docs = sorted_docs  # 세션에 저장
            
            # 2. QA 체크리스트 생성
            if sorted_docs:
                st.success(f"{len(sorted_docs)}개의 관련 문서를 찾았습니다.")
                st.info("2/2 단계: 체크리스트 생성 중...")
                
                # 프롬프트에 추가 지시사항 (기존 체크리스트와 중복 방지)
                additional_instruction = ""
                if st.session_state.generated_count > 0:
                    additional_instruction = f"""
                    중요: 이전에 이미 다음 체크리스트가 생성되었습니다. 새로운 체크리스트는 이전 항목과 중복되지 않아야 합니다:
                    
                    {st.session_state.checklist}
                    
                    이전과 다른 새로운 체크 항목들을 생성해주세요.
                    """
                
                checklist = generate_qa_checklist(file_content, sorted_docs, additional_instruction)
                
                # 처음 생성 시 또는 추가 생성 시 세션에 추가
                if st.session_state.generated_count == 0:
                    st.session_state.checklist = checklist
                else:
                    st.session_state.checklist += "\n\n--- 추가 체크리스트 항목 ---\n\n" + checklist
                
                st.session_state.generated_count += 1
                
                # 3. 결과 표시 - test_output.txt 형식에 맞게 표시
                st.header("📋 생성된 QA 체크리스트")
                
                # 체크리스트 내용을 그대로 표시
                st.markdown(st.session_state.checklist)
                
                # 4. 다운로드 옵션
                st.download_button(
                    label="체크리스트 다운로드",
                    data=st.session_state.checklist,
                    file_name=f"qa_checklist_결과_{st.session_state.generated_count}.txt",
                    mime="text/plain"
                )
                
                # 5. 추가 생성 버튼 (이미 문서를 찾은 상태일 때만 표시)
                st.button("추가 체크리스트 생성", key="generate_more", on_click=lambda: None)
            else:
                st.error("관련 문서를 찾을 수 없습니다. 다른 버그 리포트 파일을 시도해주세요.")
    
    # 추가 생성 버튼 클릭 처리 (세션에 문서가 있을 때)
    elif 'generate_more' in st.session_state and st.session_state.all_docs:
        with st.spinner("추가 체크리스트 생성 중... 잠시만 기다려주세요."):
            st.info("추가 체크 항목 생성 중...")
            
            # 추가 지시사항 (기존 체크리스트와 중복 방지)
            additional_instruction = f"""
            중요: 이전에 이미 다음 체크리스트가 생성되었습니다. 새로운 체크리스트는 이전 항목과 중복되지 않아야 합니다:
            
            {st.session_state.checklist}
            
            이전과 다른 새로운 체크 항목들을 생성해주세요.
            """
            
            # 추가 체크리스트 생성
            additional_checklist = generate_qa_checklist(st.session_state.file_content, st.session_state.all_docs, additional_instruction)
            
            # 세션에 추가
            st.session_state.checklist += "\n\n--- 추가 체크리스트 항목 ---\n\n" + additional_checklist
            st.session_state.generated_count += 1
            
            # 결과 표시
            st.header("📋 생성된 QA 체크리스트")
            st.markdown(st.session_state.checklist)
            
            # 다운로드 옵션
            st.download_button(
                label="체크리스트 다운로드",
                data=st.session_state.checklist,
                file_name=f"qa_checklist_결과_{st.session_state.generated_count}.txt",
                mime="text/plain"
            )
            
            # 추가 생성 버튼 다시 표시
            st.button("추가 체크리스트 생성", key=f"generate_more_{st.session_state.generated_count}", on_click=lambda: None)
            
else:
    st.info("버그 리포트 파일을 업로드하면 체크리스트 생성이 시작됩니다.")

# 하단 정보
st.markdown("---")
st.markdown("이 도구는 버그 리포트를 분석하여 게임 QA팀이 검증해야 할 항목들을 자동으로 생성합니다.") 
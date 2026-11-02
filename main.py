
import streamlit as st
import pandas as pd
import json
from pathlib import Path


# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

SAVE_DIR = Path("내_기록")
SAVE_FILE = SAVE_DIR / "data_notes.json"


# ==========================================
# 2. 기록 파일 관리
# ==========================================

def load_notes():
    """저장된 우리말 뜻과 데이터 출처 불러오기"""

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "meanings": {},
        "source": ""
    }


def save_notes(meanings, source):
    """입력한 내용을 JSON 파일로 저장"""

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    notes = {
        "meanings": meanings,
        "source": source
    }

    with open(SAVE_FILE, "w", encoding="utf-8") as file:
        json.dump(
            notes,
            file,
            ensure_ascii=False,
            indent=4
        )


# ==========================================
# 3. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )
    return df


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오는 데 실패했습니다.")
    st.error(f"오류 내용: {e}")
    st.stop()


# 저장된 기록 불러오기
notes = load_notes()


# ==========================================
# 4. 화면 제목
# ==========================================

st.title("🧠 뇌졸중 예측 실습실")

st.markdown(
    """
    뇌졸중 관련 데이터를 탐색하고 머신러닝을 활용한
    예측 모델을 학습하기 위한 실습 공간입니다.

    먼저 데이터의 구조와 특징을 확인해 봅시다.
    """
)

st.divider()


# ==========================================
# 5. 데이터 기본 통계
# ==========================================

total_people = len(df)
total_columns = len(df.columns)

stroke_count = int((df["stroke"] == 1).sum())

stroke_ratio = (
    stroke_count / total_people * 100
    if total_people > 0 else 0
)


st.subheader("📊 데이터 기본 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="전체 사람 수",
        value=f"{total_people:,}명"
    )

with col2:
    st.metric(
        label="전체 열 개수",
        value=f"{total_columns}개"
    )

with col3:
    st.metric(
        label="뇌졸중 발생 인원",
        value=f"{stroke_count:,}명"
    )

with col4:
    st.metric(
        label="뇌졸중 발생 비율",
        value=f"{stroke_ratio:.2f}%"
    )


st.divider()


# ==========================================
# 6. 열 정보 및 우리말 뜻 입력
# ==========================================

st.subheader("📋 데이터 열 정보")

st.caption(
    "각 열의 의미를 교재에서 찾아 우리말 뜻 칸에 직접 입력하세요. "
    "입력한 내용은 자동으로 저장됩니다."
)


# 데이터 종류 판별
def get_data_type(series, column_name):

    unique_count = series.nunique(dropna=True)

    if column_name == "id":
        return "식별자형"

    if unique_count <= 2 and set(series.dropna().unique()).issubset({0, 1}):
        return "이진형"

    if pd.api.types.is_numeric_dtype(series):
        return "수치형"

    return "범주형"


# 열 정보 테이블 생성
column_info = []

for column in df.columns:

    column_info.append({
        "열 이름": column,
        "우리말 뜻": notes["meanings"].get(column, ""),
        "값의 종류": get_data_type(df[column], column),
        "빈 값 개수": int(df[column].isna().sum())
    })


info_df = pd.DataFrame(column_info)


# 편집 가능한 테이블
edited_df = st.data_editor(
    info_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    disabled=[
        "열 이름",
        "값의 종류",
        "빈 값 개수"
    ],
    column_config={
        "열 이름": st.column_config.TextColumn(
            "열 이름"
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고하여 직접 입력하세요."
        ),
        "값의 종류": st.column_config.TextColumn(
            "값의 종류"
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            format="%d"
        )
    },
    key="column_info_editor"
)


# 입력한 우리말 뜻 저장
meanings = {}

for _, row in edited_df.iterrows():
    meanings[row["열 이름"]] = row["우리말 뜻"]


# ==========================================
# 7. 원본 데이터 미리보기
# ==========================================

st.divider()

st.subheader("🔎 원본 데이터 미리보기")

st.caption("데이터의 처음 다섯 행을 원본 형태로 표시합니다.")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 8. 데이터 출처 입력
# ==========================================

st.divider()

st.subheader("📚 데이터 출처")

st.caption(
    "교재에 제시된 데이터 출처를 아래에 입력하세요."
)


# 세션 상태 초기화
if "source_text" not in st.session_state:
    st.session_state.source_text = notes.get("source", "")


source_text = st.text_area(
    "데이터 출처",
    key="source_text",
    placeholder="교재에 있는 데이터 출처를 입력하세요.",
    height=120
)


# ==========================================
# 9. 자동 저장
# ==========================================

save_notes(
    meanings=meanings,
    source=source_text
)

st.caption("💾 우리말 뜻과 데이터 출처가 자동으로 저장됩니다.")

st.divider()

st.caption(
    "본 실습실은 머신러닝 학습을 위한 교육용 데이터 탐색 도구입니다. "
    "실제 의료 진단이나 뇌졸중 발생 여부의 판단에 사용해서는 안 됩니다."
)

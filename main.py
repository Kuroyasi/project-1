import json
from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 뇌졸중 예측 실습실")
st.caption("데이터 과학 · 데이터 모델링과 평가")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "데이터_소개.json"


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다.\n\n{e}")
    st.stop()


# --------------------------------------------------
# 기록 불러오기 및 저장하기
# --------------------------------------------------

def load_record():
    if RECORD_FILE.exists():
        try:
            with open(RECORD_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return {}

    return {}


def save_record(meanings, source):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "meanings": meanings,
        "source": source
    }

    with open(RECORD_FILE, "w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=4)


saved_record = load_record()

saved_meanings = saved_record.get("meanings", {})
saved_source = saved_record.get("source", "")


# --------------------------------------------------
# 숫자 카드
# --------------------------------------------------

total_people = len(df)
total_columns = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum())
stroke_rate = stroke_count / total_people * 100 if total_people else 0

st.subheader("데이터 규모")

col1, col2, col3, col4 = st.columns(4)

col1.metric("전체 사람 수", f"{total_people:,}명")
col2.metric("열 개수", f"{total_columns}개")
col3.metric("뇌졸중 경험자", f"{stroke_count:,}명")
col4.metric("뇌졸중 비율", f"{stroke_rate:.2f}%")


# --------------------------------------------------
# 열 설명 표
# --------------------------------------------------

st.divider()
st.subheader("데이터 열 살펴보기")

column_info = []

for column in df.columns:
    column_info.append({
        "열 이름": column,
        "우리말 뜻": saved_meanings.get(column, ""),
        "값의 종류": int(df[column].nunique(dropna=True)),
        "빈 값 개수": int(df[column].isna().sum())
    })

info_df = pd.DataFrame(column_info)

edited_df = st.data_editor(
    info_df,
    use_container_width=True,
    hide_index=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고하여 각 열의 우리말 뜻을 입력하세요."
        )
    }
)

if st.button("우리말 뜻 저장", type="primary"):
    meanings = dict(
        zip(edited_df["열 이름"], edited_df["우리말 뜻"])
    )

    save_record(meanings, saved_source)
    st.success("우리말 뜻이 저장되었습니다.")


# --------------------------------------------------
# 데이터 미리보기
# --------------------------------------------------

st.divider()
st.subheader("데이터 미리보기")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 데이터 출처
# --------------------------------------------------

st.divider()
st.subheader("데이터 출처")

st.markdown(
    """
    데이터 게시자, 라이선스, 원자료 출처 등을 확인하고 기록하세요.

    확인할 수 없는 정보는 임의로 작성하지 말고
    **확인할 수 없음**으로 남겨 두세요.
    """
)

source_text = st.text_area(
    "출처 기록",
    value=saved_source,
    height=150,
    placeholder="게시자, 라이선스, 사사 문구 등을 입력하세요."
)

if st.button("출처 저장"):
    meanings = dict(
        zip(edited_df["열 이름"], edited_df["우리말 뜻"])
    )

    save_record(meanings, source_text)
    st.success("출처가 저장되었습니다.")


# --------------------------------------------------
# 안내
# --------------------------------------------------

st.divider()

st.info(
    """
    이 데이터는 분류 모델을 학습하고 평가하는 교육용 자료입니다.

    원자료의 수집 출처를 확인할 수 없으므로 분석 결과를
    실제 환자나 일반 인구에 적용할 수 없습니다.
    실제 의학적 판단이나 진단에 사용해서는 안 됩니다.
    """
)

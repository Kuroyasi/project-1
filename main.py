
from pathlib import Path
import json

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# CSV 파일 경로
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "stroke.csv"
NOTES_PATH = BASE_DIR / "내_기록" / "data_notes.json"


# 데이터 로딩
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"stroke.csv 파일을 찾을 수 없습니다.\n경로: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH, encoding="utf-8")


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


st.title("🧠 뇌졸중 예측 실습실")

st.write(
    """
    머신러닝을 활용하여 뇌졸중 데이터의 특징을 탐색하고,
    분류 모델을 학습한 뒤 모델의 예측 성능을 평가하는 실습 공간입니다.
    """
)

st.divider()


# 데이터 기본 정보
st.subheader("데이터셋 기본 정보")

total_people = len(df)
total_columns = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum())
stroke_rate = stroke_count / total_people * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("전체 데이터 수", f"{total_people:,}명")
col2.metric("전체 변수 수", f"{total_columns}개")
col3.metric("뇌졸중 데이터 수", f"{stroke_count:,}명")
col4.metric("뇌졸중 비율", f"{stroke_rate:.2f}%")


st.divider()


# 변수 설명
st.subheader("변수 설명")

meanings = {
    "id": "개인 식별 번호",
    "gender": "성별",
    "age": "나이",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부",
    "ever_married": "결혼 여부",
    "work_type": "직업 유형",
    "Residence_type": "거주 유형",
    "avg_glucose_level": "평균 혈당 수치",
    "bmi": "체질량지수",
    "smoking_status": "흡연 상태",
    "stroke": "뇌졸중 발생 여부"
}

if NOTES_PATH.exists():
    try:
        with open(NOTES_PATH, "r", encoding="utf-8") as f:
            saved_notes = json.load(f)
        meanings.update(saved_notes.get("meanings", {}))
    except Exception:
        pass

description_df = pd.DataFrame({
    "변수명": df.columns,
    "의미": [meanings.get(col, "") for col in df.columns],
    "자료형": [str(df[col].dtype) for col in df.columns],
    "결측치 수": [int(df[col].isna().sum()) for col in df.columns]
})

edited_df = st.data_editor(
    description_df,
    use_container_width=True,
    hide_index=True,
    disabled=["변수명", "자료형", "결측치 수"],
    column_config={
        "변수명": st.column_config.TextColumn("변수명"),
        "의미": st.column_config.TextColumn("변수의 의미"),
        "자료형": st.column_config.TextColumn("자료형"),
        "결측치 수": st.column_config.NumberColumn("결측치 수")
    }
)


# 자료 저장
st.subheader("변수 설명 저장")

if st.button("변수 설명 저장"):
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "meanings": dict(zip(
            edited_df["변수명"],
            edited_df["의미"]
        ))
    }

    if NOTES_PATH.exists():
        try:
            with open(NOTES_PATH, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            save_data["source"] = old_data.get("source", "")
        except Exception:
            pass

    with open(NOTES_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)

    st.success("변수 설명이 저장되었습니다.")


st.divider()


# 데이터 미리보기
st.subheader("데이터 미리보기")

st.dataframe(
    df.head(),
    use_container_width=True
)


st.divider()


# 데이터 출처
st.subheader("데이터 출처")

default_source = "https://github.com/greatsong/modudata"

if NOTES_PATH.exists():
    try:
        with open(NOTES_PATH, "r", encoding="utf-8") as f:
            old_data = json.load(f)
        default_source = old_data.get("source", default_source)
    except Exception:
        pass

source = st.text_input(
    "데이터 출처",
    value=default_source
)

if st.button("출처 저장"):
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "meanings": dict(zip(
            edited_df["변수명"],
            edited_df["의미"]
        )),
        "source": source
    }

    with open(NOTES_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)

    st.success("데이터 출처가 저장되었습니다.")


st.divider()

st.caption(
    "본 실습은 머신러닝의 데이터 분석과 분류 성능 평가를 학습하기 위한 교육용 프로젝트입니다."
)

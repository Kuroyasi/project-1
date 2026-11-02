import json
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
RECORD_DIR = Path(__file__).resolve().parent / "내_기록"
RECORD_FILE = RECORD_DIR / "stroke_실습실_기록.json"

st.set_page_config(page_title="뇌졸중 예측 실습실", page_icon="🧠", layout="wide")


def load_notes():
    if not RECORD_FILE.exists():
        return {"meanings": {}, "source": ""}
    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            saved = json.load(file)
        return {"meanings": saved.get("meanings", {}), "source": saved.get("source", "")}
    except (OSError, json.JSONDecodeError, AttributeError):
        st.warning("기존 기록을 읽지 못해 새 기록으로 시작합니다.")
        return {"meanings": {}, "source": ""}


def save_notes(meanings, source):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump({"meanings": meanings, "source": source}, file, ensure_ascii=False, indent=2)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🧠 뇌졸중 예측 실습실")
st.write("뇌졸중 관련 공개 데이터를 살펴보고, 각 열의 의미와 데이터 구조를 직접 정리하는 교육용 실습실입니다.")

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.caption(f"오류 정보: {error}")
    st.stop()

expected = ["id", "gender", "age", "hypertension", "heart_disease", "ever_married",
            "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status", "stroke"]
missing = [column for column in expected if column not in df.columns]
if missing:
    st.error("데이터에 필요한 열이 없습니다: " + ", ".join(missing))
    st.stop()

stroke_count = int(df["stroke"].eq(1).sum())
stroke_rate = df["stroke"].eq(1).mean() * 100 if len(df) else 0.0

st.subheader("데이터 한눈에 보기")
c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 사람 수", f"{len(df):,}명")
c2.metric("열 개수", f"{len(df.columns):,}개")
c3.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
c4.metric("stroke가 1인 비율", f"{stroke_rate:.2f}%")

st.divider()
st.subheader("열 이름과 데이터 구조")
st.write("우리말 뜻 칸을 교재를 참고해 직접 채워 보세요. 입력 내용은 자동 저장됩니다.")
saved = load_notes()
rows = [{
    "열 이름": col,
    "우리말 뜻": saved["meanings"].get(col, ""),
    "값의 종류": str(df[col].dtype),
    "빈 값 개수": int(df[col].isna().sum()),
} for col in df.columns]

edited = st.data_editor(
    pd.DataFrame(rows), key="meaning_editor", use_container_width=True,
    hide_index=True, num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻", help="교재를 참고해 직접 입력하세요."),
        "값의 종류": st.column_config.TextColumn("값의 종류"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
    },
)
meanings = {
    str(row["열 이름"]): str(row["우리말 뜻"]) if pd.notna(row["우리말 뜻"]) else ""
    for _, row in edited.iterrows()
}

st.subheader("데이터 미리 보기")
st.write("원본 데이터의 처음 다섯 줄입니다.")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

st.divider()
st.subheader("데이터 출처")
source = st.text_input("교재에 나온 데이터 출처를 입력하세요.",
                       value=saved.get("source", ""), key="source_input",
                       placeholder="교재에 적힌 출처를 입력하세요.")
try:
    save_notes(meanings, source)
    st.caption(f"입력 내용 저장 위치: {RECORD_FILE.as_posix()}")
except OSError as error:
    st.error(f"기록 저장에 실패했습니다: {error}")

st.caption("주의: 이 앱은 데이터 분석과 머신러닝 학습을 위한 교육용 도구입니다. 실제 개인의 뇌졸중 진단이나 의료적 판단에 사용할 수 없습니다.")

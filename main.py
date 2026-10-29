import json
from pathlib import Path

import pandas as pd
import streamlit as st

# 앱 설정: 브라우저 탭 제목과 아이콘
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "stroke_실습실_기록.json"

COLUMNS = [
    "id", "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status", "stroke",
]


def load_notes():
    """저장된 우리말 뜻과 출처를 불러온다."""
    if not RECORD_FILE.exists():
        return {"meanings": {}, "source": ""}
    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            saved = json.load(file)
        if not isinstance(saved, dict):
            return {"meanings": {}, "source": ""}
        return {
            "meanings": saved.get("meanings", {}),
            "source": saved.get("source", ""),
        }
    except (OSError, json.JSONDecodeError):
        st.warning("기록 JSON 파일을 읽지 못해 빈 기록으로 시작합니다.")
        return {"meanings": {}, "source": ""}


def save_notes(meanings, source):
    """입력한 내용을 내_기록 폴더의 JSON 파일에 저장한다."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "meanings": meanings,
        "source": source,
    }
    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


saved_notes = load_notes()

st.title("🧠 뇌졸중 예측 실습실")
st.write(
    "이 실습실에서는 뇌졸중 관련 데이터를 살펴보고, 각 열이 어떤 정보를 담고 있는지 "
    "직접 정리할 수 있습니다. 아래의 우리말 뜻과 데이터 출처를 교재를 참고해 작성해 보세요."
)

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.caption(f"오류 정보: {error}")
    st.stop()

# 제공된 열 목록과 실제 데이터의 열을 비교해 데이터 구조를 확인한다.
missing_columns = [column for column in COLUMNS if column not in df.columns]
if missing_columns:
    st.error("CSV 파일에서 다음 열을 찾지 못했습니다: " + ", ".join(missing_columns))
    st.stop()

stroke_count = int((df["stroke"] == 1).sum())
stroke_rate = (stroke_count / len(df) * 100) if len(df) else 0.0

st.subheader("데이터 한눈에 보기")
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("전체 사람 수", f"{len(df):,}명")
metric2.metric("열 개수", f"{len(df.columns):,}개")
metric3.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
metric4.metric("stroke가 1인 비율", f"{stroke_rate:.2f}%")

st.divider()
st.subheader("열 이름과 데이터 구조")
st.write("‘우리말 뜻’ 칸을 직접 편집하세요. 입력 내용은 자동으로 JSON 파일에 저장됩니다.")

meaning_rows = []
saved_meanings = saved_notes.get("meanings", {})
for column in df.columns:
    meaning_rows.append({
        "열 이름": column,
        "우리말 뜻": saved_meanings.get(column, ""),
        "값의 종류": str(df[column].dtype),
        "빈 값 개수": int(df[column].isna().sum()),
    })

meaning_df = pd.DataFrame(meaning_rows)

edited_df = st.data_editor(
    meaning_df,
    key="meaning_editor",
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고해 각 열의 뜻을 직접 입력하세요.",
        ),
        "값의 종류": st.column_config.TextColumn("값의 종류"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
    },
)

meanings_to_save = {
    str(row["열 이름"]): str(row["우리말 뜻"]) if pd.notna(row["우리말 뜻"]) else ""
    for _, row in edited_df.iterrows()
}

st.subheader("데이터 미리 보기")
st.write("원본 데이터의 처음 다섯 줄입니다. 값은 변경하지 않고 그대로 표시합니다.")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

st.divider()
st.subheader("데이터 출처")
source_text = st.text_input(
    "교재에 나온 데이터 출처를 입력하세요.",
    value=str(saved_notes.get("source", "")),
    key="source_input",
    placeholder="예: 교재에 적힌 데이터 출처",
)

# 위젯의 현재 값을 실행 때마다 저장해 새로고침 후에도 유지한다.
try:
    save_notes(meanings_to_save, source_text)
    st.caption(f"기록 저장 위치: {RECORD_FILE.as_posix()}")
except OSError as error:
    st.error(f"기록을 저장하지 못했습니다: {error}")

st.caption(
    "참고: 이 앱은 교육용 데이터 탐색 실습입니다. 데이터의 통계적 패턴만으로 "
    "개인의 뇌졸중 여부를 진단할 수 없습니다."
)

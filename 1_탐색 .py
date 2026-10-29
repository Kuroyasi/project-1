import pandas as pd
import plotly.express as px
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide",
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🔎 데이터 탐색")
st.write(
    "나이, 평균 혈당, 고혈압, 심장병, BMI 결측 여부, 흡연 상태에 따라 "
    "데이터의 분포와 뇌졸중 비율을 살펴봅니다."
)

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.caption(f"오류 정보: {error}")
    st.stop()

required_columns = [
    "age", "avg_glucose_level", "stroke", "hypertension",
    "heart_disease", "bmi", "smoking_status",
]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    st.error("CSV 파일에서 다음 열을 찾지 못했습니다: " + ", ".join(missing_columns))
    st.stop()

# 나이와 평균 혈당 분포
st.subheader("1. 나이와 평균 혈당의 분포")
left, right = st.columns(2)

with left:
    age_fig = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "사람 수"},
    )
    age_fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(age_fig, use_container_width=True)

with right:
    glucose_fig = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당", "count": "사람 수"},
    )
    glucose_fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(glucose_fig, use_container_width=True)

st.divider()

# 뇌졸중 경험 여부에 따른 나이와 평균 혈당 상자그림
st.subheader("2. 뇌졸중 경험 여부에 따른 비교")
grouped = df.copy()
grouped["뇌졸중 경험"] = grouped["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})
grouped = grouped[grouped["뇌졸중 경험"].notna()]

left, right = st.columns(2)

with left:
    age_box = px.box(
        grouped,
        x="뇌졸중 경험",
        y="age",
        color="뇌졸중 경험",
        title="뇌졸중 경험 여부별 나이",
        labels={"age": "나이", "뇌졸중 경험": "그룹"},
        category_orders={"뇌졸중 경험": ["뇌졸중 없음", "뇌졸중 있음"]},
    )
    age_box.update_layout(showlegend=False)
    st.plotly_chart(age_box, use_container_width=True)

with right:
    glucose_box = px.box(
        grouped,
        x="뇌졸중 경험",
        y="avg_glucose_level",
        color="뇌졸중 경험",
        title="뇌졸중 경험 여부별 평균 혈당",
        labels={"avg_glucose_level": "평균 혈당", "뇌졸중 경험": "그룹"},
        category_orders={"뇌졸중 경험": ["뇌졸중 없음", "뇌졸중 있음"]},
    )
    glucose_box.update_layout(showlegend=False)
    st.plotly_chart(glucose_box, use_container_width=True)

mean_table = (
    grouped.groupby("뇌졸중 경험", observed=False)[["age", "avg_glucose_level"]]
    .mean()
    .reindex(["뇌졸중 없음", "뇌졸중 있음"])
    .rename(columns={"age": "평균 나이", "avg_glucose_level": "평균 혈당"})
    .round(2)
)
st.write("그룹별 평균값")
st.dataframe(mean_table, use_container_width=True)

st.divider()

# 고혈압과 심장병 여부별 뇌졸중 비율
st.subheader("3. 고혈압·심장병 여부별 뇌졸중 비율")


def make_rate_table(data, column, label):
    valid = data[data[column].isin([0, 1])]
    result = (
        valid.groupby(column, observed=False)["stroke"]
        .agg(사람_수="count", 뇌졸중_사람_수="sum", 뇌졸중_비율="mean")
        .reset_index()
    )
    result[label] = result[column].map({0: "없음", 1: "있음"})
    result["뇌졸중_비율"] = result["뇌졸중_비율"] * 100
    return result.rename(
        columns={
            "사람_수": "사람 수",
            "뇌졸중_사람_수": "뇌졸중 경험자 수",
            "뇌졸중_비율": "뇌졸중 비율(%)",
        }
    )


left, right = st.columns(2)

with left:
    hypertension_table = make_rate_table(df, "hypertension", "고혈압")
    hypertension_fig = px.bar(
        hypertension_table,
        x="고혈압",
        y="뇌졸중_비율",
        text=hypertension_table["뇌졸중_비율"].map(lambda value: f"{value:.2f}%"),
        title="고혈압 여부별 뇌졸중 비율",
        labels={"고혈압": "고혈압", "뇌졸중_비율": "뇌졸중 비율(%)"},
    )
    hypertension_fig.update_layout(yaxis_ticksuffix="%", yaxis_title="뇌졸중 비율(%)")
    st.plotly_chart(hypertension_fig, use_container_width=True)
    st.dataframe(
        hypertension_table[["고혈압", "사람 수", "뇌졸중 경험자 수", "뇌졸중 비율(%)"]].round(2),
        use_container_width=True,
        hide_index=True,
    )

with right:
    heart_table = make_rate_table(df, "heart_disease", "심장병")
    heart_fig = px.bar(
        heart_table,
        x="심장병",
        y="뇌졸중_비율",
        text=heart_table["뇌졸중_비율"].map(lambda value: f"{value:.2f}%"),
        title="심장병 여부별 뇌졸중 비율",
        labels={"심장병": "심장병", "뇌졸중_비율": "뇌졸중 비율(%)"},
    )
    heart_fig.update_layout(yaxis_ticksuffix="%", yaxis_title="뇌졸중 비율(%)")
    st.plotly_chart(heart_fig, use_container_width=True)
    st.dataframe(
        heart_table[["심장병", "사람 수", "뇌졸중 경험자 수", "뇌졸중 비율(%)"]].round(2),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# BMI 결측 집단과 전체 데이터의 뇌졸중 비율 비교
st.subheader("4. BMI가 비어 있는 사람들의 뇌졸중 비율")
missing_bmi = df[df["bmi"].isna()]
overall_stroke_rate = (df["stroke"].eq(1).mean() * 100) if len(df) else 0.0
missing_bmi_stroke_rate = (
    missing_bmi["stroke"].eq(1).mean() * 100 if len(missing_bmi) else 0.0
)

bmi_comparison = pd.DataFrame([
    {
        "대상": "BMI가 비어 있는 사람",
        "사람 수": len(missing_bmi),
        "뇌졸중 경험자 수": int(missing_bmi["stroke"].eq(1).sum()),
        "뇌졸중 비율(%)": round(missing_bmi_stroke_rate, 2),
    },
    {
        "대상": "전체 사람",
        "사람 수": len(df),
        "뇌졸중 경험자 수": int(df["stroke"].eq(1).sum()),
        "뇌졸중 비율(%)": round(overall_stroke_rate, 2),
    },
])
st.dataframe(bmi_comparison, use_container_width=True, hide_index=True)

st.divider()

# 흡연 상태별 사람 수
st.subheader("5. 흡연 상태별 사람 수")
smoking_counts = (
    df["smoking_status"]
    .fillna("빈 값")
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoking_counts, use_container_width=True, hide_index=True)

st.caption(
    "주의: 이 화면은 데이터의 분포와 집단 간 차이를 탐색하기 위한 교육용 자료입니다. "
    "그래프의 관계만으로 원인이나 개인의 뇌졸중 위험을 확정할 수 없습니다."
)

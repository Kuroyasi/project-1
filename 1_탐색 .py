import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="데이터 탐색",
    page_icon="📊",
    layout="wide"
)

st.title("📊 데이터 탐색")
st.caption("데이터의 분포와 뇌졸중 여부에 따른 차이를 살펴봅니다.")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다.\n\n{e}")
    st.stop()


required_columns = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
    "bmi",
    "smoking_status",
    "stroke"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"데이터에 필요한 열이 없습니다: {missing_columns}")
    st.stop()


# --------------------------------------------------
# 데이터 전처리
# --------------------------------------------------

df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["avg_glucose_level"] = pd.to_numeric(
    df["avg_glucose_level"], errors="coerce"
)
df["stroke"] = pd.to_numeric(df["stroke"], errors="coerce")

df["뇌졸중 여부"] = df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})


# --------------------------------------------------
# 1. 나이와 평균 혈당 분포
# --------------------------------------------------

st.header("1. 나이와 평균 혈당의 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={
            "age": "나이",
            "count": "사람 수"
        }
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수"
    )

    st.plotly_chart(fig_age, use_container_width=True)


with col2:
    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        }
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수"
    )

    st.plotly_chart(fig_glucose, use_container_width=True)


# --------------------------------------------------
# 2. 뇌졸중 여부에 따른 상자그림
# --------------------------------------------------

st.divider()
st.header("2. 뇌졸중 여부에 따른 비교")

stroke_data = df.dropna(
    subset=["stroke", "age", "avg_glucose_level"]
).copy()

col1, col2 = st.columns(2)

with col1:
    fig_age_box = px.box(
        stroke_data,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        title="뇌졸중 여부에 따른 나이 비교",
        labels={
            "age": "나이",
            "뇌졸중 여부": "그룹"
        }
    )

    st.plotly_chart(fig_age_box, use_container_width=True)


with col2:
    fig_glucose_box = px.box(
        stroke_data,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        title="뇌졸중 여부에 따른 평균 혈당 비교",
        labels={
            "avg_glucose_level": "평균 혈당",
            "뇌졸중 여부": "그룹"
        }
    )

    st.plotly_chart(fig_glucose_box, use_container_width=True)


# 평균값 표
st.subheader("두 그룹의 평균값")

mean_table = (
    stroke_data
    .groupby("뇌졸중 여부")[["age", "avg_glucose_level"]]
    .mean()
    .rename(columns={
        "age": "평균 나이",
        "avg_glucose_level": "평균 혈당"
    })
    .round(2)
)

st.dataframe(
    mean_table,
    use_container_width=True
)


# --------------------------------------------------
# 3. 고혈압 여부에 따른 뇌졸중 비율
# --------------------------------------------------

st.divider()
st.header("3. 고혈압과 심장병에 따른 뇌졸중 비율")


def make_stroke_rate_chart(data, column, title):
    grouped = (
        data.groupby(column, dropna=False)["stroke"]
        .agg(["mean", "count"])
        .reset_index()
    )

    grouped["뇌졸중 비율"] = grouped["mean"] * 100

    if column == "hypertension":
        grouped["구분"] = grouped[column].map({
            0: "고혈압 없음",
            1: "고혈압 있음"
        })
    elif column == "heart_disease":
        grouped["구분"] = grouped[column].map({
            0: "심장병 없음",
            1: "심장병 있음"
        })

    grouped["구분"] = grouped["구분"].fillna("결측")

    fig = px.bar(
        grouped,
        x="구분",
        y="뇌졸중 비율",
        text=grouped["뇌졸중 비율"].map(
            lambda value: f"{value:.2f}%"
        ),
        title=title,
        labels={
            "구분": "구분",
            "뇌졸중 비율": "뇌졸중 비율(%)"
        }
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        yaxis_range=[
            0,
            max(5, grouped["뇌졸중 비율"].max() * 1.25)
        ]
    )

    return fig, grouped


col1, col2 = st.columns(2)

with col1:
    hypertension_fig, hypertension_table = make_stroke_rate_chart(
        df,
        "hypertension",
        "고혈압 여부에 따른 뇌졸중 비율"
    )

    st.plotly_chart(
        hypertension_fig,
        use_container_width=True
    )

with col2:
    heart_fig, heart_table = make_stroke_rate_chart(
        df,
        "heart_disease",
        "심장병 여부에 따른 뇌졸중 비율"
    )

    st.plotly_chart(
        heart_fig,
        use_container_width=True
    )


# --------------------------------------------------
# 4. BMI 결측값 분석
# --------------------------------------------------

st.divider()
st.header("4. 체질량지수(BMI) 결측값 분석")

missing_bmi = df[df["bmi"].isna()]
total_stroke_rate = df["stroke"].mean() * 100

if len(missing_bmi) > 0:
    missing_bmi_stroke_rate = (
        missing_bmi["stroke"].mean() * 100
    )
else:
    missing_bmi_stroke_rate = 0


bmi_comparison = pd.DataFrame({
    "구분": [
        "BMI가 비어 있는 사람",
        "전체 사람"
    ],
    "사람 수": [
        len(missing_bmi),
        len(df)
    ],
    "뇌졸중 경험자 수": [
        int(missing_bmi["stroke"].sum()),
        int(df["stroke"].sum())
    ],
    "뇌졸중 비율(%)": [
        missing_bmi_stroke_rate,
        total_stroke_rate
    ]
})

bmi_comparison["뇌졸중 비율(%)"] = (
    bmi_comparison["뇌졸중 비율(%)"].round(2)
)

st.dataframe(
    bmi_comparison,
    use_container_width=True,
    hide_index=True
)

if total_stroke_rate > 0:
    ratio = missing_bmi_stroke_rate / total_stroke_rate

    st.metric(
        "BMI 결측 그룹의 뇌졸중 비율 / 전체 비율",
        f"{ratio:.2f}배"
    )


# --------------------------------------------------
# 5. 흡연 상태별 인원수
# --------------------------------------------------

st.divider()
st.header("5. 흡연 상태별 인원수")

smoking_count = (
    df["smoking_status"]
    .fillna("빈 값")
    .value_counts()
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

st.dataframe(
    smoking_count,
    use_container_width=True,
    hide_index=True
)

fig_smoking = px.bar(
    smoking_count,
    x="흡연 상태",
    y="사람 수",
    text="사람 수",
    title="흡연 상태별 인원수",
    labels={
        "흡연 상태": "흡연 상태",
        "사람 수": "사람 수"
    }
)

fig_smoking.update_traces(
    textposition="outside"
)

st.plotly_chart(
    fig_smoking,
    use_container_width=True
)


# --------------------------------------------------
# 마무리
# --------------------------------------------------

st.divider()

st.info(
    """
    두 그룹의 평균이나 비율이 다르다는 사실만으로
    개인의 뇌졸중 여부를 정확하게 예측할 수 있는 것은 아닙니다.

    실제 예측에 얼마나 도움이 되는지는 분류 모델을 학습하고
    테스트 데이터로 평가해야 알 수 있습니다.
    """
)

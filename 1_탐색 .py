import streamlit as st
import pandas as pd
import plotly.express as px

# 페이지 설정
st.set_page_config(
    page_title="데이터 탐색",
    page_icon="📊",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


# 데이터 불러오기
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("📊 데이터 탐색")
st.write("뇌졸중 데이터의 분포와 변수 사이의 관계를 그래프로 살펴봅니다.")

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.exception(error)
    st.stop()


# 필요한 열 확인
required_columns = [
    "age",
    "avg_glucose_level",
    "stroke",
    "hypertension",
    "heart_disease",
    "bmi",
    "smoking_status"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"필요한 열이 없습니다: {', '.join(missing_columns)}")
    st.stop()


# 숫자형 데이터 변환
numeric_columns = [
    "age",
    "avg_glucose_level",
    "stroke",
    "hypertension",
    "heart_disease",
    "bmi"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# 뇌졸중 값이 0 또는 1인 데이터만 사용
valid_df = df[df["stroke"].isin([0, 1])].copy()

valid_df["뇌졸중 여부"] = valid_df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})


# ==================================================
# 1. 나이와 평균 혈당의 분포
# ==================================================

st.header("1. 나이와 평균 혈당의 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df.dropna(subset=["age"]),
        x="age",
        nbins=30,
        title="나이 분포",
        labels={
            "age": "나이",
            "count": "인원수"
        },
        template="plotly_white"
    )

    fig_age.update_layout(
        height=400,
        xaxis_title="나이",
        yaxis_title="인원수"
    )

    st.plotly_chart(fig_age, use_container_width=True)


with col2:
    fig_glucose = px.histogram(
        df.dropna(subset=["avg_glucose_level"]),
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "인원수"
        },
        template="plotly_white"
    )

    fig_glucose.update_layout(
        height=400,
        xaxis_title="평균 혈당",
        yaxis_title="인원수"
    )

    st.plotly_chart(fig_glucose, use_container_width=True)


st.divider()


# ==================================================
# 2. 뇌졸중 여부에 따른 나이와 혈당 비교
# ==================================================

st.header("2. 뇌졸중 여부에 따른 비교")

col1, col2 = st.columns(2)

with col1:
    age_box_data = valid_df.dropna(subset=["age"])

    fig_age_box = px.box(
        age_box_data,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        category_orders={
            "뇌졸중 여부": ["뇌졸중 없음", "뇌졸중 있음"]
        },
        title="뇌졸중 여부에 따른 나이",
        labels={
            "age": "나이",
            "뇌졸중 여부": "그룹"
        },
        template="plotly_white"
    )

    fig_age_box.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="나이"
    )

    st.plotly_chart(fig_age_box, use_container_width=True)


with col2:
    glucose_box_data = valid_df.dropna(
        subset=["avg_glucose_level"]
    )

    fig_glucose_box = px.box(
        glucose_box_data,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        category_orders={
            "뇌졸중 여부": ["뇌졸중 없음", "뇌졸중 있음"]
        },
        title="뇌졸중 여부에 따른 평균 혈당",
        labels={
            "avg_glucose_level": "평균 혈당",
            "뇌졸중 여부": "그룹"
        },
        template="plotly_white"
    )

    fig_glucose_box.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="평균 혈당"
    )

    st.plotly_chart(fig_glucose_box, use_container_width=True)


# 그룹별 평균값
st.subheader("그룹별 평균값")

mean_table = (
    valid_df
    .groupby("뇌졸중 여부")[
        ["age", "avg_glucose_level"]
    ]
    .mean()
    .reindex(["뇌졸중 없음", "뇌졸중 있음"])
    .rename(columns={
        "age": "평균 나이",
        "avg_glucose_level": "평균 혈당"
    })
    .round(2)
)

st.dataframe(mean_table, use_container_width=True)

st.divider()


# ==================================================
# 3. 고혈압 및 심장병 유무에 따른 뇌졸중 비율
# ==================================================

st.header("3. 질환 유무에 따른 뇌졸중 비율")


def make_stroke_rate_chart(data, column, title, labels):

    chart_data = data[
        data[column].isin([0, 1])
    ].copy()

    chart_data["상태"] = chart_data[column].map(labels)

    rate_data = (
        chart_data
        .groupby("상태")["stroke"]
        .mean()
        .reset_index()
    )

    rate_data["뇌졸중 비율(%)"] = rate_data["stroke"] * 100

    rate_data["상태"] = pd.Categorical(
        rate_data["상태"],
        categories=[labels[0], labels[1]],
        ordered=True
    )

    rate_data = rate_data.sort_values("상태")

    fig = px.bar(
        rate_data,
        x="상태",
        y="뇌졸중 비율(%)",
        text=rate_data["뇌졸중 비율(%)"].map(
            lambda x: f"{x:.2f}%"
        ),
        title=title,
        template="plotly_white"
    )

    fig.update_traces(
        textposition="outside"
    )

    max_rate = rate_data["뇌졸중 비율(%)"].max()

    fig.update_layout(
        height=400,
        xaxis_title="구분",
        yaxis_title="뇌졸중 비율(%)",
        yaxis_range=[0, max(5, max_rate * 1.2)]
    )

    return fig


col1, col2 = st.columns(2)

with col1:
    hypertension_fig = make_stroke_rate_chart(
        valid_df,
        "hypertension",
        "고혈압 유무에 따른 뇌졸중 비율",
        {
            0: "고혈압 없음",
            1: "고혈압 있음"
        }
    )

    st.plotly_chart(
        hypertension_fig,
        use_container_width=True
    )


with col2:
    heart_disease_fig = make_stroke_rate_chart(
        valid_df,
        "heart_disease",
        "심장병 유무에 따른 뇌졸중 비율",
        {
            0: "심장병 없음",
            1: "심장병 있음"
        }
    )

    st.plotly_chart(
        heart_disease_fig,
        use_container_width=True
    )


st.caption(
    "뇌졸중 비율은 각 그룹에서 stroke 값이 1인 사람의 비율입니다. "
    "이 그래프만으로 질환이 뇌졸중의 직접적인 원인이라고 판단할 수는 없습니다."
)

st.divider()


# ==================================================
# 4. BMI 결측자와 전체 뇌졸중 비율 비교
# ==================================================

st.header("4. BMI 결측 여부와 뇌졸중 비율")

missing_bmi_df = valid_df[
    valid_df["bmi"].isna()
]

overall_stroke_rate = (
    valid_df["stroke"].mean() * 100
)

if len(missing_bmi_df) > 0:
    missing_bmi_rate = (
        missing_bmi_df["stroke"].mean() * 100
    )
else:
    missing_bmi_rate = None


bmi_comparison = pd.DataFrame({
    "구분": [
        "BMI가 비어 있는 사람",
        "전체 사람"
    ],
    "인원수": [
        len(missing_bmi_df),
        len(valid_df)
    ],
    "뇌졸중 비율(%)": [
        missing_bmi_rate,
        overall_stroke_rate
    ]
})

bmi_comparison["뇌졸중 비율(%)"] = (
    bmi_comparison["뇌졸중 비율(%)"].round(2)
)

st.dataframe(
    bmi_comparison,
    hide_index=True,
    use_container_width=True
)

st.divider()


# ==================================================
# 5. 흡연 상태별 인원수
# ==================================================

st.header("5. 흡연 상태별 인원수")

smoking_counts = (
    df["smoking_status"]
    .fillna("결측")
    .value_counts()
    .rename_axis("흡연 상태")
    .reset_index(name="인원수")
)

st.dataframe(
    smoking_counts,
    hide_index=True,
    use_container_width=True
)


# 흡연 상태별 인원수 그래프
fig_smoking = px.bar(
    smoking_counts,
    x="흡연 상태",
    y="인원수",
    text="인원수",
    title="흡연 상태별 인원수",
    template="plotly_white"
)

fig_smoking.update_traces(
    textposition="outside"
)

fig_smoking.update_layout(
    height=400,
    xaxis_title="흡연 상태",
    yaxis_title="인원수"
)

st.plotly_chart(
    fig_smoking,
    use_container_width=True
)


st.divider()

st.caption(
    "이 페이지는 머신러닝 학습을 위한 데이터 탐색용입니다. "
    "실제 개인의 뇌졸중 진단이나 위험 판단에 사용할 수 없습니다."
)


import streamlit as st
import pandas as pd
import plotly.express as px


# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="뇌졸중 예측 실습실 | 데이터 탐색",
    page_icon="🔎",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


# ==========================================
# 2. 데이터 불러오기
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
    st.error("데이터를 불러오지 못했습니다.")
    st.error(f"오류 내용: {e}")
    st.stop()


# 필요한 열 확인
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
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"필요한 열이 없습니다: {missing_columns}")
    st.stop()


# ==========================================
# 3. 페이지 제목
# ==========================================

st.title("🔎 데이터 탐색")

st.markdown(
    """
    뇌졸중 데이터의 주요 변수들을 다양한 그래프로 분석합니다.

    히스토그램과 상자그림을 통해 데이터의 분포를 확인하고,
    고혈압과 심장병 등의 요인이 뇌졸중 발생과 어떤 관계가 있는지 탐색합니다.
    """
)

st.divider()


# ==========================================
# 4. 나이와 평균 혈당 분포
# ==========================================

st.subheader("1. 주요 변수의 분포")

st.markdown(
    """
    전체 데이터에서 나이와 평균 혈당이 어떤 형태로 분포하는지
    히스토그램으로 확인합니다.
    """
)

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
        },
        color_discrete_sequence=["#4C78A8"]
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
        bargap=0.05
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True
    )


with col2:

    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        },
        color_discrete_sequence=["#F58518"]
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
        bargap=0.05
    )

    st.plotly_chart(
        fig_glucose,
        use_container_width=True
    )


st.divider()


# ==========================================
# 5. 뇌졸중 여부에 따른 상자그림
# ==========================================

st.subheader("2. 뇌졸중 여부에 따른 변수 비교")

st.markdown(
    """
    뇌졸중을 겪은 사람과 겪지 않은 사람을 구분하여
    나이와 평균 혈당의 분포를 비교합니다.

    상자그림을 통해 중앙값, 사분위수 및 이상치를 확인할 수 있습니다.
    """
)


# 그래프용 데이터 생성
box_df = df.copy()

box_df["뇌졸중 여부"] = box_df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})

box_df = box_df.dropna(
    subset=["뇌졸중 여부"]
)

box_df["뇌졸중 여부"] = pd.Categorical(
    box_df["뇌졸중 여부"],
    categories=["뇌졸중 없음", "뇌졸중 있음"],
    ordered=True
)


col1, col2 = st.columns(2)

with col1:

    fig_age_box = px.box(
        box_df,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        title="뇌졸중 여부에 따른 나이 비교",
        labels={
            "age": "나이",
            "뇌졸중 여부": "구분"
        },
        color_discrete_sequence=[
            "#4C78A8",
            "#E45756"
        ]
    )

    fig_age_box.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig_age_box,
        use_container_width=True
    )


with col2:

    fig_glucose_box = px.box(
        box_df,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        title="뇌졸중 여부에 따른 평균 혈당 비교",
        labels={
            "avg_glucose_level": "평균 혈당",
            "뇌졸중 여부": "구분"
        },
        color_discrete_sequence=[
            "#4C78A8",
            "#E45756"
        ]
    )

    fig_glucose_box.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig_glucose_box,
        use_container_width=True
    )


# 그룹별 평균 계산
mean_df = (
    box_df
    .groupby("뇌졸중 여부", observed=False)[
        ["age", "avg_glucose_level"]
    ]
    .mean()
    .reset_index()
)

mean_df = mean_df.rename(columns={
    "뇌졸중 여부": "뇌졸중 구분",
    "age": "평균 나이",
    "avg_glucose_level": "평균 혈당"
})

mean_df["평균 나이"] = mean_df["평균 나이"].round(2)
mean_df["평균 혈당"] = mean_df["평균 혈당"].round(2)


st.markdown("**그룹별 평균값 비교**")

st.dataframe(
    mean_df,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ==========================================
# 6. 고혈압과 심장병에 따른 뇌졸중 비율
# ==========================================

st.subheader("3. 질환 유무에 따른 뇌졸중 비율")

st.markdown(
    """
    고혈압과 심장병의 유무에 따라 뇌졸중 발생 비율이
    어떻게 달라지는지 비교합니다.

    각 그룹의 전체 인원 중 뇌졸중을 겪은 사람의 비율을 계산합니다.
    """
)


# 비율 계산 함수
def calculate_stroke_rate(data, column):

    result = (
        data.groupby(column, dropna=False)
        .agg(
            전체 인원=("stroke", "count"),
            뇌졸중 인원=("stroke", lambda x: (x == 1).sum())
        )
        .reset_index()
    )

    result["뇌졸중 비율"] = (
        result["뇌졸중 인원"]
        / result["전체 인원"]
        * 100
    )

    return result


# 고혈압
hypertension_df = calculate_stroke_rate(
    df.dropna(subset=["stroke"]),
    "hypertension"
)

hypertension_df["구분"] = hypertension_df["hypertension"].map({
    0: "고혈압 없음",
    1: "고혈압 있음"
}).fillna("결측값")


# 심장병
heart_df = calculate_stroke_rate(
    df.dropna(subset=["stroke"]),
    "heart_disease"
)

heart_df["구분"] = heart_df["heart_disease"].map({
    0: "심장병 없음",
    1: "심장병 있음"
}).fillna("결측값")


col1, col2 = st.columns(2)

with col1:

    fig_hypertension = px.bar(
        hypertension_df,
        x="구분",
        y="뇌졸중 비율",
        text=hypertension_df["뇌졸중 비율"].apply(
            lambda x: f"{x:.2f}%"
        ),
        title="고혈압 유무에 따른 뇌졸중 비율",
        labels={
            "구분": "고혈압 여부",
            "뇌졸중 비율": "뇌졸중 비율 (%)"
        },
        color="구분",
        color_discrete_sequence=[
            "#4C78A8",
            "#E45756"
        ]
    )

    fig_hypertension.update_layout(
        showlegend=False,
        yaxis_title="뇌졸중 비율 (%)"
    )

    st.plotly_chart(
        fig_hypertension,
        use_container_width=True
    )


with col2:

    fig_heart = px.bar(
        heart_df,
        x="구분",
        y="뇌졸중 비율",
        text=heart_df["뇌졸중 비율"].apply(
            lambda x: f"{x:.2f}%"
        ),
        title="심장병 유무에 따른 뇌졸중 비율",
        labels={
            "구분": "심장병 여부",
            "뇌졸중 비율": "뇌졸중 비율 (%)"
        },
        color="구분",
        color_discrete_sequence=[
            "#4C78A8",
            "#E45756"
        ]
    )

    fig_heart.update_layout(
        showlegend=False,
        yaxis_title="뇌졸중 비율 (%)"
    )

    st.plotly_chart(
        fig_heart,
        use_container_width=True
    )


st.divider()


# ==========================================
# 7. BMI 결측값과 뇌졸중 비율
# ==========================================

st.subheader("4. BMI 결측값 분석")

st.markdown(
    """
    체질량지수(BMI)가 비어 있는 사람들을 따로 구분합니다.

    BMI가 비어 있는 사람들의 뇌졸중 비율과
    전체 데이터의 뇌졸중 비율을 비교합니다.
    """
)


# BMI가 비어 있는 사람
missing_bmi_df = df[df["bmi"].isna()]

# 뇌졸중 데이터가 존재하는 행만 사용
valid_stroke_df = df.dropna(subset=["stroke"])

valid_missing_bmi = missing_bmi_df.dropna(
    subset=["stroke"]
)


# 인원 계산
missing_bmi_count = len(missing_bmi_df)

missing_bmi_stroke_count = int(
    (valid_missing_bmi["stroke"] == 1).sum()
)

total_stroke_count = int(
    (valid_stroke_df["stroke"] == 1).sum()
)


# 비율 계산
missing_bmi_rate = (
    missing_bmi_stroke_count / len(valid_missing_bmi) * 100
    if len(valid_missing_bmi) > 0 else 0
)

total_stroke_rate = (
    total_stroke_count / len(valid_stroke_df) * 100
    if len(valid_stroke_df) > 0 else 0
)


bmi_comparison = pd.DataFrame({
    "구분": [
        "BMI 결측자",
        "전체 사람"
    ],
    "전체 인원": [
        missing_bmi_count,
        len(df)
    ],
    "뇌졸중 인원": [
        missing_bmi_stroke_count,
        total_stroke_count
    ],
    "뇌졸중 비율 (%)": [
        round(missing_bmi_rate, 2),
        round(total_stroke_rate, 2)
    ]
})


st.dataframe(
    bmi_comparison,
    use_container_width=True,
    hide_index=True
)


st.info(
    f"BMI가 비어 있는 사람은 총 {missing_bmi_count:,}명입니다."
)


st.divider()


# ==========================================
# 8. 흡연 상태별 인원수
# ==========================================

st.subheader("5. 흡연 상태별 인원수")

st.markdown(
    """
    흡연 상태의 각 범주에 해당하는 사람 수를 계산합니다.

    결측값이 존재한다면 별도로 표시합니다.
    """
)


smoking_counts = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

smoking_counts["흡연 상태"] = (
    smoking_counts["흡연 상태"]
    .fillna("결측값")
)


st.dataframe(
    smoking_counts,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ==========================================
# 9. 페이지 하단
# ==========================================

st.caption(
    "뇌졸중 예측 실습실 | 데이터 탐색 페이지"
)

st.caption(
    "출처: https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
)

st.caption(
    "본 분석은 데이터의 통계적 특성을 탐색하는 교육용 실습이며, "
    "변수 사이의 관계가 인과관계를 의미하지는 않습니다."
)


from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="데이터 탐색",
    page_icon="📊",
    layout="wide"
)

# pages 폴더의 상위 폴더에서 CSV 찾기
DATA_PATH = Path(__file__).resolve().parent.parent / "stroke.csv"


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"stroke.csv 파일을 찾을 수 없습니다: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH, encoding="utf-8")


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


st.title("📊 데이터 탐색")
st.write("데이터의 분포와 변수 간의 관계를 살펴봅니다.")

st.divider()


# 1. 기본 정보
st.subheader("1. 데이터 기본 정보")

col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터", f"{len(df):,}명")
col2.metric("전체 변수", f"{len(df.columns)}개")
col3.metric(
    "뇌졸중 발생 데이터",
    f"{int((df['stroke'] == 1).sum()):,}명"
)


# 2. 히스토그램
st.divider()
st.subheader("2. 변수별 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "인원"}
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
            "count": "인원"
        }
    )

    st.plotly_chart(fig_glucose, use_container_width=True)


# 3. 뇌졸중 여부에 따른 분포
st.divider()
st.subheader("3. 뇌졸중 여부에 따른 변수 비교")

col1, col2 = st.columns(2)

with col1:
    fig_age_box = px.box(
        df,
        x="stroke",
        y="age",
        color="stroke",
        title="뇌졸중 여부에 따른 나이",
        labels={
            "stroke": "뇌졸중 여부",
            "age": "나이"
        }
    )

    fig_age_box.update_xaxes(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["정상", "뇌졸중"]
    )

    st.plotly_chart(fig_age_box, use_container_width=True)

with col2:
    fig_glucose_box = px.box(
        df,
        x="stroke",
        y="avg_glucose_level",
        color="stroke",
        title="뇌졸중 여부에 따른 평균 혈당",
        labels={
            "stroke": "뇌졸중 여부",
            "avg_glucose_level": "평균 혈당"
        }
    )

    fig_glucose_box.update_xaxes(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["정상", "뇌졸중"]
    )

    st.plotly_chart(fig_glucose_box, use_container_width=True)


# 4. 평균값 비교
st.divider()
st.subheader("4. 뇌졸중 여부에 따른 평균값")

mean_df = df.groupby("stroke")[
    ["age", "avg_glucose_level", "bmi"]
].mean().reset_index()

mean_df["stroke"] = mean_df["stroke"].map({
    0: "정상",
    1: "뇌졸중"
})

mean_df = mean_df.rename(columns={
    "stroke": "구분",
    "age": "평균 나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "평균 BMI"
})

st.dataframe(
    mean_df.round(2),
    use_container_width=True,
    hide_index=True
)


# 5. 고혈압과 심장병
st.divider()
st.subheader("5. 질병 여부에 따른 뇌졸중 발생 비율")

col1, col2 = st.columns(2)

with col1:
    hypertension_df = (
        df.groupby("hypertension")["stroke"]
        .mean()
        .reset_index()
    )

    hypertension_df["hypertension"] = (
        hypertension_df["hypertension"].map({
            0: "고혈압 없음",
            1: "고혈압 있음"
        })
    )

    hypertension_df["stroke"] *= 100

    fig_hypertension = px.bar(
        hypertension_df,
        x="hypertension",
        y="stroke",
        title="고혈압 여부에 따른 뇌졸중 비율",
        labels={
            "hypertension": "고혈압 여부",
            "stroke": "뇌졸중 비율(%)"
        },
        text_auto=".2f"
    )

    st.plotly_chart(
        fig_hypertension,
        use_container_width=True
    )

with col2:
    heart_df = (
        df.groupby("heart_disease")["stroke"]
        .mean()
        .reset_index()
    )

    heart_df["heart_disease"] = (
        heart_df["heart_disease"].map({
            0: "심장병 없음",
            1: "심장병 있음"
        })
    )

    heart_df["stroke"] *= 100

    fig_heart = px.bar(
        heart_df,
        x="heart_disease",
        y="stroke",
        title="심장병 여부에 따른 뇌졸중 비율",
        labels={
            "heart_disease": "심장병 여부",
            "stroke": "뇌졸중 비율(%)"
        },
        text_auto=".2f"
    )

    st.plotly_chart(
        fig_heart,
        use_container_width=True
    )


# 6. BMI 결측치
st.divider()
st.subheader("6. BMI 결측치 분석")

missing_bmi = df[df["bmi"].isna()]
not_missing_bmi = df[df["bmi"].notna()]

overall_rate = df["stroke"].mean() * 100
missing_rate = (
    missing_bmi["stroke"].mean() * 100
    if len(missing_bmi) > 0 else 0
)
not_missing_rate = (
    not_missing_bmi["stroke"].mean() * 100
    if len(not_missing_bmi) > 0 else 0
)

bmi_df = pd.DataFrame({
    "구분": ["BMI 결측", "BMI 정상"],
    "인원": [
        len(missing_bmi),
        len(not_missing_bmi)
    ],
    "뇌졸중 발생 비율(%)": [
        missing_rate,
        not_missing_rate
    ]
})

st.dataframe(
    bmi_df.round(2),
    use_container_width=True,
    hide_index=True
)

st.write(
    f"전체 데이터의 뇌졸중 발생 비율은 {overall_rate:.2f}%입니다."
)


# 7. 흡연 상태
st.divider()
st.subheader("7. 흡연 상태별 데이터 수")

smoking_df = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .reset_index()
)

smoking_df.columns = ["흡연 상태", "인원"]

fig_smoking = px.bar(
    smoking_df,
    x="흡연 상태",
    y="인원",
    title="흡연 상태별 인원",
    text_auto=True
)

st.plotly_chart(
    fig_smoking,
    use_container_width=True
)


st.divider()

st.caption(
    "데이터에서 나타나는 통계적 관계가 인과관계를 의미하지는 않습니다."
)

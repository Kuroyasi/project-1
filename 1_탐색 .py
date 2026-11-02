import pandas as pd
import plotly.express as px
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
st.set_page_config(page_title="탐색 | 뇌졸중 예측 실습실", page_icon="🔎", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🔎 데이터 탐색")
st.write("데이터의 분포와 뇌졸중 경험 여부에 따른 집단별 차이를 살펴봅니다.")
try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요.")
    st.caption(f"오류 정보: {error}")
    st.stop()

required = ["age", "avg_glucose_level", "stroke", "hypertension", "heart_disease", "bmi", "smoking_status"]
missing = [col for col in required if col not in df.columns]
if missing:
    st.error("데이터에 필요한 열이 없습니다: " + ", ".join(missing))
    st.stop()

st.subheader("1. 나이와 평균 혈당의 분포")
left, right = st.columns(2)
with left:
    fig = px.histogram(df, x="age", nbins=30, title="나이 분포",
                       labels={"age": "나이", "count": "사람 수"})
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.histogram(df, x="avg_glucose_level", nbins=30, title="평균 혈당 분포",
                       labels={"avg_glucose_level": "평균 혈당", "count": "사람 수"})
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("2. 뇌졸중 경험 여부에 따른 비교")
grouped = df.copy()
grouped["뇌졸중 경험"] = grouped["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})
grouped = grouped[grouped["뇌졸중 경험"].notna()]
order = ["뇌졸중 없음", "뇌졸중 있음"]
left, right = st.columns(2)
with left:
    fig = px.box(grouped, x="뇌졸중 경험", y="age", color="뇌졸중 경험",
                 category_orders={"뇌졸중 경험": order}, title="뇌졸중 경험 여부별 나이",
                 labels={"age": "나이", "뇌졸중 경험": "그룹"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.box(grouped, x="뇌졸중 경험", y="avg_glucose_level", color="뇌졸중 경험",
                 category_orders={"뇌졸중 경험": order}, title="뇌졸중 경험 여부별 평균 혈당",
                 labels={"avg_glucose_level": "평균 혈당", "뇌졸중 경험": "그룹"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

means = (grouped.groupby("뇌졸중 경험")[["age", "avg_glucose_level"]].mean()
         .reindex(order).rename(columns={"age": "평균 나이", "avg_glucose_level": "평균 혈당"}).round(2))
st.write("두 그룹의 평균값")
st.dataframe(means, use_container_width=True)

st.divider()
st.subheader("3. 고혈압·심장병 여부별 뇌졸중 비율")


def rate_table(column, label):
    valid = df[df[column].isin([0, 1])]
    result = valid.groupby(column)["stroke"].agg(사람_수="count", 뇌졸중_경험자="sum", 비율="mean").reset_index()
    result[label] = result[column].map({0: "없음", 1: "있음"})
    result["비율"] *= 100
    return result


left, right = st.columns(2)
with left:
    data = rate_table("hypertension", "고혈압")
    fig = px.bar(data, x="고혈압", y="비율",
                 text=data["비율"].map(lambda x: f"{x:.2f}%"),
                 title="고혈압 여부별 뇌졸중 비율",
                 labels={"비율": "뇌졸중 비율(%)", "고혈압": "고혈압"})
    fig.update_layout(yaxis_ticksuffix="%", yaxis_title="뇌졸중 비율(%)")
    st.plotly_chart(fig, use_container_width=True)
    shown = data.rename(columns={"사람_수": "사람 수", "뇌졸중_경험자": "뇌졸중 경험자 수", "비율": "뇌졸중 비율(%)"})
    st.dataframe(shown[["고혈압", "사람 수", "뇌졸중 경험자 수", "뇌졸중 비율(%)"]].round(2),
                 use_container_width=True, hide_index=True)
with right:
    data = rate_table("heart_disease", "심장병")
    fig = px.bar(data, x="심장병", y="비율",
                 text=data["비율"].map(lambda x: f"{x:.2f}%"),
                 title="심장병 여부별 뇌졸중 비율",
                 labels={"비율": "뇌졸중 비율(%)", "심장병": "심장병"})
    fig.update_layout(yaxis_ticksuffix="%", yaxis_title="뇌졸중 비율(%)")
    st.plotly_chart(fig, use_container_width=True)
    shown = data.rename(columns={"사람_수": "사람 수", "뇌졸중_경험자": "뇌졸중 경험자 수", "비율": "뇌졸중 비율(%)"})
    st.dataframe(shown[["심장병", "사람 수", "뇌졸중 경험자 수", "뇌졸중 비율(%)"]].round(2),
                 use_container_width=True, hide_index=True)

st.divider()
st.subheader("4. BMI가 비어 있는 사람들의 뇌졸중 비율")
missing_bmi = df[df["bmi"].isna()]
overall_rate = df["stroke"].eq(1).mean() * 100 if len(df) else 0
missing_rate = missing_bmi["stroke"].eq(1).mean() * 100 if len(missing_bmi) else 0
comparison = pd.DataFrame([
    {"대상": "BMI가 비어 있는 사람", "사람 수": len(missing_bmi),
     "뇌졸중 경험자 수": int(missing_bmi["stroke"].eq(1).sum()), "뇌졸중 비율(%)": round(missing_rate, 2)},
    {"대상": "전체 사람", "사람 수": len(df),
     "뇌졸중 경험자 수": int(df["stroke"].eq(1).sum()), "뇌졸중 비율(%)": round(overall_rate, 2)},
])
st.dataframe(comparison, use_container_width=True, hide_index=True)

st.divider()
st.subheader("5. 흡연 상태별 사람 수")
smoking = (df["smoking_status"].fillna("빈 값").value_counts(dropna=False)
           .rename_axis("흡연 상태").reset_index(name="사람 수"))
st.dataframe(smoking, use_container_width=True, hide_index=True)
st.caption("주의: 집단 간 통계적 차이는 인과관계를 의미하지 않으며, 개인의 의료적 판단에 사용할 수 없습니다.")

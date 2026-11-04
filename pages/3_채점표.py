
from pathlib import Path

import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(
    page_title="모델 채점표",
    page_icon="📋",
    layout="wide"
)

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


st.title("📋 모델 채점표")
st.write("분류 모델의 예측 결과를 혼동 행렬과 평가 지표로 분석합니다.")

st.divider()


# 사용할 변수 선택
st.subheader("1. 평가 변수 선택")

feature_labels = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

features = st.multiselect(
    "모델 학습에 사용할 변수를 선택하세요.",
    options=list(feature_labels.keys()),
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda x: feature_labels[x]
)

if len(features) < 2:
    st.warning("최소 2개의 변수를 선택해야 합니다.")
    st.stop()


# 데이터 분리
df = df.sort_values("id").reset_index(drop=True)

positions = list(range(len(df)))
test_mask = [position % 10 < 3 for position in positions]

train_df = df.loc[
    [not value for value in test_mask]
].copy()

test_df = df.loc[test_mask].copy()

X_train = train_df[features].copy()
X_test = test_df[features].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)


# BMI 결측치 처리
if "bmi" in features:
    bmi_median = X_train["bmi"].median()

    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)


# 표준화
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 모델 학습
@st.cache_resource
def train_models(feature_names, train_x, train_y, train_x_scaled):
    logistic = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        random_state=42
    )

    logistic.fit(train_x_scaled, train_y)
    tree.fit(train_x, train_y)

    return logistic, tree


logistic_model, tree_model = train_models(
    tuple(features),
    X_train,
    y_train,
    X_train_scaled
)


# 예측
logistic_pred = logistic_model.predict(X_test_scaled)
tree_pred = tree_model.predict(X_test)

baseline_class = int(y_train.mode().iloc[0])

baseline_pred = [
    baseline_class for _ in range(len(y_test))
]


# 예측 결과를 Series로 변환
predictions = {
    "로지스틱 회귀": pd.Series(
        logistic_pred,
        index=y_test.index
    ),
    "결정 트리": pd.Series(
        tree_pred,
        index=y_test.index
    ),
    "기준 모델": pd.Series(
        baseline_pred,
        index=y_test.index
    )
}


# 테스트 데이터 정보
test_people = len(test_df)
test_positive = int((y_test == 1).sum())

st.info(
    f"테스트용 사람 수: {test_people:,}명 | "
    f"그중 뇌졸중인 사람 수: {test_positive:,}명"
)


# 혼동 행렬 계산
def calculate_confusion(actual, predicted):
    actual = pd.Series(actual)
    predicted = pd.Series(
        predicted,
        index=actual.index
    )

    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())

    return tn, fp, fn, tp


# 혼동 행렬 출력
def display_confusion_matrix(name, actual, predicted):
    tn, fp, fn, tp = calculate_confusion(
        actual,
        predicted
    )

    matrix_df = pd.DataFrame({
        "예측 정상(0)": [
            tn,
            fn,
            tn + fn
        ],
        "예측 뇌졸중(1)": [
            fp,
            tp,
            fp + tp
        ],
        "전체": [
            tn + fp,
            fn + tp,
            tn + fp + fn + tp
        ]
    }, index=[
        "실제 정상(0)",
        "실제 뇌졸중(1)",
        "전체"
    ])

    st.markdown(f"**{name}**")

    st.dataframe(
        matrix_df,
        use_container_width=True
    )

    st.caption(
        f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}"
    )


# 혼동 행렬 비교
st.divider()
st.subheader("2. 혼동 행렬 비교")

col1, col2 = st.columns(2)

with col1:
    display_confusion_matrix(
        "로지스틱 회귀",
        y_test,
        predictions["로지스틱 회귀"]
    )

with col2:
    display_confusion_matrix(
        "결정 트리",
        y_test,
        predictions["결정 트리"]
    )

st.divider()

display_confusion_matrix(
    "기준 모델",
    y_test,
    predictions["기준 모델"]
)


# 평가 지표 계산
def calculate_metrics(actual, predicted):
    tn, fp, fn, tp = calculate_confusion(
        actual,
        predicted
    )

    total = tn + fp + fn + tp

    accuracy = (
        (tn + tp) / total
        if total > 0 else None
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0 else None
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else None
    )

    if precision is not None and recall is not None:
        if precision + recall > 0:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )
        else:
            f1 = 0
    else:
        f1 = None

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1 점수": f1
    }


def format_metric(value):
    if value is None:
        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


# 지표 비교
st.subheader("3. 모델 평가 지표 비교")

metric_rows = []

for name, prediction in predictions.items():
    metrics = calculate_metrics(
        y_test,
        prediction
    )

    metric_rows.append({
        "모델": name,
        "정확도": format_metric(metrics["정확도"]),
        "재현율": format_metric(metrics["재현율"]),
        "정밀도": format_metric(metrics["정밀도"]),
        "F1 점수": format_metric(metrics["F1 점수"])
    })

metric_df = pd.DataFrame(metric_rows)

st.dataframe(
    metric_df,
    use_container_width=True,
    hide_index=True
)


# 평가 지표 설명
with st.expander("평가 지표 설명"):
    st.markdown(
        """
        **정확도 (Accuracy)**

        전체 예측 중 실제 정답을 맞힌 비율입니다.

        **재현율 (Recall)**

        실제 뇌졸중 환자 중 모델이 뇌졸중이라고 예측한 비율입니다.

        **정밀도 (Precision)**

        모델이 뇌졸중이라고 예측한 사람 중 실제 뇌졸중 환자의 비율입니다.

        **F1 점수 (F1-score)**

        정밀도와 재현율의 조화평균입니다.
        """
    )


# FN / FP 분석
st.divider()
st.subheader("4. 오분류 데이터 분석")

selected_model = st.selectbox(
    "분석할 모델 선택",
    options=[
        "로지스틱 회귀",
        "결정 트리"
    ]
)

selected_prediction = predictions[selected_model]

# FN: 실제 뇌졸중이지만 정상으로 예측
fn_mask = (
    (y_test == 1) &
    (selected_prediction == 0)
)

fn_df = test_df.loc[fn_mask].copy()


# FP: 실제 정상인데 뇌졸중으로 예측
fp_mask = (
    (y_test == 0) &
    (selected_prediction == 1)
)

fp_df = test_df.loc[fp_mask].copy()


display_columns = [
    "id",
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]

display_names = {
    "id": "번호",
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}


def prepare_error_table(data):
    result = data[display_columns].copy()

    result = result.rename(
        columns=display_names
    )

    result["고혈압"] = result["고혈압"].map({
        0: "없음",
        1: "있음"
    })

    result["심장병"] = result["심장병"].map({
        0: "없음",
        1: "있음"
    })

    result = result.sort_values(
        "나이",
        ascending=False
    )

    return result


col1, col2 = st.columns(2)

with col1:
    st.markdown("**FN - 실제 뇌졸중을 정상으로 예측**")

    st.metric(
        "FN 인원",
        f"{len(fn_df):,}명"
    )

    if len(fn_df) > 0:
        st.dataframe(
            prepare_error_table(fn_df).head(20),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"전체 {len(fn_df):,}명 중 최대 20명을 표시합니다."
        )
    else:
        st.success("FN 데이터가 없습니다.")

with col2:
    st.markdown("**FP - 실제 정상인을 뇌졸중으로 예측**")

    st.metric(
        "FP 인원",
        f"{len(fp_df):,}명"
    )

    if len(fp_df) > 0:
        st.dataframe(
            prepare_error_table(fp_df).head(20),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"전체 {len(fp_df):,}명 중 최대 20명을 표시합니다."
        )
    else:
        st.success("FP 데이터가 없습니다.")


# 혼동 행렬과 오분류 목록 비교
st.divider()
st.subheader("5. 오분류 데이터 검증")

tn, fp, fn, tp = calculate_confusion(
    y_test,
    selected_prediction
)

col1, col2 = st.columns(2)

col1.metric(
    "혼동 행렬의 FN",
    f"{fn}명"
)

col2.metric(
    "FN 목록의 인원",
    f"{len(fn_df)}명"
)

if fn == len(fn_df):
    st.success("FN 목록과 혼동 행렬의 수가 일치합니다.")
else:
    st.error("FN 목록과 혼동 행렬의 수가 일치하지 않습니다.")


col1, col2 = st.columns(2)

col1.metric(
    "혼동 행렬의 FP",
    f"{fp}명"
)

col2.metric(
    "FP 목록의 인원",
    f"{len(fp_df)}명"
)

if fp == len(fp_df):
    st.success("FP 목록과 혼동 행렬의 수가 일치합니다.")
else:
    st.error("FP 목록과 혼동 행렬의 수가 일치하지 않습니다.")


st.divider()

st.warning(
    "본 프로그램은 머신러닝 분류 실습을 위한 교육용 도구입니다. "
    "실제 환자의 진단, 치료 또는 의료적 판단에 사용할 수 없습니다."
)

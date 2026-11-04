
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="뇌졸중 예측 실습실 | 채점표",
    page_icon="📋",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

RANDOM_STATE = 42

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

ALL_FEATURES = list(FEATURE_NAMES.keys())


# ==========================================
# 2. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.error(f"오류 내용: {e}")
    st.stop()


required_columns = [
    "id",
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
    "stroke"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"필요한 열이 없습니다: {missing_columns}")
    st.stop()


if df["stroke"].isna().any():
    st.error("정답 열인 stroke에 빈 값이 있습니다.")
    st.stop()

if not set(df["stroke"].unique()).issubset({0, 1}):
    st.error("stroke 열에는 0과 1만 있어야 합니다.")
    st.stop()


# ==========================================
# 3. 페이지 제목
# ==========================================

st.title("📋 채점표")

st.markdown(
    """
    학습한 분류 모델이 테스트 데이터의 뇌졸중 여부를
    얼마나 정확하게 예측했는지 평가합니다.

    혼동행렬을 통해 실제 정답과 모델의 예측을 비교하고,
    정확도, 재현율, 정밀도, F1-score를 계산합니다.
    """
)

st.divider()


# ==========================================
# 4. 입력 속성 선택
# ==========================================

st.subheader("1. 입력 속성 선택")

st.markdown(
    """
    분류 모델 페이지와 동일하게 학습에 사용할 속성을 선택합니다.

    처음에는 체질량지수(BMI)를 제외한 네 가지 속성이 선택되어 있습니다.
    """
)

selected_features = st.multiselect(
    "학습에 사용할 속성",
    options=ALL_FEATURES,
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda x: FEATURE_NAMES[x]
)


if len(selected_features) < 2:
    st.warning("입력 속성을 최소 두 개 이상 선택해야 합니다.")
    st.stop()


st.write(
    "**선택한 속성:**",
    ", ".join(FEATURE_NAMES[x] for x in selected_features)
)

st.divider()


# ==========================================
# 5. 데이터 정렬 및 분리
# ==========================================

st.subheader("2. 학습 데이터와 테스트 데이터 분리")

# 번호 순 정렬
df = df.sort_values(
    "id"
).reset_index(drop=True)


# 10명씩 묶어 앞의 3명 테스트용으로 지정
row_positions = np.arange(len(df))

test_mask = (row_positions % 10) < 3

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()


# 입력 변수와 정답 분리
X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)


# 테스트 데이터 인원
test_people = len(test_df)

test_positive = int(
    (y_test == 1).sum()
)

st.write(
    f"**테스트 데이터: {test_people:,}명 / "
    f"그중 실제 뇌졸중인 사람: {test_positive:,}명**"
)


# ==========================================
# 6. BMI 결측값 처리
# ==========================================

if "bmi" in selected_features:

    bmi_median = X_train["bmi"].median()

    X_train["bmi"] = X_train["bmi"].fillna(
        bmi_median
    )

    X_test["bmi"] = X_test["bmi"].fillna(
        bmi_median
    )

    st.info(
        f"BMI 결측값을 학습 데이터의 중앙값 "
        f"{bmi_median:.2f}으로 채웠습니다."
    )


# ==========================================
# 7. 표준화
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ==========================================
# 8. 로지스틱 회귀 학습
# ==========================================

st.subheader(
    "3. 로지스틱 회귀 (확률로 답하는 모델)"
)

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE
)

logistic_model.fit(
    X_train_scaled,
    y_train
)


logistic_train_pred = logistic_model.predict(
    X_train_scaled
)

logistic_test_pred = logistic_model.predict(
    X_test_scaled
)


# ==========================================
# 9. 의사결정트리 학습
# ==========================================

st.subheader(
    "4. 의사결정트리 (질문으로 답하는 모델)"
)

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)

tree_model.fit(
    X_train,
    y_train
)


tree_train_pred = tree_model.predict(X_train)

tree_test_pred = tree_model.predict(X_test)


# ==========================================
# 10. 기준 모델
# ==========================================

st.subheader("5. 기준 모델")

# 학습 데이터에서 가장 많은 클래스
majority_class = int(
    y_train.value_counts().idxmax()
)

# 모든 사람을 다수 클래스라고 예측
baseline_train_pred = np.full(
    len(y_train),
    majority_class
)

baseline_test_pred = np.full(
    len(y_test),
    majority_class
)


st.caption(
    "기준 모델은 입력 속성을 전혀 사용하지 않고 "
    "훈련 데이터에서 가장 많은 쪽으로만 답합니다."
)


# ==========================================
# 11. 혼동행렬 계산 함수
# ==========================================

def calculate_confusion(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 실제 뇌졸중이고 뇌졸중이라고 예측
    TP = int(
        ((y_true == 1) & (y_pred == 1)).sum()
    )

    # 실제 정상인데 뇌졸중이라고 예측
    FP = int(
        ((y_true == 0) & (y_pred == 1)).sum()
    )

    # 실제 뇌졸중인데 정상이라고 예측
    FN = int(
        ((y_true == 1) & (y_pred == 0)).sum()
    )

    # 실제 정상이고 정상이라고 예측
    TN = int(
        ((y_true == 0) & (y_pred == 0)).sum()
    )

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN
    }


logistic_cm = calculate_confusion(
    y_test,
    logistic_test_pred
)

tree_cm = calculate_confusion(
    y_test,
    tree_test_pred
)

baseline_cm = calculate_confusion(
    y_test,
    baseline_test_pred
)


# ==========================================
# 12. 혼동행렬 출력 함수
# ==========================================

def display_confusion_matrix(cm):

    TN = cm["TN"]
    FP = cm["FP"]
    FN = cm["FN"]
    TP = cm["TP"]

    matrix_df = pd.DataFrame(
        [
            [
                f"TN: {TN:,}명",
                f"FP: {FP:,}명",
                f"{TN + FP:,}명"
            ],
            [
                f"FN: {FN:,}명",
                f"TP: {TP:,}명",
                f"{FN + TP:,}명"
            ],
            [
                f"{TN + FN:,}명",
                f"{FP + TP:,}명",
                f"{TN + FP + FN + TP:,}명"
            ]
        ],
        index=[
            "실제: 아님 (0)",
            "실제: 뇌졸중 (1)",
            "열 합계"
        ],
        columns=[
            "예측: 아님 (0)",
            "예측: 뇌졸중 (1)",
            "행 합계"
        ]
    )

    return matrix_df


# ==========================================
# 13. 혼동행렬 표시
# ==========================================

st.divider()

st.subheader("6. 모델별 혼동행렬")

st.markdown(
    """
    행은 실제 뇌졸중 여부이고 열은 모델이 예측한 결과입니다.

    각 칸에는 TP, FP, FN, TN 약자와 해당 인원수가 표시됩니다.
    마지막 행과 열에는 합계를 표시합니다.
    """
)


col1, col2 = st.columns(2)

with col1:

    st.markdown(
        "**로지스틱 회귀 (확률로 답하는 모델)**"
    )

    st.dataframe(
        display_confusion_matrix(logistic_cm),
        use_container_width=True
    )


with col2:

    st.markdown(
        "**의사결정트리 (질문으로 답하는 모델)**"
    )

    st.dataframe(
        display_confusion_matrix(tree_cm),
        use_container_width=True
    )


# ==========================================
# 14. 평가 지표 계산 함수
# ==========================================

def calculate_metrics(cm):

    TP = cm["TP"]
    FP = cm["FP"]
    FN = cm["FN"]
    TN = cm["TN"]

    total = TP + FP + FN + TN

    # 정확도
    accuracy = (
        (TP + TN) / total
        if total > 0 else None
    )

    # 재현율: 실제 뇌졸중인 사람 중 찾아낸 비율
    recall = (
        TP / (TP + FN)
        if (TP + FN) > 0 else None
    )

    # 정밀도: 뇌졸중이라고 예측한 사람 중 실제 뇌졸중인 비율
    precision = (
        TP / (TP + FP)
        if (TP + FP) > 0 else None
    )

    # F1-score
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
        "F1-score": f1
    }


logistic_metrics = calculate_metrics(logistic_cm)

tree_metrics = calculate_metrics(tree_cm)

baseline_metrics = calculate_metrics(baseline_cm)


# ==========================================
# 15. 평가 지표 표 생성
# ==========================================

def format_metric(value):

    if value is None:
        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


metrics_df = pd.DataFrame([
    {
        "모델": "로지스틱 회귀 (확률로 답하는 모델)",
        "정확도": format_metric(logistic_metrics["정확도"]),
        "재현율": format_metric(logistic_metrics["재현율"]),
        "정밀도": format_metric(logistic_metrics["정밀도"]),
        "F1-score": format_metric(logistic_metrics["F1-score"])
    },
    {
        "모델": "의사결정트리 (질문으로 답하는 모델)",
        "정확도": format_metric(tree_metrics["정확도"]),
        "재현율": format_metric(tree_metrics["재현율"]),
        "정밀도": format_metric(tree_metrics["정밀도"]),
        "F1-score": format_metric(tree_metrics["F1-score"])
    },
    {
        "모델": "기준 모델 (다수 클래스 예측)",
        "정확도": format_metric(baseline_metrics["정확도"]),
        "재현율": format_metric(baseline_metrics["재현율"]),
        "정밀도": format_metric(baseline_metrics["정밀도"]),
        "F1-score": format_metric(baseline_metrics["F1-score"])
    }
])


st.divider()

st.subheader("7. 평가 지표 비교")

st.markdown(
    """
    세 모델의 테스트 데이터 평가 결과를 비교합니다.

    정확도, 재현율, 정밀도, F1-score는 모두 뇌졸중(1)을 양성으로
    설정하여 계산했습니다.
    """
)


st.dataframe(
    metrics_df,
    use_container_width=True,
    hide_index=True
)


st.caption(
    "재현율 = TP / (TP + FN) | "
    "정밀도 = TP / (TP + FP) | "
    "F1-score = 2 × 정밀도 × 재현율 / (정밀도 + 재현율)"
)


# ==========================================
# 16. 놓친 환자와 헛짚은 사람 분석
# ==========================================

st.divider()

st.subheader("8. 놓친 환자와 헛짚은 사람 확인")

st.markdown(
    """
    실제 정답과 모델의 예측이 다른 사람들을 확인합니다.

    두 모델 중 하나를 선택하면 해당 모델의 테스트 데이터 예측 결과를
    기준으로 목록을 생성합니다.
    """
)


selected_model_name = st.selectbox(
    "확인할 모델 선택",
    options=[
        "로지스틱 회귀 (확률로 답하는 모델)",
        "의사결정트리 (질문으로 답하는 모델)"
    ]
)


# 선택한 모델의 예측값
if selected_model_name.startswith("로지스틱"):

    selected_predictions = logistic_test_pred
    selected_cm = logistic_cm

else:

    selected_predictions = tree_test_pred
    selected_cm = tree_cm


# ==========================================
# 17. 오답 데이터 생성
# ==========================================

result_df = test_df.copy()

result_df["실제 정답"] = y_test.values

result_df["예측 결과"] = selected_predictions


# FN: 실제 뇌졸중인데 정상으로 예측
missed_df = result_df[
    (result_df["실제 정답"] == 1)
    & (result_df["예측 결과"] == 0)
].copy()


# FP: 실제 정상인데 뇌졸중으로 예측
false_alarm_df = result_df[
    (result_df["실제 정답"] == 0)
    & (result_df["예측 결과"] == 1)
].copy()


# ==========================================
# 18. 목록 표시용 데이터 정리
# ==========================================

def prepare_error_list(data):

    display_df = data[
        [
            "id",
            "age",
            "avg_glucose_level",
            "hypertension",
            "heart_disease"
        ]
    ].copy()

    display_df = display_df.rename(
        columns={
            "id": "번호",
            "age": "나이",
            "avg_glucose_level": "평균 혈당",
            "hypertension": "고혈압 여부",
            "heart_disease": "심장병 여부"
        }
    )

    # 숫자 대신 우리말로 표시
    display_df["고혈압 여부"] = (
        display_df["고혈압 여부"]
        .map({
            0: "없음",
            1: "있음"
        })
        .fillna("확인 불가")
    )

    display_df["심장병 여부"] = (
        display_df["심장병 여부"]
        .map({
            0: "없음",
            1: "있음"
        })
        .fillna("확인 불가")
    )

    # 나이가 많은 순서로 정렬
    display_df = display_df.sort_values(
        "나이",
        ascending=False
    )

    # 소수점 정리
    display_df["평균 혈당"] = (
        display_df["평균 혈당"].round(2)
    )

    return display_df


missed_list = prepare_error_list(missed_df)

false_alarm_list = prepare_error_list(false_alarm_df)


# ==========================================
# 19. 두 목록 나란히 표시
# ==========================================

col1, col2 = st.columns(2)


with col1:

    st.markdown(
        f"**놓친 환자 (FN): {len(missed_list):,}명**"
    )

    st.caption(
        "실제 뇌졸중 환자이지만 모델이 뇌졸중이 아니라고 예측한 사람"
    )

    if len(missed_list) > 0:

        st.dataframe(
            missed_list.head(20),
            use_container_width=True,
            hide_index=True
        )

        if len(missed_list) > 20:
            st.caption(
                f"전체 {len(missed_list):,}명 중 나이가 많은 순서로 "
                "앞의 20명만 표시했습니다."
            )

    else:

        st.success("놓친 환자가 없습니다.")


with col2:

    st.markdown(
        f"**헛짚은 사람 (FP): {len(false_alarm_list):,}명**"
    )

    st.caption(
        "실제로는 뇌졸중이 아니지만 모델이 뇌졸중이라고 예측한 사람"
    )

    if len(false_alarm_list) > 0:

        st.dataframe(
            false_alarm_list.head(20),
            use_container_width=True,
            hide_index=True
        )

        if len(false_alarm_list) > 20:
            st.caption(
                f"전체 {len(false_alarm_list):,}명 중 나이가 많은 순서로 "
                "앞의 20명만 표시했습니다."
            )

    else:

        st.success("헛짚은 사람이 없습니다.")


# ==========================================
# 20. 혼동행렬과 목록 인원 비교
# ==========================================

st.divider()

st.subheader("9. 혼동행렬과 오답 목록 확인")


fn_count = selected_cm["FN"]
fp_count = selected_cm["FP"]

missed_count = len(missed_list)
false_alarm_count = len(false_alarm_list)


st.write(
    f"**놓친 환자 목록 {missed_count:,}명 = "
    f"혼동행렬 FN {fn_count:,}명 / "
    f"헛짚은 사람 목록 {false_alarm_count:,}명 = "
    f"혼동행렬 FP {fp_count:,}명**"
)


if (
    missed_count == fn_count
    and false_alarm_count == fp_count
):
    st.success(
        "두 목록의 인원수가 혼동행렬의 FN, FP 값과 일치합니다."
    )

else:
    st.error(
        "목록과 혼동행렬의 인원수가 일치하지 않습니다."
    )


# ==========================================
# 21. 마무리
# ==========================================

st.divider()

st.subheader("평가 결과 해석")

st.markdown(
    """
    정확도는 전체 예측 중 정답을 맞힌 비율입니다.

    재현율은 실제 뇌졸중 환자 중 모델이 찾아낸 비율이므로,
    뇌졸중 환자를 놓치지 않는 것이 중요한 상황에서 의미가 큽니다.

    정밀도는 뇌졸중이라고 예측한 사람 중 실제 뇌졸중인 사람의 비율입니다.

    F1-score는 정밀도와 재현율을 함께 고려하는 지표입니다.

    따라서 모델을 비교할 때는 정확도 하나만 확인하기보다
    재현율, 정밀도, F1-score를 함께 살펴보는 것이 좋습니다.
    """
)

st.caption(
    "뇌졸중 예측 실습실 | 채점표 페이지"
)

st.caption(
    "본 분석은 교육용 머신러닝 실습이며 실제 의료 진단에 사용할 수 없습니다."
)

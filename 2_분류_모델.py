import html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="분류 모델",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 분류 모델 만들기")
st.caption("로지스틱 회귀와 의사결정트리의 성능을 비교합니다.")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]


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
    "id",
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
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
# 입력 변수 선택
# --------------------------------------------------

st.header("1. 모델 입력 변수 선택")

st.write(
    "모델이 뇌졸중 여부를 예측할 때 사용할 속성을 선택하세요. "
    "처음에는 교재에서 제시한 네 가지 속성이 선택되어 있습니다."
)

selected_features = st.multiselect(
    "입력 변수",
    options=list(FEATURE_NAMES.keys()),
    default=DEFAULT_FEATURES,
    format_func=lambda x: FEATURE_NAMES[x]
)

if len(selected_features) < 2:
    st.warning("입력 변수는 두 개 이상 선택해야 합니다.")
    st.stop()

st.write(
    "선택한 변수: "
    + ", ".join(FEATURE_NAMES[x] for x in selected_features)
)


# --------------------------------------------------
# 데이터 전처리 및 훈련/테스트 분리
# --------------------------------------------------

data = df.copy()

for column in required_columns:
    if column != "id":
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

data = data.dropna(subset=["id", "stroke"])

data = data.sort_values("id").reset_index(drop=True)

# 10명씩 묶어 각 묶음의 앞 3명을 테스트 데이터로 지정
test_mask = np.arange(len(data)) % 10 < 3

train_data = data.loc[~test_mask].copy()
test_data = data.loc[test_mask].copy()

X_train = train_data[selected_features].copy()
X_test = test_data[selected_features].copy()

y_train = train_data["stroke"].astype(int)
y_test = test_data["stroke"].astype(int)

if len(train_data) == 0 or len(test_data) == 0:
    st.error("훈련 데이터 또는 테스트 데이터가 없습니다.")
    st.stop()

if len(train_data) != 3577 or len(test_data) != 1533:
    st.warning(
        f"현재 데이터 분할 결과: 훈련용 {len(train_data):,}명, "
        f"테스트용 {len(test_data):,}명입니다. "
        "교재의 데이터와 행 수가 다를 수 있습니다."
    )


# --------------------------------------------------
# 결측값 처리
# --------------------------------------------------

# BMI를 선택한 경우에만 훈련 데이터의 중앙값으로 결측값 처리
if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()

    X_train.loc[:, "bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test.loc[:, "bmi"] = X_test["bmi"].fillna(bmi_median)


# 나머지 선택 변수에 결측값이 있는지 확인
if X_train.isna().any().any() or X_test.isna().any().any():
    st.error(
        "선택한 입력 변수에 BMI 이외의 결측값이 있습니다. "
        "현재 교재의 처리 조건에서는 모델을 학습할 수 없습니다."
    )
    st.stop()


# --------------------------------------------------
# 데이터 크기 확인
# --------------------------------------------------

st.divider()
st.header("2. 데이터 분할 확인")

col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터", f"{len(data):,}명")
col2.metric("훈련 데이터", f"{len(train_data):,}명")
col3.metric("테스트 데이터", f"{len(test_data):,}명")

st.info(
    """
    데이터를 번호 순으로 정렬한 뒤 10명씩 묶었습니다.
    각 묶음의 앞 3명은 테스트용으로 남기고,
    나머지 7명만 모델 학습에 사용합니다.
    """
)


# --------------------------------------------------
# 3. 로지스틱 회귀 모델
# --------------------------------------------------

# 크기 맞추기는 훈련 데이터에 대해서만 학습
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

logistic_model.fit(X_train_scaled, y_train)

logistic_train_pred = logistic_model.predict(X_train_scaled)
logistic_test_pred = logistic_model.predict(X_test_scaled)

logistic_train_accuracy = (
    logistic_train_pred == y_train
).mean()

logistic_test_accuracy = (
    logistic_test_pred == y_test
).mean()


# --------------------------------------------------
# 4. 의사결정트리 모델
# --------------------------------------------------

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(X_train, y_train)

tree_train_pred = tree_model.predict(X_train)
tree_test_pred = tree_model.predict(X_test)

tree_train_accuracy = (
    tree_train_pred == y_train
).mean()

tree_test_accuracy = (
    tree_test_pred == y_test
).mean()


# --------------------------------------------------
# 5. 기준 모델
# --------------------------------------------------

# 훈련 데이터에서 더 많은 범주를 확인
majority_class = y_train.mode()[0]

baseline_train_pred = np.full(
    len(y_train),
    majority_class
)

baseline_test_pred = np.full(
    len(y_test),
    majority_class
)

baseline_train_accuracy = (
    baseline_train_pred == y_train
).mean()

baseline_test_accuracy = (
    baseline_test_pred == y_test
).mean()


# --------------------------------------------------
# 6. 정확도 카드
# --------------------------------------------------

st.divider()
st.header("3. 모델 정확도 비교")

st.write(
    "정확도는 전체 예측 가운데 실제 정답과 일치한 비율입니다. "
    "큰 숫자는 테스트 데이터의 정확도입니다."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "로지스틱 회귀",
        f"{logistic_test_accuracy * 100:.2f}%",
        help="확률로 답하는 모델의 테스트 정확도"
    )

    st.caption(
        f"훈련 데이터: {logistic_train_accuracy * 100:.2f}%"
    )
    st.caption(
        f"테스트 데이터: {logistic_test_accuracy * 100:.2f}%"
    )


with col2:
    st.metric(
        "의사결정트리",
        f"{tree_test_accuracy * 100:.2f}%",
        help="질문으로 답하는 모델의 테스트 정확도"
    )

    st.caption(
        f"훈련 데이터: {tree_train_accuracy * 100:.2f}%"
    )
    st.caption(
        f"테스트 데이터: {tree_test_accuracy * 100:.2f}%"
    )


with col3:
    st.metric(
        "기준 모델",
        f"{baseline_test_accuracy * 100:.2f}%",
        help="입력을 보지 않고 훈련 데이터의 다수 범주만 예측"
    )

    st.caption(
        f"훈련 데이터: {baseline_train_accuracy * 100:.2f}%"
    )
    st.caption(
        f"테스트 데이터: {baseline_test_accuracy * 100:.2f}%"
    )


accuracy_table = pd.DataFrame({
    "모델": [
        "로지스틱 회귀",
        "의사결정트리",
        "기준 모델"
    ],
    "훈련 데이터 정확도(%)": [
        logistic_train_accuracy * 100,
        tree_train_accuracy * 100,
        baseline_train_accuracy * 100
    ],
    "테스트 데이터 정확도(%)": [
        logistic_test_accuracy * 100,
        tree_test_accuracy * 100,
        baseline_test_accuracy * 100
    ]
})

accuracy_table.iloc[:, 1:] = accuracy_table.iloc[:, 1:].round(2)

st.dataframe(
    accuracy_table,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 7. 두 입력 변수 선택
# --------------------------------------------------

st.divider()
st.header("4. 두 모델의 분류 영역 비교")

st.write(
    "선택한 입력 변수 중 두 가지를 골라 산점도의 가로축과 세로축으로 사용합니다."
)

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축",
        selected_features,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x]
    )

with axis_col2:
    y_default_index = 1 if selected_features[1] != x_feature else 0

    y_feature = st.selectbox(
        "세로축",
        selected_features,
        index=y_default_index,
        format_func=lambda x: FEATURE_NAMES[x]
    )

if x_feature == y_feature:
    st.warning("가로축과 세로축은 서로 다른 변수를 선택해야 합니다.")
    st.stop()


# --------------------------------------------------
# 8. 나머지 변수는 테스트 데이터의 중앙값으로 고정
# --------------------------------------------------

other_features = [
    feature for feature in selected_features
    if feature not in [x_feature, y_feature]
]

fixed_values = {}

for feature in other_features:
    fixed_values[feature] = float(X_test[feature].median())

if other_features:
    fixed_text = ", ".join(
        f"{FEATURE_NAMES[feature]}: {value:.2f}"
        for feature, value in fixed_values.items()
    )

    st.info(
        f"가로축과 세로축이 아닌 변수는 테스트 데이터의 중앙값으로 "
        f"고정했습니다. ({fixed_text})"
    )
else:
    st.info("선택한 입력 변수가 두 개이므로 고정할 나머지 변수는 없습니다.")


# --------------------------------------------------
# 9. 분류 영역 계산
# --------------------------------------------------

x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())

y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())

x_padding = (x_max - x_min) * 0.05
y_padding = (y_max - y_min) * 0.05

if x_padding == 0:
    x_padding = 1

if y_padding == 0:
    y_padding = 1

x_range = np.linspace(
    x_min - x_padding,
    x_max + x_padding,
    120
)

y_range = np.linspace(
    y_min - y_padding,
    y_max + y_padding,
    120
)

grid_x, grid_y = np.meshgrid(x_range, y_range)

grid_data = pd.DataFrame(
    index=range(grid_x.size),
    columns=selected_features
)

for feature in selected_features:
    if feature == x_feature:
        grid_data[feature] = grid_x.ravel()

    elif feature == y_feature:
        grid_data[feature] = grid_y.ravel()

    else:
        grid_data[feature] = fixed_values[feature]

grid_data = grid_data.astype(float)


# 로지스틱 회귀의 분류 영역
grid_scaled = scaler.transform(grid_data)

logistic_grid_probability = logistic_model.predict_proba(
    grid_scaled
)[:, 1].reshape(grid_x.shape)

logistic_grid_prediction = (
    logistic_grid_probability > 0.5
).astype(int)


# 의사결정트리의 분류 영역
tree_grid_prediction = tree_model.predict(
    grid_data
).reshape(grid_x.shape)


# --------------------------------------------------
# 10. 산점도 데이터
# --------------------------------------------------

plot_data = test_data[
    [x_feature, y_feature, "stroke"]
].copy()

plot_data["실제 뇌졸중 여부"] = plot_data["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})


# --------------------------------------------------
# 11. 로지스틱 회귀 분류 영역
# --------------------------------------------------

st.subheader("로지스틱 회귀 — 확률로 답하는 모델")

fig_logistic = go.Figure()

fig_logistic.add_trace(
    go.Contour(
        x=x_range,
        y=y_range,
        z=logistic_grid_prediction,
        contours=dict(
            start=0,
            end=1,
            size=1,
            coloring="fill",
            showlines=False
        ),
        colorscale=[
            [0, "rgba(100, 160, 220, 0.16)"],
            [1, "rgba(240, 100, 100, 0.16)"]
        ],
        showscale=False,
        hoverinfo="skip",
        name="분류 영역"
    )
)

fig_logistic.add_trace(
    go.Contour(
        x=x_range,
        y=y_range,
        z=logistic_grid_probability,
        contours=dict(
            start=0.5,
            end=0.5,
            coloring="none",
            showlabels=True
        ),
        line=dict(
            color="black",
            width=2
        ),
        showscale=False,
        hoverinfo="skip",
        name="확률 0.5 경계"
    )
)

for label, color in [
    ("뇌졸중 없음", "#2878B5"),
    ("뇌졸중 있음", "#D62728")
]:
    subset = plot_data[
        plot_data["실제 뇌졸중 여부"] == label
    ]

    fig_logistic.add_trace(
        go.Scatter(
            x=subset[x_feature],
            y=subset[y_feature],
            mode="markers",
            name=label,
            marker=dict(
                color=color,
                size=6,
                opacity=0.7
            ),
            hovertemplate=(
                f"{FEATURE_NAMES[x_feature]}: %{{x:.2f}}<br>"
                f"{FEATURE_NAMES[y_feature]}: %{{y:.2f}}<br>"
                f"{label}<extra></extra>"
            )
        )
    )

fig_logistic.update_layout(
    title="로지스틱 회귀의 0.5 분류 기준",
    xaxis_title=FEATURE_NAMES[x_feature],
    yaxis_title=FEATURE_NAMES[y_feature],
    height=650
)

st.plotly_chart(
    fig_logistic,
    use_container_width=True
)

# 0.5 경계가 화면 범위 안에 있는지 확인
prob_min = logistic_grid_probability.min()
prob_max = logistic_grid_probability.max()

if prob_min > 0.5 or prob_max < 0.5:
    st.warning(
        "확률 0.5 경계가 현재 그림의 범위 안에 나타나지 않습니다. "
        "이 범위에서는 모든 지점이 같은 범주로 분류됩니다."
    )


# --------------------------------------------------
# 12. 의사결정트리 분류 영역
# --------------------------------------------------

st.subheader("의사결정트리 — 질문으로 답하는 모델")

fig_tree = go.Figure()

fig_tree.add_trace(
    go.Contour(
        x=x_range,
        y=y_range,
        z=tree_grid_prediction,
        contours=dict(
            start=0,
            end=1,
            size=1,
            coloring="fill",
            showlines=False
        ),
        colorscale=[
            [0, "rgba(100, 160, 220, 0.16)"],
            [1, "rgba(240, 100, 100, 0.16)"]
        ],
        showscale=False,
        hoverinfo="skip",
        name="트리 분류 영역"
    )
)

for label, color in [
    ("뇌졸중 없음", "#2878B5"),
    ("뇌졸중 있음", "#D62728")
]:
    subset = plot_data[
        plot_data["실제 뇌졸중 여부"] == label
    ]

    fig_tree.add_trace(
        go.Scatter(
            x=subset[x_feature],
            y=subset[y_feature],
            mode="markers",
            name=label,
            marker=dict(
                color=color,
                size=6,
                opacity=0.7
            ),
            hovertemplate=(
                f"{FEATURE_NAMES[x_feature]}: %{{x:.2f}}<br>"
                f"{FEATURE_NAMES[y_feature]}: %{{y:.2f}}<br>"
                f"{label}<extra></extra>"
            )
        )
    )

fig_tree.update_layout(
    title="의사결정트리의 분류 영역",
    xaxis_title=FEATURE_NAMES[x_feature],
    yaxis_title=FEATURE_NAMES[y_feature],
    height=650
)

st.plotly_chart(
    fig_tree,
    use_container_width=True
)


# --------------------------------------------------
# 13. 의사결정트리 구조 DOT 생성
# --------------------------------------------------

st.divider()
st.header("5. 의사결정트리의 질문 구조")

tree = tree_model.tree_
feature_names = selected_features

dot_lines = [
    "digraph Tree {",
    'graph [rankdir=TB, bgcolor="white"];',
    'node [shape=box, style="filled,rounded", fontname="Arial"];',
    'edge [fontname="Arial", fontsize=10];'
]

leaf_count = 0
negative_leaf_count = 0
used_features = set()


def build_tree_dot(node_id):
    global leaf_count, negative_leaf_count

    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    sample_count = int(tree.n_node_samples[node_id])

    # sklearn의 tree.value에 저장된 각 범주의 인원수
    class_counts = tree.value[node_id][0]

    negative_count = int(round(class_counts[0]))
    positive_count = int(round(class_counts[1]))

    positive_rate = (
        positive_count / sample_count * 100
        if sample_count > 0 else 0
    )

    is_leaf = left_child == right_child

    if is_leaf:
        leaf_count += 1

        predicted_class = int(np.argmax(class_counts))

        if predicted_class == 0:
            negative_leaf_count += 1
            fill_color = "#DCEBFA"
            answer = "뇌졸중 없음"
        else:
            fill_color = "#F8DADA"
            answer = "뇌졸중 있음"

        label = (
            f"최종 답: {answer}\\n"
            f"훈련 인원: {sample_count}명\\n"
            f"실제 뇌졸중: {positive_count}명\\n"
            f"실제 뇌졸중 비율: {positive_rate:.1f}%"
        )

        dot_lines.append(
            f'{node_id} [label="{label}", fillcolor="{fill_color}"];'
        )

        return

    feature_index = tree.feature[node_id]
    threshold = tree.threshold[node_id]

    feature_name = feature_names[feature_index]
    used_features.add(feature_name)

    label = (
        f"{FEATURE_NAMES[feature_name]} <= {threshold:.2f}?\\n"
        f"훈련 인원: {sample_count}명\\n"
        f"실제 뇌졸중: {positive_count}명\\n"
        f"실제 뇌졸중 비율: {positive_rate:.1f}%"
    )

    dot_lines.append(
        f'{node_id} [label="{label}", fillcolor="#FFF2CC"];'
    )

    dot_lines.append(
        f'{node_id} -> {left_child} [label="예"];'
    )

    dot_lines.append(
        f'{node_id} -> {right_child} [label="아니요"];'
    )

    build_tree_dot(left_child)
    build_tree_dot(right_child)


build_tree_dot(0)

dot_lines.append("}")

dot_string = "\n".join(dot_lines)

st.graphviz_chart(
    dot_string,
    use_container_width=True
)


# --------------------------------------------------
# 14. 트리 구조 요약
# --------------------------------------------------

col1, col2 = st.columns(2)

col1.metric(
    "전체 답변 마디 수",
    f"{leaf_count}개"
)

col2.metric(
    "뇌졸중 없음으로 답하는 마디 수",
    f"{negative_leaf_count}개"
)

if used_features:
    used_korean_names = [
        FEATURE_NAMES[feature]
        for feature in feature_names
        if feature in used_features
    ]

    st.write(
        "**실제 질문에 사용된 속성:** "
        + ", ".join(used_korean_names)
    )
else:
    st.write("실제 질문에 사용된 속성이 없습니다.")


# --------------------------------------------------
# 15. 마무리 안내
# --------------------------------------------------

st.divider()

st.info(
    """
    정확도가 높다고 해서 반드시 좋은 분류 모델이라고 할 수는 없습니다.

    이 데이터는 뇌졸중 경험자가 전체에서 차지하는 비율이 낮습니다.
    따라서 입력을 보지 않고 다수 범주만 예측하는 기준 모델도
    높은 정확도를 얻을 수 있습니다.

    다음 단계에서는 정확도뿐 아니라 실제 뇌졸중 경험자를
    얼마나 찾아냈는지도 함께 살펴보아야 합니다.

    이 모델은 교육용 실습이며 실제 진단이나 의료 판단에 사용할 수 없습니다.
    """
)

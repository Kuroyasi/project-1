
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, _tree


st.set_page_config(
    page_title="분류 모델",
    page_icon="🤖",
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


st.title("🤖 분류 모델")
st.write("머신러닝 모델을 학습하고 예측 결과를 시각화합니다.")

st.divider()


# 사용할 변수 선택
st.subheader("1. 학습 변수 선택")

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
    st.warning("시각화를 위해 최소 2개의 변수를 선택해야 합니다.")
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


# 로지스틱 회귀용 표준화
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


# 정확도
logistic_accuracy = (
    sum(logistic_pred == y_test) / len(y_test)
)

tree_accuracy = (
    sum(tree_pred == y_test) / len(y_test)
)

baseline_accuracy = (
    sum(
        prediction == actual
        for prediction, actual in zip(baseline_pred, y_test)
    ) / len(y_test)
)


# 모델 결과
st.divider()
st.subheader("2. 모델 성능 비교")

col1, col2, col3 = st.columns(3)

col1.metric(
    "로지스틱 회귀 정확도",
    f"{logistic_accuracy * 100:.2f}%"
)

col2.metric(
    "결정 트리 정확도",
    f"{tree_accuracy * 100:.2f}%"
)

col3.metric(
    "기준 모델 정확도",
    f"{baseline_accuracy * 100:.2f}%"
)

st.write(
    f"학습 데이터: {len(train_df):,}명 / "
    f"테스트 데이터: {len(test_df):,}명"
)


# 2차원 예측 영역
st.divider()
st.subheader("3. 모델 예측 영역 시각화")

x_feature = st.selectbox(
    "X축 변수",
    options=features,
    index=0,
    format_func=lambda x: feature_labels[x]
)

y_candidates = [
    feature for feature in features
    if feature != x_feature
]

y_feature = st.selectbox(
    "Y축 변수",
    options=y_candidates,
    index=0,
    format_func=lambda x: feature_labels[x]
)


# 나머지 변수는 테스트 데이터 중앙값으로 고정
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

x_min -= x_padding
x_max += x_padding
y_min -= y_padding
y_max += y_padding

grid_size = 45

x_values = [
    x_min + (x_max - x_min) * i / (grid_size - 1)
    for i in range(grid_size)
]

y_values = [
    y_min + (y_max - y_min) * i / (grid_size - 1)
    for i in range(grid_size)
]

grid_rows = []

for y_value in y_values:
    for x_value in x_values:
        row = {}

        for feature in features:
            row[feature] = float(X_train[feature].median())

        row[x_feature] = x_value
        row[y_feature] = y_value

        grid_rows.append(row)

grid_df = pd.DataFrame(grid_rows)[features]


# 두 모델 예측 영역 계산
grid_tree_pred = tree_model.predict(grid_df)

grid_scaled = scaler.transform(grid_df)
grid_logistic_prob = logistic_model.predict_proba(grid_scaled)

positive_index = list(logistic_model.classes_).index(1)

grid_logistic_prob = grid_logistic_prob[
    :, positive_index
]

tree_z = [
    grid_tree_pred[i * grid_size:(i + 1) * grid_size]
    for i in range(grid_size)
]

logistic_z = [
    grid_logistic_prob[i * grid_size:(i + 1) * grid_size]
    for i in range(grid_size)
]


# 그래프 생성
fig = go.Figure()

fig.add_trace(
    go.Contour(
        x=x_values,
        y=y_values,
        z=tree_z,
        name="결정 트리 예측 영역",
        colorscale=[
            [0, "#DCEBFA"],
            [1, "#F8DCDC"]
        ],
        opacity=0.45,
        showscale=False,
        contours={
            "start": 0,
            "end": 1,
            "size": 1,
            "coloring": "heatmap"
        },
        hoverinfo="skip"
    )
)

fig.add_trace(
    go.Contour(
        x=x_values,
        y=y_values,
        z=logistic_z,
        name="로지스틱 회귀 경계",
        showscale=False,
        contours={
            "start": 0.5,
            "end": 0.5,
            "size": 1,
            "coloring": "none",
            "showlines": True
        },
        line={
            "color": "black",
            "width": 2
        },
        hoverinfo="skip"
    )
)

for label, color, name in [
    (0, "#2878B5", "실제 정상"),
    (1, "#D9534F", "실제 뇌졸중")
]:
    selected = X_test[y_test == label]

    fig.add_trace(
        go.Scatter(
            x=selected[x_feature],
            y=selected[y_feature],
            mode="markers",
            name=name,
            marker={
                "color": color,
                "size": 7,
                "opacity": 0.75
            }
        )
    )


fig.update_layout(
    title="결정 트리 예측 영역과 로지스틱 회귀 결정 경계",
    xaxis_title=feature_labels[x_feature],
    yaxis_title=feature_labels[y_feature],
    height=600
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "다른 선택 변수들은 학습 데이터의 중앙값으로 고정했습니다. "
    "색으로 표시된 결정 트리 영역은 해당 조건에서의 예측 클래스이며, "
    "검은 선은 로지스틱 회귀의 예측 확률 0.5 경계입니다."
)


# 결정 트리 구조
st.divider()
st.subheader("4. 결정 트리 구조")

tree_structure = tree_model.tree_
tree_classes = list(tree_model.classes_)
positive_index = tree_classes.index(1)

feature_names = [
    feature_labels[feature]
    for feature in features
]

leaf_count = 0
negative_leaf_count = 0


def build_tree_graph(node_id=0):
    global leaf_count, negative_leaf_count

    node = tree_structure
    samples = node.n_node_samples[node_id]
    values = node.value[node_id][0]

    negative_count = int(values[tree_classes.index(0)])
    positive_count = int(values[positive_index])

    positive_rate = (
        positive_count / samples * 100
        if samples > 0 else 0
    )

    is_leaf = (
        node.children_left[node_id] == _tree.TREE_LEAF
    )

    if is_leaf:
        leaf_count += 1

        prediction = (
            "뇌졸중" if positive_count > negative_count
            else "정상"
        )

        if prediction == "정상":
            negative_leaf_count += 1

        color = (
            "#F8DCDC" if prediction == "뇌졸중"
            else "#DCEBFA"
        )

        return (
            f'{node_id} [label="예측: {prediction}\\n'
            f'데이터: {samples}명\\n'
            f'정상: {negative_count}명\\n'
            f'뇌졸중: {positive_count}명\\n'
            f'뇌졸중 비율: {positive_rate:.1f}%", '
            f'fillcolor="{color}"];\n'
        )

    feature_index = node.feature[node_id]
    feature_name = feature_names[feature_index]
    threshold = node.threshold[node_id]

    label = (
        f"{feature_name} <= {threshold:.2f}\\n"
        f"데이터: {samples}명\\n"
        f"정상: {negative_count}명\\n"
        f"뇌졸중: {positive_count}명\\n"
        f"뇌졸중 비율: {positive_rate:.1f}%"
    )

    result = (
        f'{node_id} [label="{label}", fillcolor="#EEEEEE"];\n'
    )

    left_id = node.children_left[node_id]
    right_id = node.children_right[node_id]

    result += f'{node_id} -> {left_id} [label="예"];\n'
    result += f'{node_id} -> {right_id} [label="아니오"];\n'

    result += build_tree_graph(left_id)
    result += build_tree_graph(right_id)

    return result


dot_graph = """
digraph DecisionTree {
    graph [
        rankdir=TB,
        bgcolor="transparent",
        nodesep=0.35,
        ranksep=0.55
    ];

    node [
        shape=box,
        style="rounded,filled",
        fontname="Malgun Gothic",
        fontsize=10,
        color="#777777"
    ];

    edge [
        fontname="Malgun Gothic",
        fontsize=9
    ];
"""

dot_graph += build_tree_graph()
dot_graph += "}"

st.graphviz_chart(
    dot_graph,
    use_container_width=True
)


# 트리 정보
st.subheader("5. 결정 트리 분석")

used_features = sorted(set(
    tree_model.tree_.feature[
        tree_model.tree_.feature >= 0
    ]
))

used_feature_names = [
    feature_names[index]
    for index in used_features
]

col1, col2, col3 = st.columns(3)

col1.metric(
    "전체 리프 노드",
    f"{leaf_count}개"
)

col2.metric(
    "정상 예측 리프 노드",
    f"{negative_leaf_count}개"
)

col3.metric(
    "실제로 사용된 변수",
    f"{len(used_feature_names)}개"
)

st.write(
    "결정 트리에 사용된 변수:",
    ", ".join(used_feature_names)
    if used_feature_names else "없음"
)


st.divider()

st.warning(
    "이 모델은 교육용 데이터 분석 실습입니다. "
    "실제 뇌졸중 진단이나 의료적 판단에 사용할 수 없습니다."
)

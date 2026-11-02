import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, _tree

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
KOREAN = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
FEATURES = list(KOREAN)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def tree_to_dot(model, features):
    tree = model.tree_
    lines = [
        "digraph Tree {",
        'graph [rankdir=TB, nodesep=0.35, ranksep=0.55];',
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10];',
        'edge [fontname="Arial", fontsize=9];',
    ]
    leaf_total = 0
    negative_leaves = 0
    asked = set()

    def visit(node):
        nonlocal leaf_total, negative_leaves
        counts = tree.value[node][0]
        n = int(tree.n_node_samples[node])
        positive = int(counts[1]) if len(counts) > 1 else 0
        rate = positive / counts.sum() * 100 if counts.sum() else 0
        left, right = tree.children_left[node], tree.children_right[node]

        if left == _tree.TREE_LEAF:
            leaf_total += 1
            prediction = int(np.argmax(counts))
            answer = "뇌졸중" if prediction == 1 else "아님"
            if prediction == 0:
                negative_leaves += 1
            color = "#f4caca" if prediction == 1 else "#cce8d5"
            label = f"답: {answer}\\n훈련용 사람: {n}명\\n실제 뇌졸중: {positive}명\\n뇌졸중 비율: {rate:.1f}%"
            lines.append(f'n{node} [label="{label}", fillcolor="{color}"];')
            return

        feature = features[tree.feature[node]]
        asked.add(feature)
        threshold = tree.threshold[node]
        label = f"{KOREAN[feature]} ≤ {threshold:.2f}?\\n훈련용 사람: {n}명\\n실제 뇌졸중: {positive}명\\n뇌졸중 비율: {rate:.1f}%"
        lines.append(f'n{node} [label="{label}", fillcolor="#e8eef8"];')
        lines.append(f'n{node} -> n{left} [label="예"];')
        lines.append(f'n{node} -> n{right} [label="아니요"];')
        visit(left)
        visit(right)

    visit(0)
    lines.append("}")
    return "\\n".join(lines), leaf_total, negative_leaves, asked


st.title("🌳 분류 모델")
st.write("선택한 속성으로 뇌졸중 여부를 분류하는 두 모델을 학습하고, 테스트 데이터에서 성능과 판단 구조를 비교합니다.")

try:
    raw = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요.")
    st.caption(f"오류 정보: {error}")
    st.stop()

required = ["id", "stroke"] + FEATURES
missing = [col for col in required if col not in raw.columns]
if missing:
    st.error("데이터에 필요한 열이 없습니다: " + ", ".join(missing))
    st.stop()

st.subheader("1. 입력 속성 선택")
selected = st.multiselect(
    "모델 학습에 사용할 속성을 선택하세요.",
    options=FEATURES,
    default=["age", "avg_glucose_level", "hypertension", "heart_disease"],
    format_func=lambda value: KOREAN[value],
)
if len(selected) < 2:
    st.warning("입력 속성을 두 개 이상 선택해야 모델을 학습할 수 있습니다.")
    st.stop()

# id 오름차순 정렬 후, 각 10명 묶음의 앞 3명을 테스트 데이터로 고정한다.
ordered = raw.sort_values("id", kind="mergesort").reset_index(drop=True).copy()
ordered["_position"] = np.arange(len(ordered))
test_mask = ordered["_position"] % 10 < 3
train_raw = ordered.loc[~test_mask].copy()
test_raw = ordered.loc[test_mask].copy()
y_train = train_raw["stroke"].astype(int)
y_test = test_raw["stroke"].astype(int)

if len(test_raw) != 1533:
    st.warning(f"현재 테스트 데이터는 {len(test_raw):,}명입니다. 기대한 1,533명과 다르면 원본 CSV 행 수를 확인하세요.")

x_train = train_raw[selected].copy()
x_test = test_raw[selected].copy()

if "bmi" in selected:
    bmi_median = x_train["bmi"].median()
    x_train["bmi"] = x_train["bmi"].fillna(bmi_median)
    x_test["bmi"] = x_test["bmi"].fillna(bmi_median)
    st.info(f"체질량지수의 빈 값은 훈련 데이터 중앙값 {bmi_median:.2f}으로 채웠습니다.")

if x_train.isna().any().any() or x_test.isna().any().any():
    st.error("선택한 속성에 결측값이 남아 있습니다. BMI 이외의 결측 여부를 확인해 주세요.")
    st.stop()

# 스케일러는 훈련 데이터에만 fit한다.
logistic = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
])
logistic.fit(x_train, y_train)

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
)
tree_model.fit(x_train, y_train)


def scores(model):
    return (
        accuracy_score(y_train, model.predict(x_train)),
        accuracy_score(y_test, model.predict(x_test)),
    )


log_train, log_test = scores(logistic)
tree_train, tree_test = scores(tree_model)
majority = int(y_train.value_counts().idxmax())
base_train = accuracy_score(y_train, np.full(len(y_train), majority))
base_test = accuracy_score(y_test, np.full(len(y_test), majority))

st.divider()
st.subheader("2. 모델 정확도 비교")
st.caption(f"훈련 데이터 {len(x_train):,}명 · 테스트 데이터 {len(x_test):,}명 · 번호순 10명 중 앞의 3명을 테스트용으로 고정")
cards = st.columns(3)
results = [
    ("로지스틱 회귀(확률로 답하는 모델)", log_train, log_test),
    ("의사결정트리(질문으로 답하는 모델)", tree_train, tree_test),
    ("다수 클래스 기준 모델", base_train, base_test),
]
for col, (name, train_score, test_score) in zip(cards, results):
    with col:
        st.markdown(f"**{name}**")
        st.metric("테스트 정확도", f"{test_score * 100:.2f}%")
        st.caption(f"훈련 데이터 정확도: {train_score * 100:.2f}%  \n테스트 데이터 정확도: {test_score * 100:.2f}%")
st.caption("다수 클래스 기준 모델은 입력 속성을 보지 않고 훈련 데이터에서 더 많은 클래스로만 답합니다.")

st.divider()
st.subheader("3. 테스트 데이터와 두 모델의 결정 영역")
left, right = st.columns(2)
with left:
    x_feature = st.selectbox("가로축", selected, format_func=lambda value: KOREAN[value])
with right:
    y_choices = [f for f in selected if f != x_feature]
    y_feature = st.selectbox("세로축", y_choices, format_func=lambda value: KOREAN[value])

fixed = {f: float(x_test[f].median()) for f in selected if f not in (x_feature, y_feature)}
if fixed:
    st.write("두 축 이외의 속성은 테스트 데이터 중앙값에 고정했습니다: " +
             ", ".join(f"{KOREAN[f]} {v:.2f}" for f, v in fixed.items()))
else:
    st.write("두 축 이외에 고정할 속성은 없습니다.")

x_min, x_max = float(x_test[x_feature].min()), float(x_test[x_feature].max())
y_min, y_max = float(x_test[y_feature].min()), float(x_test[y_feature].max())
x_pad = (x_max - x_min) * 0.05 or 1.0
y_pad = (y_max - y_min) * 0.05 or 1.0
x_values = np.linspace(x_min - x_pad, x_max + x_pad, 180)
y_values = np.linspace(y_min - y_pad, y_max + y_pad, 180)
xx, yy = np.meshgrid(x_values, y_values)

grid = pd.DataFrame({
    f: np.full(xx.size, fixed[f]) if f in fixed
    else xx.ravel() if f == x_feature else yy.ravel()
    for f in selected
})
probability = logistic.predict_proba(grid)[:, 1].reshape(xx.shape)
tree_area = tree_model.predict(grid).reshape(xx.shape)

fig = go.Figure()
fig.add_trace(go.Contour(
    x=x_values, y=y_values, z=tree_area,
    contours=dict(start=0, end=1, size=1, coloring="fill", showlines=False),
    colorscale=[
        [0.0, "rgba(70,150,110,0.13)"], [0.499, "rgba(70,150,110,0.13)"],
        [0.5, "rgba(220,90,90,0.13)"], [1.0, "rgba(220,90,90,0.13)"],
    ],
    showscale=False, hoverinfo="skip", name="의사결정트리 영역",
))
fig.add_trace(go.Contour(
    x=x_values, y=y_values, z=probability,
    contours=dict(start=0.5, end=0.5, size=1, coloring="lines", showlabels=False),
    line=dict(color="#222222", width=3),
    showscale=False, hoverinfo="skip", name="로지스틱 회귀 0.5 경계",
))
for target, color, label in [
    (0, "#2878B5", "실제 뇌졸중 없음"),
    (1, "#D94A4A", "실제 뇌졸중 있음"),
]:
    mask = y_test.eq(target).to_numpy()
    fig.add_trace(go.Scatter(
        x=x_test.loc[mask, x_feature],
        y=x_test.loc[mask, y_feature],
        mode="markers", name=label,
        marker=dict(color=color, size=7, opacity=0.65),
        hovertemplate=f"{KOREAN[x_feature]}: %{{x:.2f}}<br>{KOREAN[y_feature]}: %{{y:.2f}}<extra></extra>",
    ))
fig.update_layout(
    title="테스트 데이터와 모델의 결정 영역",
    xaxis_title=KOREAN[x_feature], yaxis_title=KOREAN[y_feature],
    legend_title="실제 정답 및 모델 영역", height=650,
)
st.plotly_chart(fig, use_container_width=True)

if probability.min() <= 0.5 <= probability.max():
    st.write("로지스틱 회귀의 확률 0.5 경계선이 현재 그림 범위 안에 있습니다.")
else:
    st.write("로지스틱 회귀의 확률 0.5 경계선은 현재 그림 범위 밖에 있습니다.")
st.caption("배경의 옅은 색은 의사결정트리 예측 영역, 검은 선은 로지스틱 회귀의 0.5 경계이며 점 색은 실제 테스트 정답입니다.")

st.divider()
st.subheader("4. 의사결정트리의 질문 구조")
dot, leaves, negative_leaves, asked_features = tree_to_dot(tree_model, selected)
st.graphviz_chart(dot, use_container_width=True)
st.write(f"답을 내는 마디는 총 {leaves}칸이며, 그중 {negative_leaves}칸이 '뇌졸중 아님'으로 답합니다.")
if asked_features:
    asked_korean = [KOREAN[f] for f in selected if f in asked_features]
    st.write("이 나무가 실제로 질문한 속성: " + ", ".join(asked_korean))
else:
    st.write("이 나무는 속성을 질문하지 않고 바로 답합니다.")
st.caption("각 마디의 사람 수와 실제 뇌졸중 수는 해당 마디에 도달한 훈련 데이터 기준입니다. '예'는 질문 조건을 만족하고, '아니요'는 만족하지 않는 경우입니다.")
st.warning("교육용 분류 실습입니다. 정확도는 의료 진단 성능을 보증하지 않으며 실제 진단이나 치료 결정에 사용하면 안 됩니다.")

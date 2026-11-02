
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="뇌졸중 예측 실습실 | 분류 모델",
    page_icon="🤖",
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


# ==========================================
# 3. 페이지 제목
# ==========================================

st.title("🤖 분류 모델")

st.markdown(
    """
    선택한 건강 관련 변수를 활용하여 뇌졸중 여부를 예측하는
    머신러닝 분류 모델을 학습합니다.

    로지스틱 회귀와 의사결정트리의 예측 성능 및
    분류 과정을 비교해 봅니다.
    """
)

st.divider()


# ==========================================
# 4. 입력 속성 선택
# ==========================================

st.subheader("1. 입력 속성 선택")

st.markdown(
    """
    모델 학습에 사용할 속성을 선택하세요.

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
# 5. 데이터 정렬 및 학습/테스트 분리
# ==========================================

st.subheader("2. 학습 데이터와 테스트 데이터 분리")

st.markdown(
    """
    데이터를 번호 순으로 정렬한 다음 10명씩 묶습니다.

    각 묶음의 앞 세 명은 테스트 데이터로 고정하고,
    나머지 일곱 명만 학습 데이터로 사용합니다.
    """
)


# 번호 순 정렬
df = df.sort_values(
    "id"
).reset_index(drop=True)


# 10명 단위로 앞의 3명 선택
row_positions = np.arange(len(df))

test_mask = (row_positions % 10) < 3

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()


# 테스트 인원 확인
expected_test_count = 1533

if len(test_df) != expected_test_count:
    st.warning(
        f"현재 테스트 데이터는 {len(test_df):,}명입니다. "
        f"요청한 1,533명과 다른 경우 원본 데이터의 전체 인원과 "
        "정렬 상태를 확인하세요."
    )


# 입력 변수와 정답 분리
X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)


# 데이터 분할 통계
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "전체 데이터",
        f"{len(df):,}명"
    )

with col2:
    st.metric(
        "학습 데이터",
        f"{len(train_df):,}명"
    )

with col3:
    st.metric(
        "테스트 데이터",
        f"{len(test_df):,}명"
    )


st.caption(
    "테스트 데이터는 각 10명 묶음의 앞 3명으로 고정됩니다."
)


# ==========================================
# 6. 결측값 처리
# ==========================================

st.subheader("3. 결측값 처리")

bmi_median = None

if "bmi" in selected_features:

    # 학습 데이터의 BMI 중앙값만 사용
    bmi_median = X_train["bmi"].median()

    # 동일한 학습 데이터 중앙값으로 양쪽 데이터 처리
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

    st.success(
        f"체질량지수의 빈 값을 학습 데이터의 중앙값 "
        f"{bmi_median:.2f}으로 채웠습니다."
    )

else:
    st.info(
        "체질량지수를 선택하지 않았으므로 BMI 결측값 처리를 진행하지 않았습니다."
    )


# ==========================================
# 7. 크기 맞추기 (표준화)
# ==========================================

st.subheader("4. 데이터 크기 맞추기")

st.markdown(
    """
    로지스틱 회귀는 변수의 크기에 영향을 받을 수 있으므로
    표준화를 진행합니다.

    평균과 표준편차는 학습 데이터에서만 계산하며,
    테스트 데이터에는 학습 데이터의 기준을 그대로 적용합니다.
    """
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


st.success(
    "표준화 완료: 학습 데이터의 평균과 표준편차를 기준으로 변환했습니다."
)


st.divider()


# ==========================================
# 8. 로지스틱 회귀 모델
# ==========================================

st.subheader(
    "5. 로지스틱 회귀 (확률로 답하는 모델)"
)

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE
)

logistic_model.fit(
    X_train_scaled,
    y_train
)


# 학습 / 테스트 예측
logistic_train_pred = logistic_model.predict(
    X_train_scaled
)

logistic_test_pred = logistic_model.predict(
    X_test_scaled
)


logistic_train_accuracy = accuracy_score(
    y_train,
    logistic_train_pred
)

logistic_test_accuracy = accuracy_score(
    y_test,
    logistic_test_pred
)


# 뇌졸중일 확률
positive_index = list(logistic_model.classes_).index(1)

logistic_test_probability = (
    logistic_model.predict_proba(X_test_scaled)[:, positive_index]
)


st.write(
    "로지스틱 회귀는 입력 속성을 바탕으로 뇌졸중일 확률을 계산하고, "
    "그 확률이 0.5 이상이면 뇌졸중(양성)으로 분류합니다."
)


# ==========================================
# 9. 의사결정트리 모델
# ==========================================

st.subheader(
    "6. 의사결정트리 (질문으로 답하는 모델)"
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


tree_train_accuracy = accuracy_score(
    y_train,
    tree_train_pred
)

tree_test_accuracy = accuracy_score(
    y_test,
    tree_test_pred
)


st.write(
    """
    의사결정트리는 질문을 통해 데이터를 나누는 방식입니다.

    이번 실습에서는 질문을 최대 세 번까지만 허용하고,
    마지막 마디에 사람이 다섯 명 미만으로 남으면 더 나누지 않습니다.
    """
)


st.divider()


# ==========================================
# 10. 다수 클래스 기준 모델
# ==========================================

st.subheader("7. 기준 모델")

st.markdown(
    """
    입력 속성을 하나도 사용하지 않고,
    학습 데이터에서 더 많은 쪽으로만 답하는 모델입니다.

    이 모델의 성능을 기준으로 머신러닝 모델이
    실제로 얼마나 유용한지 비교할 수 있습니다.
    """
)


majority_class = int(
    y_train.value_counts().idxmax()
)

baseline_train_pred = np.full(
    len(y_train),
    majority_class
)

baseline_test_pred = np.full(
    len(y_test),
    majority_class
)


baseline_train_accuracy = accuracy_score(
    y_train,
    baseline_train_pred
)

baseline_test_accuracy = accuracy_score(
    y_test,
    baseline_test_pred
)


majority_name = (
    "뇌졸중 있음 (1)"
    if majority_class == 1
    else "뇌졸중 아님 (0)"
)

st.info(
    f"기준 모델의 답: {majority_name}"
)


# ==========================================
# 11. 정확도 비교 카드
# ==========================================

st.subheader("8. 모델별 정확도 비교")

st.markdown(
    """
    테스트 데이터의 정확도를 크게 표시하고,
    학습 데이터와 테스트 데이터의 정확도를 함께 비교합니다.
    """
)


def accuracy_card(
    title,
    train_accuracy,
    test_accuracy
):
    with st.container(border=True):

        st.markdown(f"**{title}**")

        st.metric(
            label="테스트 데이터 정확도",
            value=f"{test_accuracy * 100:.2f}%"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.caption("학습 데이터")
            st.write(f"**{train_accuracy * 100:.2f}%**")

        with col2:
            st.caption("테스트 데이터")
            st.write(f"**{test_accuracy * 100:.2f}%**")


col1, col2, col3 = st.columns(3)

with col1:
    accuracy_card(
        "로지스틱 회귀",
        logistic_train_accuracy,
        logistic_test_accuracy
    )

with col2:
    accuracy_card(
        "의사결정트리",
        tree_train_accuracy,
        tree_test_accuracy
    )

with col3:
    accuracy_card(
        "기준 모델",
        baseline_train_accuracy,
        baseline_test_accuracy
    )


st.divider()


# ==========================================
# 12. 2차원 분류 결과 시각화
# ==========================================

st.subheader("9. 분류 영역 시각화")

st.markdown(
    """
    선택한 속성 가운데 두 가지를 가로축과 세로축으로 지정합니다.

    테스트 데이터를 실제 뇌졸중 여부에 따라 색으로 구분하고,
    로지스틱 회귀의 0.5 확률 경계선과
    의사결정트리의 분류 영역을 함께 표시합니다.
    """
)


# 축 선택
axis_col1, axis_col2 = st.columns(2)

with axis_col1:

    x_feature = st.selectbox(
        "가로축",
        options=selected_features,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x]
    )

with axis_col2:

    y_options = [
        feature
        for feature in selected_features
        if feature != x_feature
    ]

    y_feature = st.selectbox(
        "세로축",
        options=y_options,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x]
    )


# 축 이름
x_name = FEATURE_NAMES[x_feature]
y_name = FEATURE_NAMES[y_feature]


# ------------------------------------------
# 12-1. 나머지 속성 고정
# ------------------------------------------

# 테스트 데이터 중앙값 계산
test_medians = X_test.median()

fixed_features = [
    feature
    for feature in selected_features
    if feature not in [x_feature, y_feature]
]


st.markdown("**나머지 속성의 고정값**")

if fixed_features:

    fixed_text = []

    for feature in fixed_features:

        fixed_value = test_medians[feature]

        fixed_text.append(
            f"{FEATURE_NAMES[feature]}: {fixed_value:.2f}"
        )

    st.write(", ".join(fixed_text))

else:
    st.write(
        "선택한 속성이 두 개이므로 고정할 나머지 속성이 없습니다."
    )


# ------------------------------------------
# 12-2. 그래프 범위 설정
# ------------------------------------------

x_min = X_test[x_feature].min()
x_max = X_test[x_feature].max()

y_min = X_test[y_feature].min()
y_max = X_test[y_feature].max()

x_padding = (x_max - x_min) * 0.08
y_padding = (y_max - y_min) * 0.08

if x_padding == 0:
    x_padding = 1

if y_padding == 0:
    y_padding = 1

x_range = (
    x_min - x_padding,
    x_max + x_padding
)

y_range = (
    y_min - y_padding,
    y_max + y_padding
)


# ------------------------------------------
# 12-3. 격자 데이터 생성
# ------------------------------------------

grid_size = 120

x_values = np.linspace(
    x_range[0],
    x_range[1],
    grid_size
)

y_values = np.linspace(
    y_range[0],
    y_range[1],
    grid_size
)

xx, yy = np.meshgrid(
    x_values,
    y_values
)


# 나머지 속성은 테스트 데이터 중앙값으로 고정
grid_data = pd.DataFrame(
    np.tile(
        test_medians.values,
        (grid_size * grid_size, 1)
    ),
    columns=selected_features
)

grid_data[x_feature] = xx.ravel()
grid_data[y_feature] = yy.ravel()


# ------------------------------------------
# 12-4. 두 모델의 분류 영역 계산
# ------------------------------------------

# 의사결정트리 영역
tree_grid_pred = tree_model.predict(
    grid_data
).reshape(grid_size, grid_size)


# 로지스틱 회귀 확률 영역
grid_scaled = scaler.transform(grid_data)

logistic_grid_probability = (
    logistic_model.predict_proba(grid_scaled)[:, positive_index]
).reshape(grid_size, grid_size)


# ------------------------------------------
# 12-5. Plotly 그림 생성
# ------------------------------------------

fig = go.Figure()


# 의사결정트리 분류 영역
fig.add_trace(
    go.Heatmap(
        x=x_values,
        y=y_values,
        z=tree_grid_pred,
        colorscale=[
            [0.0, "rgba(76, 120, 168, 0.16)"],
            [0.499, "rgba(76, 120, 168, 0.16)"],
            [0.5, "rgba(228, 87, 86, 0.16)"],
            [1.0, "rgba(228, 87, 86, 0.16)"]
        ],
        showscale=False,
        hoverinfo="skip",
        name="의사결정트리 분류 영역"
    )
)


# 로지스틱 회귀 0.5 경계선
probability_min = logistic_grid_probability.min()
probability_max = logistic_grid_probability.max()

boundary_exists = (
    probability_min <= 0.5 <= probability_max
)

if boundary_exists:

    fig.add_trace(
        go.Contour(
            x=x_values,
            y=y_values,
            z=logistic_grid_probability,
            contours=dict(
                start=0.5,
                end=0.5,
                size=1,
                coloring="none",
                showlabels=False
            ),
            line=dict(
                color="#7B2CBF",
                width=3
            ),
            showscale=False,
            hoverinfo="skip",
            name="로지스틱 회귀 0.5 경계선"
        )
    )


# 범례용 영역 표시
fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            size=10,
            color="rgba(76, 120, 168, 0.45)"
        ),
        name="트리: 뇌졸중 아님 영역"
    )
)

fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            size=10,
            color="rgba(228, 87, 86, 0.45)"
        ),
        name="트리: 뇌졸중 영역"
    )
)

fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(
            color="#7B2CBF",
            width=3
        ),
        name="로지스틱 회귀 0.5 경계선"
    )
)


# ------------------------------------------
# 12-6. 실제 테스트 데이터 점 표시
# ------------------------------------------

for label, color, name in [
    (0, "#2878B5", "실제 뇌졸중 아님"),
    (1, "#D62728", "실제 뇌졸중")
]:

    mask = y_test.values == label

    fig.add_trace(
        go.Scatter(
            x=X_test.loc[mask, x_feature],
            y=X_test.loc[mask, y_feature],
            mode="markers",
            marker=dict(
                size=7,
                color=color,
                opacity=0.75,
                line=dict(
                    width=0.5,
                    color="white"
                )
            ),
            name=name,
            hovertemplate=(
                f"{x_name}: %{{x:.2f}}<br>"
                f"{y_name}: %{{y:.2f}}"
                "<extra></extra>"
            )
        )
    )


fig.update_layout(
    title="테스트 데이터와 두 분류 모델의 영역",
    xaxis_title=x_name,
    yaxis_title=y_name,
    template="plotly_white",
    height=650,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ------------------------------------------
# 12-7. 경계선 존재 여부 안내
# ------------------------------------------

if boundary_exists:

    st.success(
        "현재 그림의 범위 안에 로지스틱 회귀의 "
        "0.5 확률 경계선이 존재합니다."
    )

else:

    st.warning(
        "현재 그림의 범위 안에는 로지스틱 회귀의 "
        "0.5 확률 경계선이 없습니다. "
        "경계선이 그림의 범위 밖에 있거나 형성되지 않았습니다."
    )


st.caption(
    "주의: 이 그림은 선택한 두 속성만 변화시키고 "
    "나머지 속성은 테스트 데이터의 중앙값으로 고정하여 계산한 "
    "2차원 단면입니다."
)


st.divider()


# ==========================================
# 13. 의사결정트리 가지 그림
# ==========================================

st.subheader("10. 의사결정트리 구조")

st.markdown(
    """
    의사결정트리가 학습 데이터에 던진 질문과
    각 질문에 도달한 사람 수를 확인합니다.

    각 마디에는 전체 인원, 실제 뇌졸중 인원, 뇌졸중 비율이 표시됩니다.
    """
)


tree_structure = tree_model.tree_

dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, nodesep=0.5, ranksep=0.7];',
    'node [shape=box, style="rounded,filled", fontname="Malgun Gothic", fontsize=11];',
    'edge [fontname="Malgun Gothic", fontsize=10];'
]


used_features = set()
leaf_count = 0
negative_leaf_count = 0


# ------------------------------------------
# 13-1. DOT 문자열 생성
# ------------------------------------------

def build_tree_dot(node_id):

    global leaf_count
    global negative_leaf_count

    samples = int(
        tree_structure.n_node_samples[node_id]
    )

    class_values = tree_structure.value[node_id][0]

    total_value = class_values.sum()

    positive_rate = (
        class_values[positive_index] / total_value
        if total_value > 0 else 0
    )

    positive_count = round(
        positive_rate * samples
    )

    predicted_class = int(
        tree_model.classes_[
            np.argmax(class_values)
        ]
    )

    rate_percent = positive_rate * 100

    is_leaf = (
        tree_structure.children_left[node_id]
        == tree_structure.children_right[node_id]
    )


    # --------------------------------------
    # 마지막 마디
    # --------------------------------------

    if is_leaf:

        leaf_count += 1

        if predicted_class == 0:
            negative_leaf_count += 1

        answer = (
            "뇌졸중 있음"
            if predicted_class == 1
            else "뇌졸중 아님"
        )

        fill_color = (
            "#F7CACA"
            if predicted_class == 1
            else "#C9DDF2"
        )

        label = (
            f"마디 {node_id}\\n"
            f"답: {answer}\\n"
            f"전체: {samples}명\\n"
            f"실제 뇌졸중: {positive_count}명\\n"
            f"뇌졸중 비율: {rate_percent:.1f}%"
        )

        dot_lines.append(
            f'{node_id} [label="{label}", fillcolor="{fill_color}"];'
        )

        return


    # --------------------------------------
    # 질문이 있는 마디
    # --------------------------------------

    feature_index = tree_structure.feature[node_id]

    feature_name = selected_features[feature_index]

    used_features.add(feature_name)

    threshold = tree_structure.threshold[node_id]

    threshold_text = (
        f"{threshold:.2f}"
    )

    question = (
        f"{FEATURE_NAMES[feature_name]} "
        f"≤ {threshold_text}?"
    )

    label = (
        f"마디 {node_id}\\n"
        f"{question}\\n"
        f"전체: {samples}명\\n"
        f"실제 뇌졸중: {positive_count}명\\n"
        f"뇌졸중 비율: {rate_percent:.1f}%"
    )

    dot_lines.append(
        f'{node_id} [label="{label}", fillcolor="#FFF0C2"];'
    )


    left_child = tree_structure.children_left[node_id]
    right_child = tree_structure.children_right[node_id]


    # 자식 마디 생성
    build_tree_dot(left_child)
    build_tree_dot(right_child)


    # 가지 연결
    dot_lines.append(
        f'{node_id} -> {left_child} [label="예"];'
    )

    dot_lines.append(
        f'{node_id} -> {right_child} [label="아니요"];'
    )


build_tree_dot(0)

dot_lines.append("}")

tree_dot = "\n".join(dot_lines)


# ------------------------------------------
# 13-2. 의사결정트리 출력
# ------------------------------------------

st.graphviz_chart(
    tree_dot,
    use_container_width=True
)


# ------------------------------------------
# 13-3. 의사결정트리 분석
# ------------------------------------------

st.markdown("**의사결정트리 분석 결과**")

used_feature_names = [
    FEATURE_NAMES[feature]
    for feature in selected_features
    if feature in used_features
]

unused_feature_names = [
    FEATURE_NAMES[feature]
    for feature in selected_features
    if feature not in used_features
]


col1, col2 = st.columns(2)

with col1:

    st.metric(
        "답을 내는 마디 수",
        f"{leaf_count}개"
    )

with col2:

    st.metric(
        "뇌졸중 아님으로 답하는 마디",
        f"{negative_leaf_count}개"
    )


st.write(
    "**실제로 질문에 사용한 속성:**",
    ", ".join(used_feature_names)
    if used_feature_names
    else "사용한 속성이 없습니다."
)

st.write(
    "**선택했지만 질문에 사용하지 않은 속성:**",
    ", ".join(unused_feature_names)
    if unused_feature_names
    else "없습니다."
)


st.divider()


# ==========================================
# 14. 마무리
# ==========================================

st.subheader("분석 시 주의할 점")

st.markdown(
    """
    1. 정확도는 전체 예측 중 정답을 맞힌 비율입니다.
    2. 뇌졸중 데이터는 양성과 음성의 비율이 다를 수 있으므로 정확도만으로 모델의 성능을 판단해서는 안 됩니다.
    3. 학습 데이터와 테스트 데이터를 분리한 이유는 학습에 사용하지 않은 데이터에서도 모델이 잘 작동하는지 확인하기 위해서입니다.
    4. 의사결정트리의 가지는 학습 데이터에서 만들어진 질문이며, 실제 의학적 인과관계를 의미하지 않습니다.
    5. 그래프의 2차원 분류 영역은 나머지 속성을 중앙값으로 고정한 결과이므로 전체 데이터의 모든 상황을 표현하지는 않습니다.
    """
)

st.caption(
    "뇌졸중 예측 실습실 | 분류 모델 페이지"
)

st.caption(
    "본 모델은 교육용 실습을 위한 것이며 실제 의료 진단에 사용할 수 없습니다."
)

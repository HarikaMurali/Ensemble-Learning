import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier

st.set_page_config(page_title="Voting Simulator | EnsembleLab", page_icon="🗳️", layout="wide")

# -------------------- DATA & MODELS --------------------

@st.cache_resource
def train_models():
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target  # 0 = Malignant, 1 = Benign

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
            n_estimators=100, max_samples=0.8, random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=100, learning_rate=0.5, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_features="sqrt",
            random_state=42
        ),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return X_test, y_test, models

X_test, y_test, models = train_models()

def label(value):
    return "Benign" if int(value) == 1 else "Malignant"

# -------------------- PAGE HEADER --------------------

st.title("🗳️ Ensemble Voting Simulator")
st.markdown(
    "Explore how individual trees contribute to an ensemble prediction. "
    "Choose a test sample and reveal the learners step by step."
)
st.caption(
    "This page trains the four models on the same Breast Cancer Wisconsin dataset "
    "and fixed train/test split used by this simulator."
)

# -------------------- SAMPLE SELECTION --------------------

sample_options = list(range(len(X_test)))
sample_id = st.selectbox(
    "Choose a test sample",
    sample_options,
    format_func=lambda i: f"Test sample #{i + 1}"
)

row = X_test.iloc[[sample_id]]
actual = int(y_test.iloc[sample_id])

c1, c2, c3 = st.columns(3)
c1.metric("Selected sample", f"#{sample_id + 1}")
c2.metric("Actual diagnosis", label(actual))
c3.metric("Features", f"{X_test.shape[1]}")

st.divider()

# -------------------- MODEL SELECTOR --------------------

model_name = st.selectbox(
    "Choose an algorithm to explore",
    ["Decision Tree", "Bagging", "AdaBoost", "Random Forest"]
)
model = models[model_name]
final_prediction = int(model.predict(row)[0])

left, right = st.columns([1, 1])

with left:
    st.subheader("Final model prediction")
    st.metric("Ensemble output", label(final_prediction))
    if final_prediction == actual:
        st.success("The model prediction matches the actual test label.")
    else:
        st.warning("The model prediction differs from the actual test label.")

with right:
    st.subheader("Prediction confidence")
    probabilities = model.predict_proba(row)[0]
    probability_df = pd.DataFrame({
        "Class": ["Malignant", "Benign"],
        "Probability": probabilities
    })
    fig = px.bar(
        probability_df, x="Class", y="Probability", color="Class",
        range_y=[0, 1], text_auto=".1%"
    )
    fig.update_layout(showlegend=False, height=280, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# -------------------- DECISION TREE --------------------

if model_name == "Decision Tree":
    st.subheader("How the Decision Tree decides")
    st.write(
        "A single decision tree follows a sequence of feature-based questions. "
        "The final leaf gives the prediction."
    )
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(
        model, feature_names=list(X_test.columns),
        class_names=["Malignant", "Benign"],
        filled=True, rounded=True, max_depth=3, fontsize=7, ax=ax
    )
    st.pyplot(fig)
    plt.close(fig)

# -------------------- BAGGING & RANDOM FOREST --------------------

elif model_name in ["Bagging", "Random Forest"]:
    st.subheader("Individual tree predictions")

    estimators = model.estimators_
    if model_name == "Bagging":
        feature_sets = model.estimators_features_
        predictions = [
            int(tree.predict(row.iloc[:, feature_indices])[0])
            for tree, feature_indices in zip(estimators, feature_sets)
        ]
        st.write(
            "Bagging trains trees on different bootstrap samples. "
            "Each tree predicts the selected sample independently."
        )
    else:
        predictions = [int(tree.predict(row)[0]) for tree in estimators]
        st.write(
            "Random Forest combines many trees. Each tree makes a prediction, "
            "while the forest's actual output is based on the average class probabilities."
        )

    reveal_count = st.slider(
        "Reveal the first N trees",
        min_value=1, max_value=len(predictions),
        value=min(10, len(predictions)), step=1
    )

    visible = predictions[:reveal_count]
    counts = pd.Series([label(v) for v in visible]).value_counts()
    vote_df = pd.DataFrame({
        "Prediction": ["Malignant", "Benign"],
        "Number of trees": [
            int(counts.get("Malignant", 0)),
            int(counts.get("Benign", 0))
        ]
    })

    st.plotly_chart(
        px.bar(
            vote_df, x="Prediction", y="Number of trees",
            color="Prediction", text="Number of trees"
        ).update_layout(showlegend=False, height=330),
        use_container_width=True
    )

    st.write(f"Showing **{reveal_count} of {len(predictions)} trees**.")
    st.dataframe(
        pd.DataFrame({
            "Tree": range(1, reveal_count + 1),
            "Prediction": [label(v) for v in visible]
        }),
        use_container_width=True, hide_index=True
    )

    full_counts = pd.Series([label(v) for v in predictions]).value_counts()
    st.info(
        f"Across all {len(predictions)} trees, the individual hard predictions are "
        f"{full_counts.get('Malignant', 0)} Malignant and "
        f"{full_counts.get('Benign', 0)} Benign."
    )

    if model_name == "Bagging":
        st.caption(
            "Note: the chart shows individual tree labels. Scikit-learn's BaggingClassifier "
            "produces its final class using the ensemble's averaged class probabilities, "
            "so its final output is shown separately above."
        )
    else:
        st.caption(
            "Note: Random Forest's final prediction uses averaged tree class probabilities, "
            "not simply the count of hard labels shown in this chart."
        )

# -------------------- ADABOOST --------------------

elif model_name == "AdaBoost":
    st.subheader("AdaBoost: weighted learner contributions")
    st.write(
        "AdaBoost trains weak learners sequentially. Learners receive different weights, "
        "and stronger learners have a larger influence on the combined decision."
    )

    estimators = model.estimators_
    weights = model.estimator_weights_
    errors = model.estimator_errors_

    learner_predictions = [
        int(estimator.predict(row)[0]) for estimator in estimators
    ]

    reveal_count = st.slider(
        "Reveal the first N weak learners",
        min_value=1, max_value=len(estimators),
        value=min(10, len(estimators)), step=1
    )

    learner_df = pd.DataFrame({
        "Learner": range(1, len(estimators) + 1),
        "Prediction": [label(v) for v in learner_predictions],
        "Weight": weights,
        "Training error": errors
    })

    st.dataframe(
        learner_df.head(reveal_count).style.format({
            "Weight": "{:.4f}",
            "Training error": "{:.4f}"
        }),
        use_container_width=True, hide_index=True
    )

    # Weighted class totals across all fitted weak learners.
    weighted_totals = {
        0: sum(float(w) for p, w in zip(learner_predictions, weights) if p == 0),
        1: sum(float(w) for p, w in zip(learner_predictions, weights) if p == 1)
    }

    weighted_df = pd.DataFrame({
        "Class": ["Malignant", "Benign"],
        "Total learner weight": [
            weighted_totals[0], weighted_totals[1]
        ]
    })

    st.plotly_chart(
        px.bar(
            weighted_df, x="Class", y="Total learner weight",
            color="Class", text_auto=".3f"
        ).update_layout(showlegend=False, height=330),
        use_container_width=True
    )

    st.caption(
        "The weighted totals summarize each weak learner's predicted class. "
        "The final prediction displayed above comes directly from AdaBoost's own predict() method."
    )

    st.info(
        f"Displayed {reveal_count} of {len(estimators)} weak learners. "
        "The weighted class totals use the complete fitted ensemble."
    )

# -------------------- TAKEAWAY --------------------

st.divider()
st.subheader("What to notice")
if model_name == "Decision Tree":
    st.write("A single tree makes the decision. It is easy to interpret but can be sensitive to the training data.")
elif model_name == "Bagging":
    st.write("Many trees are trained independently on bootstrap samples. Combining them can reduce the instability of a single tree.")
elif model_name == "AdaBoost":
    st.write("Weak learners are trained sequentially and contribute with different weights to the final decision.")
else:
    st.write("Random Forest combines many trees and random feature selection to build a more diverse collection of trees.")

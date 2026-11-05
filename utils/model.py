import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

MODEL_NAMES = ["Decision Tree", "Bagging", "AdaBoost", "Random Forest"]

@st.cache_data
def load_dataset():
    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.frame.copy()
    # sklearn target: 0 = malignant, 1 = benign
    frame["diagnosis"] = frame["target"].map({0: "Malignant", 1: "Benign"})
    return frame, bunch.data.copy(), bunch.target.copy(), list(bunch.feature_names)

def make_models(n_estimators=100, max_depth=3, learning_rate=0.5):
    tree_depth = None if max_depth == 0 else max_depth
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=tree_depth, random_state=42),
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=tree_depth, random_state=42),
            n_estimators=n_estimators, max_samples=0.8,
            random_state=42, n_jobs=-1
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=n_estimators, learning_rate=learning_rate,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=n_estimators, max_depth=None if max_depth == 0 else max_depth,
            max_features="sqrt", random_state=42, n_jobs=-1
        )
    }

def evaluate_models(models, X_train, X_test, y_train, y_test):
    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        # sklearn label 1 is benign; metrics below use benign as positive class
        proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
        rows.append({
            "Algorithm": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, pos_label=1, zero_division=0),
            "Recall": recall_score(y_test, pred, pos_label=1, zero_division=0),
            "F1 Score": f1_score(y_test, pred, pos_label=1, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, proba)
        })
        fitted[name] = model
    return fitted, pd.DataFrame(rows)

@st.cache_resource
def get_default_experiment(test_size=0.2):
    _, X, y, _ = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    models, metrics = evaluate_models(make_models(), X_train, X_test, y_train, y_test)
    return models, metrics, X_train, X_test, y_train, y_test

def run_custom_experiment(test_size=0.2, n_estimators=100, max_depth=3, learning_rate=0.5):
    _, X, y, _ = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    models, metrics = evaluate_models(
        make_models(n_estimators, max_depth, learning_rate),
        X_train, X_test, y_train, y_test
    )
    return models, metrics, X_train, X_test, y_train, y_test

def class_name(value):
    return "Benign" if int(value) == 1 else "Malignant"

def get_model_probability(model, row):
    probs = model.predict_proba(row)[0]
    class_to_prob = {int(c): float(p) for c, p in zip(model.classes_, probs)}
    return class_to_prob

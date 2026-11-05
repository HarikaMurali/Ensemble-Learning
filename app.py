import streamlit as st
from utils.helpers import inject_theme, hero
from utils.model import load_dataset, get_default_experiment

st.set_page_config(page_title="EnsembleLab | ML Studio", page_icon="🧬", layout="wide")
inject_theme()
frame, X, y, feature_names = load_dataset()
models, metrics, X_train, X_test, y_train, y_test = get_default_experiment()

st.markdown("""
<div class="hero">
  <div class="eyebrow">ENSEMBLE LEARNING · INTERACTIVE ML STUDIO</div>
  <div class="hero-title">Ensemble<span style="color:#9b8cff">Lab</span></div>
  <div class="hero-sub">Explore how multiple learning models work together. Compare four classifiers, inspect their decisions, and experiment with predictions on the Breast Cancer Wisconsin dataset.</div>
</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4=st.columns(4)
c1.metric("Dataset records", f"{len(frame):,}")
c2.metric("Input features", len(feature_names))
c3.metric("Algorithms", "04")
c4.metric("Test samples", len(X_test))

st.markdown("### Your experiment at a glance")
left,right=st.columns([1.2,1])
with left:
    st.markdown("#### Model performance")
    st.dataframe(metrics.style.format({c:"{:.3f}" for c in metrics.columns if c!="Algorithm"}),
                 width="stretch", hide_index=True)
with right:
    best=metrics.loc[metrics["F1 Score"].idxmax()]
    st.markdown("#### Current leader")
    st.metric("Top F1 score", f"{best['F1 Score']:.3f}", best["Algorithm"])
    st.info("The scores are calculated on the same stratified 80/20 train-test split. Precision, recall and F1 treat **Benign** (target 1) as the positive class.")

st.markdown("### Explore the studio")
a,b,c=st.columns(3)
with a:
    st.markdown("**01 · Data Explorer**")
    st.write("Explore class balance, feature distributions and a rotatable 3D PCA projection.")
with b:
    st.markdown("**02 · Algorithm Lab**")
    st.write("Inspect decision trees, boosting rounds, bagged estimators and feature importance.")
with c:
    st.markdown("**03 · Model Arena**")
    st.write("Compare metrics, ROC curves, confusion matrices and model errors.")
st.caption("Educational machine-learning demonstration only. This dataset and these predictions are not intended for medical diagnosis.")

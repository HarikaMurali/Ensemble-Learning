import streamlit as st
import plotly.express as px
from utils.helpers import inject_theme, hero
from utils.model import get_default_experiment, load_dataset
from utils.visualizations import metrics_bar, radar_chart, confusion_heatmap, roc_plot, feature_importance_plot

st.set_page_config(page_title="Model Arena | EnsembleLab",page_icon="⚔️",layout="wide")
inject_theme()
hero("03 · HEAD-TO-HEAD EVALUATION","Model Arena","Put all four classifiers through the same test set. Compare their strengths instead of relying on accuracy alone.")
models,metrics,X_train,X_test,y_train,y_test=get_default_experiment()
_,_,_,features=load_dataset()

metric=st.selectbox("Metric to compare",["Accuracy","Precision","Recall","F1 Score","ROC-AUC"])
st.plotly_chart(metrics_bar(metrics,metric),width="stretch")
left,right=st.columns(2)
with left:
    st.markdown("#### Multi-metric radar")
    st.plotly_chart(radar_chart(metrics),width="stretch")
with right:
    st.markdown("#### Complete scorecard")
    st.dataframe(metrics.style.format({c:"{:.3f}" for c in metrics.columns if c!="Algorithm"}),
                 width="stretch",hide_index=True)

selected=st.selectbox("Inspect confusion matrix for",list(models.keys()))
a,b=st.columns([1,1.2])
with a:
    st.markdown("#### Confusion matrix")
    st.plotly_chart(confusion_heatmap(models[selected],X_test,y_test),width="stretch")
with b:
    st.markdown("#### ROC curves")
    st.plotly_chart(roc_plot(models,X_test,y_test),width="stretch")

st.markdown("#### Most influential features")
which=st.selectbox("Feature importance model",["Random Forest","Decision Tree","Bagging","AdaBoost"])
fig=feature_importance_plot(models,features,which)
if fig is not None:
    st.plotly_chart(fig,width="stretch")
else:
    st.warning("Feature importance is unavailable for this estimator.")

st.markdown("#### Where does the selected model make mistakes?")
pred=models[selected].predict(X_test)
wrong=X_test.copy()
wrong["Actual"]=y_test.map({0:"Malignant",1:"Benign"})
wrong["Predicted"]=[ "Benign" if p==1 else "Malignant" for p in pred]
wrong=wrong[wrong["Actual"]!=wrong["Predicted"]]
st.metric("Misclassified test samples",len(wrong))
if len(wrong):
    st.dataframe(wrong[["Actual","Predicted"]+features[:5]],width="stretch",hide_index=True)
else:
    st.success("No errors on this test split for this model.")

st.caption("Precision, recall and F1 use Benign (target 1) as the positive class. For a medical dataset, malignant-class recall is also important; do not interpret these model scores as clinical performance.")

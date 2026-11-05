import streamlit as st
import plotly.express as px
from utils.helpers import inject_theme, hero
from utils.model import get_default_experiment, run_custom_experiment

st.set_page_config(page_title="Experiment Lab | EnsembleLab",page_icon="🧪",layout="wide")
inject_theme()
hero("05 · CHANGE THE RECIPE","Experiment Lab","Adjust training settings and rerun all four algorithms. Observe how model complexity and ensemble size influence the scores.")

base_models,base_metrics,*_=get_default_experiment()
st.markdown("#### Experiment controls")
with st.sidebar:
    st.markdown("### Experiment settings")
    test_size=st.slider("Test set proportion",0.15,0.35,0.20,0.05)
    n_estimators=st.slider("Number of ensemble estimators",10,200,100,10)
    max_depth=st.slider("Tree maximum depth (0 = unlimited)",0,12,3)
    learning_rate=st.slider("AdaBoost learning rate",0.05,1.5,0.5,0.05)
    run=st.button("Run experiment",width="stretch")

st.write("The default benchmark uses an 80/20 stratified split, 100 estimators, tree depth 3 and AdaBoost learning rate 0.5.")
if run:
    with st.spinner("Training the four models with your selected settings..."):
        models,metrics,X_train,X_test,y_train,y_test=run_custom_experiment(
            test_size=test_size,n_estimators=n_estimators,
            max_depth=max_depth,learning_rate=learning_rate
        )
    st.session_state["custom_metrics"]=metrics
    st.session_state["custom_settings"]={
        "Test fraction":test_size,"Estimators":n_estimators,
        "Tree depth":max_depth,"AdaBoost learning rate":learning_rate
    }

if "custom_metrics" not in st.session_state:
    st.info("Adjust the controls in the sidebar and click **Run experiment** to generate a custom result.")
    st.markdown("#### Baseline results")
    st.dataframe(base_metrics.style.format({c:"{:.3f}" for c in base_metrics.columns if c!="Algorithm"}),
                 width="stretch",hide_index=True)
else:
    metrics=st.session_state["custom_metrics"]
    st.markdown("#### Your experiment results")
    st.json(st.session_state["custom_settings"])
    st.dataframe(metrics.style.format({c:"{:.3f}" for c in metrics.columns if c!="Algorithm"}),
                 width="stretch",hide_index=True)
    compare=metrics.melt(id_vars="Algorithm",value_vars=["Accuracy","Precision","Recall","F1 Score","ROC-AUC"],
                         var_name="Metric",value_name="Score")
    fig=px.bar(compare,x="Algorithm",y="Score",color="Metric",barmode="group",
               color_discrete_sequence=["#9b8cff","#36c5d8","#f5a65b","#58d6a5","#f278a5"])
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",height=470,
                      yaxis=dict(range=[0,1.05],tickformat=".0%"))
    st.plotly_chart(fig,width="stretch")
    base=base_metrics.set_index("Algorithm")
    current=metrics.set_index("Algorithm")
    delta=current[["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]]-base[["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]]
    st.markdown("#### Change from the baseline")
    st.dataframe(delta.style.format("{:+.3f}").background_gradient(cmap="RdYlGn",axis=None),
                 width="stretch")
    st.caption("A positive change is not automatically better in every context. Compare recall for both classes when interpreting medical classification experiments.")

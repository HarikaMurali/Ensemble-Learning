import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from utils.helpers import inject_theme, hero
from utils.model import get_default_experiment, load_dataset
from utils.visualizations import feature_importance_plot

st.set_page_config(page_title="Algorithm Lab | EnsembleLab",page_icon="🧠",layout="wide")
inject_theme()
hero("02 · OPEN THE BLACK BOX","Algorithm Lab","See what sits inside each learner. These views explain the mechanics behind the ensemble predictions.")
models,metrics,X_train,X_test,y_train,y_test=get_default_experiment()
_,_,_,features=load_dataset()

choice=st.selectbox("Select an algorithm",list(models.keys()))
descriptions={
"Decision Tree":"A single tree learns a sequence of feature-based rules. It is our baseline model.",
"Bagging":"Multiple trees are trained independently on bootstrap samples. Their predictions are aggregated by voting.",
"AdaBoost":"Weak learners are added sequentially. Later learners focus more on examples that earlier learners handled poorly.",
"Random Forest":"Many trees are trained on bootstrapped data while considering random feature subsets at splits."
}
st.info(descriptions[choice])

if choice=="Decision Tree":
    estimator=models[choice]
    st.markdown("#### The baseline decision tree")
elif choice=="Bagging":
    idx=st.slider("Choose a tree from the bagging ensemble",1,len(models[choice].estimators_),1)-1
    estimator=models[choice].estimators_[idx]
    st.markdown(f"#### Bagging member {idx+1} of {len(models[choice].estimators_)}")
elif choice=="AdaBoost":
    model=models[choice]
    n=len(model.estimators_)
    idx=st.slider("Choose a boosting round",1,n,1)-1
    estimator=model.estimators_[idx]
    st.markdown(f"#### Weak learner {idx+1} of {n}")
    a,b=st.columns(2)
    a.metric("Estimator error",f"{model.estimator_errors_[idx]:.3f}")
    b.metric("Learner weight",f"{model.estimator_weights_[idx]:.3f}")
    st.caption("A higher estimator weight means that learner contributes more to the final boosted decision.")
else:
    idx=st.slider("Choose a tree from the forest",1,len(models[choice].estimators_),1)-1
    estimator=models[choice].estimators_[idx]
    st.markdown(f"#### Random Forest tree {idx+1} of {len(models[choice].estimators_)}")

fig,ax=plt.subplots(figsize=(18,8))
fig.patch.set_facecolor("#0b1020")
ax.set_facecolor("#0b1020")
plot_tree(estimator,feature_names=features,class_names=["Malignant","Benign"],
          filled=True,rounded=True,fontsize=7,max_depth=3,ax=ax)
ax.set_title("Tree structure (display limited to depth 3)",color="white",pad=15)
ax.tick_params(colors="white")
st.pyplot(fig)
plt.close(fig)
st.caption("For ensembles, the diagram shows one selected member tree, not the entire ensemble. Tree display is limited to depth 3 so the rules remain readable.")

if choice=="AdaBoost":
    st.markdown("#### How boosting changes across rounds")
    st.line_chart({"Estimator error":model.estimator_errors_[:n]})
    st.bar_chart({"Estimator weight":model.estimator_weights_[:n]})
if choice=="Random Forest":
    st.markdown("#### Feature importance")
    fig2=feature_importance_plot(models,features,"Random Forest")
    st.plotly_chart(fig2,width="stretch")
if choice=="Bagging":
    st.markdown("#### Bootstrap sample")
    samples=models["Bagging"].estimators_samples_[idx]
    st.write(f"This tree trained using a bootstrap sample of {len(samples)} rows, sampled with replacement from the training data.")
    st.metric("Unique training rows in this sample",len(set(samples)))
st.markdown("#### Test-set score")
score=metrics.loc[metrics["Algorithm"]==choice,"F1 Score"].iloc[0]
st.metric("F1 score (Benign positive class)",f"{score:.3f}")

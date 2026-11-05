import streamlit as st
import plotly.express as px
from utils.helpers import inject_theme, hero
from utils.model import load_dataset
from utils.visualizations import pca_3d_plot, polish

st.set_page_config(page_title="Data Explorer | EnsembleLab", page_icon="🔍", layout="wide")
inject_theme()
frame,X,y,features=load_dataset()
hero("01 · UNDERSTAND THE DATA","Data Explorer","Explore the dataset before training. Rotate the 3D projection, inspect feature distributions and understand the class balance.")

mal=int((y==0).sum()); ben=int((y==1).sum())
a,b,c=st.columns(3)
a.metric("Total records",f"{len(X):,}")
b.metric("Malignant",f"{mal:,}")
c.metric("Benign",f"{ben:,}")

left,right=st.columns([1.15,1])
with left:
    st.markdown("#### Class distribution")
    counts=frame["diagnosis"].value_counts().rename_axis("Diagnosis").reset_index(name="Records")
    fig=px.pie(counts,names="Diagnosis",values="Records",hole=.62,
               color="Diagnosis",color_discrete_map={"Malignant":"#ff6b8a","Benign":"#36c5d8"})
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",height=360,
                      margin=dict(l=10,r=10,t=35,b=10))
    st.plotly_chart(fig,width="stretch")
with right:
    st.markdown("#### Dataset snapshot")
    st.write("The Scikit-learn Breast Cancer Wisconsin dataset contains 30 numeric measurements computed from digitized images of fine needle aspirates.")
    st.write("Each record has a target label: **Malignant (0)** or **Benign (1)**.")
    st.dataframe(frame[["mean radius","mean texture","mean perimeter","mean area","diagnosis"]].head(8),
                 width="stretch",hide_index=True)

st.markdown("#### 3D feature-space projection")
st.caption("PCA compresses the standardized 30-dimensional data into three components for visualization only. The classifiers still use all 30 original features.")
with st.spinner("Building the 3D projection..."):
    fig,variance=pca_3d_plot(X,y)
st.plotly_chart(fig,width="stretch")
st.caption("Variance captured by the three displayed components: "+", ".join(f"{v:.1%}" for v in variance)+f" (total {sum(variance):.1%}).")

st.markdown("#### Explore individual features")
selected=st.selectbox("Choose a feature",features,index=0)
hist=px.histogram(frame,x=selected,color="diagnosis",marginal="box",opacity=.78,
                  color_discrete_map={"Malignant":"#ff6b8a","Benign":"#36c5d8"},
                  barmode="overlay",nbins=35)
hist.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",height=430)
st.plotly_chart(hist,width="stretch")
with st.expander("View full dataset"):
    st.dataframe(frame,width="stretch",height=360)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

BG = "rgba(0,0,0,0)"
GRID = "rgba(148,163,184,.12)"
COLORS = ["#9b8cff", "#36c5d8", "#f5a65b", "#58d6a5"]

def polish(fig, height=430):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        height=height, margin=dict(l=18,r=18,t=55,b=18),
        font=dict(family="DM Sans", color="#dce5f5"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID)
    )
    return fig

def pca_3d_plot(X, y):
    scaled = StandardScaler().fit_transform(X)
    coords = PCA(n_components=3, random_state=42).fit_transform(scaled)
    df = pd.DataFrame(coords, columns=["Component 1", "Component 2", "Component 3"])
    df["Diagnosis"] = pd.Series(y).map({0:"Malignant",1:"Benign"}).values
    fig = px.scatter_3d(
        df, x="Component 1", y="Component 2", z="Component 3",
        color="Diagnosis", color_discrete_map={"Malignant":"#ff6b8a","Benign":"#36c5d8"},
        opacity=.78, hover_data={"Diagnosis":True}
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(scene=dict(
        xaxis=dict(backgroundcolor="#0b1020", gridcolor=GRID),
        yaxis=dict(backgroundcolor="#0b1020", gridcolor=GRID),
        zaxis=dict(backgroundcolor="#0b1020", gridcolor=GRID)
    ))
    return polish(fig, 600), PCA(n_components=3, random_state=42).fit(scaled).explained_variance_ratio_

def metrics_bar(metrics, metric):
    fig = px.bar(metrics, x="Algorithm", y=metric, color="Algorithm",
                 color_discrete_sequence=COLORS, text_auto=".3f")
    fig.update_yaxes(range=[0,1.05], tickformat=".0%")
    return polish(fig, 390)

def radar_chart(metrics):
    cols = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]
    fig = go.Figure()
    for i, row in metrics.iterrows():
        vals = [row[c] for c in cols]
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=cols+[cols[0]], fill="toself",
            name=row["Algorithm"], line=dict(color=COLORS[i%len(COLORS)])
        ))
    fig.update_layout(polar=dict(
        bgcolor=BG, radialaxis=dict(visible=True, range=[0,1], gridcolor=GRID),
        angularaxis=dict(gridcolor=GRID)
    ))
    return polish(fig, 470)

def confusion_heatmap(model, X_test, y_test):
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=[0,1])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Teal",
                    x=["Predicted Malignant","Predicted Benign"],
                    y=["Actual Malignant","Actual Benign"], aspect="auto")
    fig.update_layout(coloraxis_showscale=False)
    return polish(fig, 390)

def roc_plot(models, X_test, y_test):
    fig = go.Figure()
    for i, (name, model) in enumerate(models.items()):
        pos = list(model.classes_).index(1)
        from sklearn.metrics import roc_curve, roc_auc_score
        prob = model.predict_proba(X_test)[:,pos]
        fpr,tpr,_ = roc_curve(y_test,prob)
        auc = roc_auc_score(y_test,prob)
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"{name} (AUC {auc:.3f})",
                                 line=dict(color=COLORS[i%4],width=2)))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random baseline",
                             line=dict(color="#7b879c",dash="dash")))
    fig.update_layout(xaxis_title="False Positive Rate",yaxis_title="True Positive Rate")
    return polish(fig, 420)

def feature_importance_plot(models, feature_names, selected="Random Forest"):
    model=models[selected]
    if not hasattr(model,"feature_importances_"):
        return None
    imp=pd.DataFrame({"Feature":feature_names,"Importance":model.feature_importances_})
    imp=imp.nlargest(12,"Importance").sort_values("Importance")
    fig=px.bar(imp,x="Importance",y="Feature",orientation="h",
               color="Importance",color_continuous_scale="Teal")
    fig.update_layout(coloraxis_showscale=False)
    return polish(fig, 440)

import streamlit as st

def inject_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --bg:#080b16; --panel:#11172a; --line:rgba(148,163,184,.16); --muted:#9aa7bd; }
    .stApp { background: radial-gradient(ellipse at 15% 0%, #1b1740 0%, #0b1020 36%, #080b16 75%); color:#edf2ff; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.5px; }
    section[data-testid="stSidebar"] { background:linear-gradient(180deg,#11172a,#0b1020); border-right:1px solid var(--line); }
    div[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(28,38,67,.92),rgba(17,23,42,.92)); border:1px solid var(--line); padding:16px 18px; border-radius:16px; }
    div[data-testid="stMetricLabel"] { color:#aab6cc; }
    div[data-testid="stMetricValue"] { color:#f3f6ff; }
    div[data-testid="stExpander"], div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; }
    .hero { padding:28px 30px; border-radius:22px; background:linear-gradient(120deg,rgba(91,64,180,.30),rgba(23,153,185,.14)); border:1px solid rgba(139,128,255,.28); margin-bottom:22px; }
    .eyebrow { color:#a99cff; text-transform:uppercase; letter-spacing:2px; font-size:11px; font-weight:700; }
    .hero-title { font-family:'Space Grotesk',sans-serif; font-size:42px; line-height:1.1; font-weight:700; margin:8px 0; color:#f5f7ff; }
    .hero-sub { color:#b7c3d9; font-size:15px; max-width:760px; }
    .section-kicker { color:#9b8cff; font-size:11px; font-weight:700; letter-spacing:1.7px; text-transform:uppercase; }
    .small-note { color:#9aa7bd; font-size:12px; }
    .stButton>button { border-radius:10px; border:1px solid rgba(139,128,255,.35); background:linear-gradient(90deg,#6554c8,#287eae); color:white; font-weight:600; }
    .stButton>button:hover { border-color:#9e91ff; color:white; }
    </style>
    """, unsafe_allow_html=True)

def hero(eyebrow, title, subtitle):
    st.markdown(f"""
    <div class="hero">
      <div class="eyebrow">{eyebrow}</div>
      <div class="hero-title">{title}</div>
      <div class="hero-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import os
import sys
import plotly.express as px
import duckdb
import joblib
import numpy as np
from pathlib import Path

# Import KikoLens Core
from kikolens.analyze.stats import basic_statistics, correlation_matrix
from kikolens.sql.generator import generate_sql_from_text
from kikolens.analyze.automl import run_automl_analysis
from kikolens.ai.insights import get_ai_insights
from kikolens.utils.loaders import load_data
from kikolens.analyze.discovery import run_discovery_engine
from kikolens.sql.runner import run_sql_query
from kikolens.analyze.clustering import run_clustering_analysis

# --- Configuration ---
st.set_page_config(page_title="KikoLens", page_icon="📊", layout="wide")

# --- Optimized CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.1); padding: 1rem !important; border-radius: 0.5rem !important;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .brand-header { background: #1E293B; padding: 2rem; border-radius: 1rem; color: white; margin-bottom: 2rem; text-align: center; }
    .brand-title { font-size: 3rem; font-weight: 700; letter-spacing: -1px; text-transform: uppercase; margin: 0; }
    .insight-card { padding: 1.5rem; border-radius: 0.75rem; border-left: 5px solid #4F46E5; background: rgba(128, 128, 128, 0.05); margin-bottom: 1rem; }
    .sidebar-brand { font-size: 1.5rem; font-weight: 700; text-align: center; padding: 2rem 0; border-bottom: 1px solid rgba(128, 128, 128, 0.2); }
</style>
""", unsafe_allow_html=True)

if 'df' not in st.session_state: st.session_state.df = None
if 'filename' not in st.session_state: st.session_state.filename = None

def main():
    st.sidebar.markdown("<div class='sidebar-brand'>KIKOLENS</div>", unsafe_allow_html=True)
    st.sidebar.markdown("### Data Source")
    source_mode = st.sidebar.radio("Connection", ["File", "Database"], key="s_mode")
    source = None
    
    if source_mode == "File":
        source = st.sidebar.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'], key="u")
    else:
        db_type = st.sidebar.selectbox("Engine", ["PostgreSQL", "MySQL", "SQLite"])
        if db_type == "SQLite":
            db_path = st.sidebar.text_input("Path to .db"); source = f"sqlite:///{db_path}"
        else:
            host = st.sidebar.text_input("Host", "localhost"); port = st.sidebar.text_input("Port", "5432" if db_type == "PostgreSQL" else "3306")
            user = st.sidebar.text_input("User"); password = st.sidebar.text_input("Password", type="password")
            database = st.sidebar.text_input("DB Name"); driver = "postgresql" if db_type == "PostgreSQL" else "mysql+pymysql"
            source = f"{driver}://{user}:{password}@{host}:{port}/{database}"

    if source:
        current_name = source.name if hasattr(source, 'name') else os.path.basename(str(source))
        if st.session_state.filename != current_name:
            try:
                with st.spinner("Indexing data..."):
                    st.session_state.df = load_data(source); st.session_state.filename = current_name
                    if 'automl_res' in st.session_state: del st.session_state.automl_res
                    st.rerun()
            except Exception as e: st.sidebar.error(f"Error: {e}")

    if st.session_state.df is not None:
        df = st.session_state.df
        page = st.sidebar.radio("Navigation", ["Vision", "Discovery", "Stats", "Chat", "Refinery", "AutoML", "Clusters"], key="nav")
        st.markdown(f"<div class='brand-header'><div class='brand-title'>KIKOLENS</div><div style='opacity:0.8'>Analyzing: {st.session_state.filename}</div></div>", unsafe_allow_html=True)

        if page == "Vision":
            c1, c2, c3 = st.columns(3); c1.metric("Rows", f"{len(df):,}"); c2.metric("Features", len(df.columns)); c3.metric("Health", f"{(1 - df.isnull().sum().sum()/df.size)*100:.1f}%")
            ctrl, plot = st.columns([1, 3])
            x = ctrl.selectbox("X Axis", df.columns); y = ctrl.selectbox("Y Axis", df.columns, index=min(1, len(df.columns)-1))
            ctype = ctrl.radio("Type", ["Scatter", "Bar", "Time Series", "Histogram", "Box Plot"])
            color_dim = ctrl.selectbox("Color", [None] + list(df.columns))
            if "Scatter" in ctype: fig = px.scatter(df, x=x, y=y, color=color_dim, template="plotly_white")
            elif "Bar" in ctype: fig = px.bar(df, x=x, y=y, color=color_dim, template="plotly_white")
            elif "Time" in ctype: fig = px.line(df, x=x, y=y, color=color_dim, template="plotly_white")
            elif "Histogram" in ctype: fig = px.histogram(df, x=x, color=color_dim, template="plotly_white")
            else: fig = px.box(df, x=x, y=y, color=color_dim, template="plotly_white")
            plot.plotly_chart(fig, use_container_width=True)

        elif page == "Discovery":
            st.markdown("### Intelligent Discovery")
            insights = run_discovery_engine(df); cols = st.columns(2)
            for i, insight in enumerate(insights):
                with cols[i % 2]:
                    color = "#EF4444" if insight['severity'] == "high" else "#F59E0B" if insight['severity'] == "medium" else "#3B82F6"
                    st.markdown(f"<div class='insight-card' style='border-left-color: {color}'><div style='font-weight:700; color:{color}; font-size:0.75rem; text-transform:uppercase;'>{insight['type']}</div><div style='font-size:1.1rem; font-weight:600;'>{insight['message']}</div><div style='opacity:0.7; font-size:0.9rem;'>{insight['detail']}</div></div>", unsafe_allow_html=True)

        elif page == "Stats":
            tab1, tab2 = st.tabs(["Statistics", "Correlation"])
            with tab1:
                st.dataframe(basic_statistics(df).style.background_gradient(cmap='viridis', axis=0).format(precision=2), use_container_width=True)
                if st.button("Generate AI Insight"):
                    with st.spinner("Analyzing..."): st.info(get_ai_insights(df))
            with tab2:
                corr = correlation_matrix(df)
                if not corr.empty: st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)

        elif page == "Chat":
            q = st.text_input("Ask anything:", placeholder="Total sales by category")
            if st.button("Run"):
                with st.spinner("Processing..."): sql = generate_sql_from_text(df, q)
                st.code(sql, language="sql")
                try: st.dataframe(run_sql_query(df, sql), use_container_width=True)
                except Exception as e: st.error(f"Error: {e}")

        elif page == "Refinery":
            st.markdown("### Data Refinery")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Drop Columns"); cols = st.multiselect("Select", df.columns)
                if st.button("Remove"): st.session_state.df = df.drop(columns=cols); st.rerun()
            with c2:
                st.markdown("#### Fill Nulls"); null_cols = df.columns[df.isnull().any()].tolist()
                if null_cols:
                    t = st.selectbox("Column", null_cols); s = st.radio("Strategy", ["Mean", "Median", "Mode", "Zero"], horizontal=True)
                    if st.button("Fix"):
                        if s == "Mean": v = df[t].mean()
                        elif s == "Median": v = df[t].median()
                        elif s == "Mode": v = df[t].mode()[0]
                        else: v = 0
                        st.session_state.df[t] = df[t].fillna(v); st.rerun()
            st.dataframe(df.head(10), use_container_width=True)

        elif page == "AutoML":
            st.markdown("### Predictive Analysis")
            mode = st.radio("Selection Mode", ["💎 Auto", "🛠️ Manual", "📤 Import"], horizontal=True)
            target_col = st.selectbox("Target Variable", df.columns)
            
            manual_type = None; imported_model = None
            if mode == "🛠️ Manual": manual_type = st.selectbox("Algorithm", ["Random Forest", "Gradient Boosting", "Decision Tree", "Ridge/Logistic", "Lasso/KNN"])
            elif mode == "📤 Import":
                up = st.file_uploader("Upload .pkl", type=['pkl'])
                if up: imported_model = joblib.load(up); st.success("Model Loaded")

            c_btn, c_opt = st.columns(2)
            export_needed = c_opt.checkbox("Export model after training")

            if c_btn.button("🚀 Train Pipeline") or (mode == "📤 Import" and imported_model):
                with st.spinner("Optimizing Engine..."):
                    if mode == "📤 Import" and imported_model:
                        res = run_automl_analysis(df, target_col, manual_type="Random Forest")
                        res['model'] = imported_model; res['narrative'] = "### 📤 Custom Model Ready\nAnalysis based on imported pre-trained intelligence."
                    else:
                        res = run_automl_analysis(df, target_col, manual_type=manual_type)
                    st.session_state.automl_res = res

            if 'automl_res' in st.session_state and st.session_state.automl_res['target'] == target_col:
                res = st.session_state.automl_res
                st.success(f"Optimized Metrics: {res['metrics']}")
                st.markdown(res["narrative"])
                
                # --- NEW INTERACTIVE PERFORMANCE DASHBOARD ---
                st.markdown("#### Model Performance Dashboard")
                pc1, pc2 = st.columns(2)
                pc1.plotly_chart(res['plots']['performance'], use_container_width=True)
                pc2.plotly_chart(res['plots']['importance'], use_container_width=True)

                if export_needed:
                    out_p = Path("./kikolens_output") / f"kikolens_{target_col}.pkl"
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(res['model'], out_p)
                    with open(out_p, "rb") as f: st.download_button("💾 Download .pkl", f, f"{target_col}_model.pkl")

                st.markdown("---"); st.markdown("### 🔬 Explain & Simulate")
                t1, t2 = st.tabs(["Explain Row", "What-If Simulator"])
                with t1:
                    idx = st.number_input("Row Index", 0, len(df)-1, 0)
                    if st.button("Explain Row Prediction"):
                        row = df.iloc[idx]; data = []
                        for f, i in res["feature_importance"].items():
                            bf = f.split('_')[0] if '_' in f else f
                            if bf in df.columns and pd.api.types.is_numeric_dtype(df[bf]):
                                data.append({"Factor": f, "Impact": (row[bf] - df[bf].mean()) * i})
                        st.plotly_chart(px.bar(pd.DataFrame(data).sort_values(by="Impact"), x="Impact", y="Factor", orientation='h', template="plotly_dark"), use_container_width=True)

                with t2:
                    top_f = list(res["feature_importance"].keys())[:5]; sim_v = {}
                    cols = st.columns(len(top_f))
                    for i, f in enumerate(top_f):
                        bf = next((c for c in res["categorical_mappings"].keys() if f.startswith(c)), None)
                        if bf: sim_v[bf] = cols[i].selectbox(f, res["categorical_mappings"][bf])
                        elif f in df.columns: sim_v[f] = cols[i].slider(f, float(df[f].min()), float(df[f].max()), float(df[f].mean()))
                    if st.button("Simulate"):
                        ir = pd.DataFrame([0] * len(res["columns"]), index=res["columns"]).T
                        for f, v in sim_v.items():
                            if f in ir.columns: ir[f] = v
                            else:
                                d = f"{f}_{v}"
                                if d in ir.columns: ir[d] = 1
                        pred = res["model"].predict(res["scaler"].transform(res["imputer"].transform(ir)))[0]
                        st.metric("Predicted Outcome", f"{pred:.4f}")

        elif page == "Clusters":
            k = st.slider("Segments", 2, 8, 3)
            if st.button("Run Clustering"):
                res = run_clustering_analysis(df, k)
                st.plotly_chart(res["fig"], use_container_width=True)
                st.dataframe(res["profiles"].style.background_gradient(cmap='Purples'), use_container_width=True)

    else: st.markdown("<div style='text-align:center; padding-top:100px;'><h1>KIKOLENS</h1><p>Connect your data via the sidebar to start.</p></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

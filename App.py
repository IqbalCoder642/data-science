# app.py
"""
Streamlit Data Science Dashboard
Features:
- Upload CSV/Excel
- Data inspection: head, dtypes, summary, missing values
- Interactive Plotly & Seaborn plots
- Simple ML helpers (classification/regression baseline, PCA)
- Download plots & processed data
"""

import io
import base64
from pathlib import Path

import pandas as pd
import numpy as np

import streamlit as st
from stqdm import stqdm  # optional progress bar (safe fallback if missing)
import plotly.express as px
import plotly.graph_objects as go

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------
# Helpers
# ---------------------------
st.set_page_config(page_title="Data Science Dashboard", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data(file) -> pd.DataFrame:
    if isinstance(file, pd.DataFrame):
        return file
    file.seek(0)
    fname = getattr(file, "name", "")
    if fname.endswith(".csv") or fname.lower().endswith(".txt"):
        return pd.read_csv(file)
    elif fname.endswith(".xlsx") or fname.endswith(".xls"):
        return pd.read_excel(file)
    else:
        # try csv first, then excel
        try:
            return pd.read_csv(file)
        except Exception:
            file.seek(0)
            return pd.read_excel(file)

def download_df_csv(df: pd.DataFrame, filename="data.csv"):
    return st.download_button(
        label="Download data as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=filename,
        mime="text/csv"
    )

def fig_to_png_bytes_matplotlib(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def fig_to_png_bytes_plotly(fig):
    # requires kaleido in environment
    try:
        img_bytes = fig.to_image(format="png", engine="kaleido")
        return img_bytes
    except Exception as e:
        st.warning("Plotly PNG export failed (kaleido missing?). Returning interactive plot only. Error: " + str(e))
        return None

def download_plotly(fig, filename="plot.png", label="Download plot as PNG"):
    img = fig_to_png_bytes_plotly(fig)
    if img:
        st.download_button(label=label, data=img, file_name=filename, mime="image/png")

def download_matplotlib(plt_fig, filename="plot.png", label="Download plot as PNG"):
    img = fig_to_png_bytes_matplotlib(plt_fig)
    st.download_button(label=label, data=img, file_name=filename, mime="image/png")

# ---------------------------
# UI: Sidebar - Upload & Options
# ---------------------------
st.sidebar.title("Data & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel file", type=["csv", "xlsx", "xls", "txt"])
use_sample = st.sidebar.checkbox("Use sample dataset (Iris)", value=False)

show_all_plots = st.sidebar.checkbox("Show advanced plots", value=True)
random_seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("## Deployment / Files")
st.sidebar.code("app.py")
st.sidebar.code("environment.yml")
st.sidebar.code("requirements.txt")
st.sidebar.markdown("Tip: push these files to a GitHub repo and connect the repo to Streamlit Cloud (https://streamlit.io/cloud)")

# ---------------------------
# Load data
# ---------------------------
if use_sample and uploaded_file is None:
    df = sns.load_dataset("iris")
    st.sidebar.success("Using sample Iris dataset")
elif uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Failed to load file: {e}")
        st.stop()
else:
    st.info("Upload a CSV/Excel file or check 'Use sample dataset' to get started.")
    st.stop()

# ---------------------------
# Main layout
# ---------------------------
st.title("📊 Data Science Dashboard")
st.markdown("Upload your data, inspect it, create visualizations, run quick baseline models, and download results.")

# Data preview + download
st.header("1. Data preview & basic info")
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Dataframe (first rows)")
    st.dataframe(df.head(200))
with col2:
    st.subheader("Basic info")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])
    download_df_csv(df, filename="uploaded_data.csv")

st.subheader("Column types & missing values")
dtypes = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(), "unique": df.nunique()})
st.dataframe(dtypes)

st.subheader("Summary statistics")
st.write(df.describe(include='all').T)

# ---------------------------
# Column selectors
# ---------------------------
st.header("2. Choose columns for analysis")
all_cols = df.columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

colA, colB, colC = st.columns(3)
with colA:
    x_col = st.selectbox("X column (for scatter / time)", options=[None] + all_cols, index=0)
with colB:
    y_col = st.selectbox("Y column (for scatter / time)", options=[None] + all_cols, index=0)
with colC:
    hue_col = st.selectbox("Color / category (hue)", options=[None] + all_cols, index=0)

# If a column appears datetime-like, coerce
for c in all_cols:
    if df[c].dtype == 'object':
        try:
            parsed = pd.to_datetime(df[c], errors='raise', infer_datetime_format=True)
            df[c] = parsed
        except Exception:
            pass

# ---------------------------
# Visualizations
# ---------------------------
st.header("3. Visualizations")

# 3.1: Time series (if x is datetime)
if x_col and np.issubdtype(df[x_col].dtype, np.datetime64) and y_col and np.issubdtype(df[y_col].dtype, np.number):
    st.subheader("Time series")
    fig = px.line(df.sort_values(by=x_col), x=x_col, y=y_col, color=hue_col, title=f"Time series: {y_col} over {x_col}")
    st.plotly_chart(fig, use_container_width=True)
    download_plotly(fig, filename="timeseries.png")

# 3.2: Scatter
if x_col and y_col and (np.issubdtype(df[x_col].dtype, np.number) and np.issubdtype(df[y_col].dtype, np.number)):
    st.subheader("Scatter plot")
    fig = px.scatter(df, x=x_col, y=y_col, color=hue_col, title=f"Scatter: {y_col} vs {x_col}", trendline="ols")
    st.plotly_chart(fig, use_container_width=True)
    download_plotly(fig, filename="scatter.png")

# 3.3: Histogram / Distribution
st.subheader("Distribution / Histogram")
dist_col = st.selectbox("Choose numeric column for distribution", options=[None] + numeric_cols, index=0)
if dist_col:
    nbins = st.slider("Number of bins", 5, 200, 30)
    fig = px.histogram(df, x=dist_col, nbins=nbins, color=hue_col, marginal="box", title=f"Distribution of {dist_col}")
    st.plotly_chart(fig, use_container_width=True)
    download_plotly(fig, filename=f"hist_{dist_col}.png")

# 3.4: Boxplot
st.subheader("Boxplot")
box_num = st.selectbox("Numeric column for boxplot", options=[None] + numeric_cols, index=0, key="box_num")
box_cat = st.selectbox("Categorical column (optional)", options=[None] + categorical_cols, index=0, key="box_cat")
if box_num:
    fig = px.box(df, x=box_cat if box_cat else None, y=box_num, color=box_cat if box_cat else None, title=f"Boxplot of {box_num}")
    st.plotly_chart(fig, use_container_width=True)
    download_plotly(fig, filename=f"box_{box_num}.png")

# 3.5: Correlation heatmap (Seaborn)
st.subheader("Correlation heatmap (Seaborn)")
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr()
    fig_m, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=ax)
    ax.set_title("Correlation matrix")
    st.pyplot(fig_m)
    download_matplotlib(fig_m, filename="corr_heatmap.png")
else:
    st.info("At least two numeric columns required for correlation heatmap.")

# 3.6: Pairplot (Seaborn) - careful for large data
if show_all_plots:
    st.subheader("Pairplot (Seaborn) — small sample only")
    max_points = st.slider("Max points to sample for pairplot", 50, 2000, 300)
    numeric_df = df.select_dtypes(include=[np.number])
    # sample only from numeric columns; handle case where there are none
    if numeric_df.shape[1] == 0:
        st.info("Pairplot requires numeric columns — none found in the dataset.")
    else:
        sample_n = min(max_points, len(numeric_df)) if len(numeric_df) > 0 else 0
        sample_df = numeric_df.sample(n=sample_n, random_state=int(random_seed)) if sample_n > 0 else pd.DataFrame()
        # Require at least 2 numeric columns for a meaningful pairplot, and limit to <=6 columns
        if sample_df.shape[0] == 0 or sample_df.shape[1] == 0:
            st.info("No data available to sample for pairplot (empty selection).")
        elif sample_df.shape[1] > 6:
            st.warning("Pairplot disabled: too many numeric columns. Reduce to ≤6 numeric columns or uncheck advanced plots.")
        elif sample_df.shape[1] >= 2:
            fig_pair = sns.pairplot(sample_df)
            st.pyplot(fig_pair.fig)
            download_matplotlib(fig_pair.fig, filename="pairplot.png")
        else:
            st.info("Only one numeric column found — pairplot skipped. Use histogram/boxplot for single-column analysis.")

# ---------------------------
# Quick ML: Baseline model
# ---------------------------
st.header("4. Quick baseline ML")
task = st.radio("Task type", options=["None", "Classification (target categorical)", "Regression (target numeric)"], index=0)
if task != "None":
    target = st.selectbox("Select target column", options=[None] + all_cols, index=0, key="ml_target")
    if target:
        features = st.multiselect("Select feature columns (if empty, auto-select numeric)", options=[c for c in all_cols if c != target], default=None)
        if not features:
            features = [c for c in numeric_cols if c != target]
        st.write("Using features:", features)

        X = df[features].copy()
        y = df[target].copy()

        # Drop rows with NA in chosen columns
        data_ml = pd.concat([X, y], axis=1).dropna()
        X = data_ml[features]
        y = data_ml[target]

        # Encode categorical features (simple)
        X_encoded = pd.get_dummies(X, drop_first=True)
        # If classification, try to encode y if object
        if task.startswith("Classification") and y.dtype == object:
            y_encoded = pd.factorize(y)[0]
        else:
            y_encoded = y

        test_size = st.slider("Test size (%)", 5, 50, 20)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=test_size/100.0, random_state=int(random_seed))

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if task.startswith("Classification"):
            model = RandomForestClassifier(random_state=int(random_seed), n_estimators=100)
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            acc = accuracy_score(y_test, preds)
            st.write("Accuracy:", acc)
            if len(np.unique(y_test)) == 2:
                if hasattr(model, "predict_proba"):
                    auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
                    st.write("ROC AUC:", auc)
            st.subheader("Feature importance (top 20)")
            feat_imp = pd.Series(model.feature_importances_, index=X_encoded.columns).sort_values(ascending=False).head(20)
            st.bar_chart(feat_imp)
            st.write(feat_imp)
            # download feature importances
            st.download_button("Download feature importance CSV", data=feat_imp.to_csv().encode('utf-8'), file_name="feature_importance.csv", mime="text/csv")

        else:
            model = RandomForestRegressor(random_state=int(random_seed), n_estimators=100)
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            rmse = mean_squared_error(y_test, preds, squared=False)
            r2 = r2_score(y_test, preds)
            st.write("RMSE:", rmse)
            st.write("R^2:", r2)
            st.subheader("Feature importance (top 20)")
            feat_imp = pd.Series(model.feature_importances_, index=X_encoded.columns).sort_values(ascending=False).head(20)
            st.bar_chart(feat_imp)
            st.write(feat_imp)
            st.download_button("Download feature importance CSV", data=feat_imp.to_csv().encode('utf-8'), file_name="feature_importance.csv", mime="text/csv")

        # PCA visualization
        st.subheader("PCA (2 components) visualization of features")
        pca = PCA(n_components=2, random_state=int(random_seed))
        components = pca.fit_transform(X_encoded.fillna(0))
        pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
        if len(y_encoded) == len(pca_df):
            pca_df["target"] = y_encoded
        fig = px.scatter(pca_df, x="PC1", y="PC2", color="target", title="PCA 2D")
        st.plotly_chart(fig, use_container_width=True)
        download_plotly(fig, filename="pca.png")

# ---------------------------
# Export processed data
# ---------------------------
st.header("5. Export / Save")
st.write("You can download the processed dataframe (coerced dtypes and filled samples).")
download_df_csv(df, filename="processed_data.csv")

# ---------------------------
# Footer: Tips to extend
# ---------------------------
st.markdown("---")
st.markdown("#### Tips to extend this app")
st.markdown("""
- Add more model types (XGBoost, LightGBM) — install packages and import.  
- Add advanced profiling (ydata-profiling / pandas-profiling) - note large memory.  
- Use caching for heavy computations (`@st.cache_data` / `@st.cache_resource`).  
- For Plotly PNG export ensure `kaleido` is installed in the environment.
""")

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="AI Similitude Learning", layout="wide")
st.title("AI Similitude Learning GUI")

# ---------------- Helpers ----------------
def load_table(uploaded_file):
    if uploaded_file is None:
        return None
    if getattr(uploaded_file, "size", 0) == 0:
        st.error("Uploaded file is empty.")
        return None

    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    try:
        if name.endswith(".csv"):
            try:
                return pd.read_csv(io.BytesIO(data))
            except Exception:
                return pd.read_csv(io.BytesIO(data), sep=";", engine="python")
        elif name.endswith(".xlsx"):
            return pd.read_excel(io.BytesIO(data), engine="openpyxl")
        else:
            st.error("Unsupported file type. Please upload CSV or XLSX.")
            return None
    except Exception as e:
        st.error("Failed to read uploaded file.")
        st.exception(e)
        return None

def safe_log10(arr, name="value"):
    arr = np.asarray(arr, dtype=float)
    if np.any(arr <= 0):
        raise ValueError(f"{name} contains non-positive values; cannot apply log10.")
    return np.log10(arr)

def parse_hidden_layers(n_layers: int, neurons_text: str):
    neurons_text = (neurons_text or "").strip()
    if neurons_text == "":
        return tuple([64] * n_layers)
    parts = [p.strip() for p in neurons_text.split(",") if p.strip() != ""]
    nums = [int(p) for p in parts]
    if len(nums) == 1:
        return tuple([nums[0]] * n_layers)
    if len(nums) != n_layers:
        raise ValueError("Neurons list must match number of layers, or give a single value.")
    return tuple(nums)

def train_ann_curve(df, xcol, ycol, use_log=True, scaler_type="standard",
                    hidden_layers=(64, 32), lr=0.005, alpha=0.001,
                    test_size=0.2, early_stopping=True, random_state=42):

    x = df[xcol].to_numpy(dtype=float).reshape(-1, 1)
    y = df[ycol].to_numpy(dtype=float).reshape(-1, 1)

    if use_log:
        x_t = safe_log10(x, xcol)
        y_t = safe_log10(y, ycol)
    else:
        x_t, y_t = x, y

    if scaler_type == "minmax":
        xs, ys = MinMaxScaler(), MinMaxScaler()
    else:
        xs, ys = StandardScaler(), StandardScaler()

    X = xs.fit_transform(x_t)
    Y = ys.fit_transform(y_t).ravel()

    X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        learning_rate_init=lr,
        alpha=alpha,
        early_stopping=early_stopping,
        validation_fraction=0.2 if early_stopping else 0.0,
        max_iter=10000,
        random_state=random_state,
    )
    model.fit(X_tr, y_tr)

    # diagnostics data
    y_pred_te_s = model.predict(X_te).reshape(-1, 1)
    y_te_s = y_te.reshape(-1, 1)
    y_pred_te_t = ys.inverse_transform(y_pred_te_s)
    y_te_t = ys.inverse_transform(y_te_s)

    if use_log:
        y_pred_te = (10 ** y_pred_te_t).ravel()
        y_te_orig = (10 ** y_te_t).ravel()
    else:
        y_pred_te = y_pred_te_t.ravel()
        y_te_orig = y_te_t.ravel()

    r2 = r2_score(y_te_orig, y_pred_te)
    rmse = float(np.sqrt(mean_squared_error(y_te_orig, y_pred_te)))

    # predict curve on all X (sorted by original del_K)
    y_pred_all_s = model.predict(X).reshape(-1, 1)
    y_pred_all_t = ys.inverse_transform(y_pred_all_s)
    if use_log:
        y_pred_all = (10 ** y_pred_all_t).ravel()
    else:
        y_pred_all = y_pred_all_t.ravel()

    curve = pd.DataFrame({"del_K": df[xcol].to_numpy(dtype=float), "y_pred": y_pred_all}).sort_values("del_K")

    plot_data = {
        "loss": model.loss_curve_,
        "y_true": y_te_orig,
        "y_pred": y_pred_te,
        "r2": float(r2),
        "rmse": float(rmse),
    }
    return curve.reset_index(drop=True), plot_data

def render_diagnostics(plot_data, title):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(plot_data["loss"])
    ax[0].set_title(f"{title} Loss")
    ax[0].grid(True)

    y_true = plot_data["y_true"]
    y_pred = plot_data["y_pred"]
    ax[1].scatter(y_true, y_pred, s=10)
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax[1].plot([lo, hi], [lo, hi], linestyle="--")
    ax[1].set_title(f"{title} Pred vs Actual\nR²={plot_data['r2']:.4f}, RMSE={plot_data['rmse']:.3e}")
    ax[1].grid(True)
    fig.tight_layout()
    return fig

def first_order_similitude(ts1_curve, ts2_curve, beta1, beta2, gt_df=None, gt_x=None, gt_y=None):
    R = (1 - (1 / beta1)) / ((1 / beta1) - (1 / beta2))

    w1 = ts1_curve.sort_values("del_K")
    w2 = ts2_curve.sort_values("del_K")

    x1min, x1max = w1["del_K"].min(), w1["del_K"].max()
    x2min, x2max = w2["del_K"].min(), w2["del_K"].max()

    if gt_df is not None and gt_x and gt_y:
        delK_phys_all = gt_df[gt_x].to_numpy(dtype=float)
        GT_all = gt_df[gt_y].to_numpy(dtype=float)
    else:
        xmin_phys = max(x1min / np.sqrt(beta1), x2min / np.sqrt(beta2))
        xmax_phys = min(x1max / np.sqrt(beta1), x2max / np.sqrt(beta2))
        delK_phys_all = np.linspace(xmin_phys, xmax_phys, 2000)
        GT_all = None

    xmin_phys = max(delK_phys_all.min(), x1min / np.sqrt(beta1), x2min / np.sqrt(beta2))
    xmax_phys = min(delK_phys_all.max(), x1max / np.sqrt(beta1), x2max / np.sqrt(beta2))
    mask = (delK_phys_all >= xmin_phys) & (delK_phys_all <= xmax_phys)

    delK_phys = delK_phys_all[mask]
    GT = GT_all[mask] if GT_all is not None else None

    delK_ts1 = np.sqrt(beta1) * delK_phys
    delK_ts2 = np.sqrt(beta2) * delK_phys

    da_ts1 = np.interp(delK_ts1, w1["del_K"].to_numpy(), w1["y_pred"].to_numpy())
    da_ts2 = np.interp(delK_ts2, w2["del_K"].to_numpy(), w2["y_pred"].to_numpy())

    p = (beta1 ** (-1.5)) * da_ts1
    q = (beta2 ** (-1.5)) * da_ts2
    pred = p + R * (p - q)

    out = pd.DataFrame({"del_K": delK_phys, "AI_W36_p": p, "AI_W18_q": q, "AI_similitude_W72": pred})
    metrics = {"R": float(R), "xmin_phys": float(xmin_phys), "xmax_phys": float(xmax_phys)}
    if GT is not None:
        out["GT_W72"] = GT
        metrics["r2_vs_gt"] = float(r2_score(GT, pred))
        metrics["rmse_vs_gt"] = float(np.sqrt(mean_squared_error(GT, pred)))
    return out, metrics


# ---------------- Sidebar ----------------
st.sidebar.header("Controls")
order = st.sidebar.selectbox("Order of similitude", [1], index=0)  # MVP: first-order only
beta1 = st.sidebar.number_input("β1", value=0.5, min_value=1e-6, format="%.6f")
beta2 = st.sidebar.number_input("β2", value=0.25, min_value=1e-6, format="%.6f")

st.sidebar.subheader("Upload data (CSV/XLSX)")
ts1_file = st.sidebar.file_uploader("TS1 (β1)", type=["csv", "xlsx"], key="ts1")
ts2_file = st.sidebar.file_uploader("TS2 (β2)", type=["csv", "xlsx"], key="ts2")
gt_file = st.sidebar.file_uploader("GT (optional)", type=["csv", "xlsx"], key="gt")

ts1_df = load_table(ts1_file)
ts2_df = load_table(ts2_file)
gt_df = load_table(gt_file)

colL, colR = st.columns([1, 1.2], gap="large")
with colR:
    st.subheader("Similitude rule")

    st.latex(r"R_1 = \frac{1-\beta_1^{-1}}{\beta_1^{-1}-\beta_2^{-1}}")

    # First-order crack-growth (experimental) rule
    st.latex(
        r"\left(\frac{da}{dN}\right)_{\mathrm{phys}}"
        r"=\beta_1^{-3/2}\left(\frac{da}{dN}\right)_{ts1}"
        r"+R_1\left[\beta_1^{-3/2}\left(\frac{da}{dN}\right)_{ts1}"
        r"-\beta_2^{-3/2}\left(\frac{da}{dN}\right)_{ts2}\right]"
    )

    st.latex(r"\Delta K_{ts} = \sqrt{\beta}\,\Delta K_{\mathrm{phys}}")

with colL:
    st.subheader("Step 3: Column mapping")
    def mapping_ui(df, label, key):
        if df is None:
            st.info(f"Upload {label} to continue.")
            return None, None
        st.dataframe(df.head(8), use_container_width=True)
        cols = list(df.columns)
        x = st.selectbox(f"{label} x (ΔK)", cols, key=f"{key}_x")
        y = st.selectbox(f"{label} y (da/dN)", cols, key=f"{key}_y")
        return x, y

    ts1_x, ts1_y = mapping_ui(ts1_df, "TS1", "ts1")
    ts2_x, ts2_y = mapping_ui(ts2_df, "TS2", "ts2")
    gt_x, gt_y = mapping_ui(gt_df, "GT (optional)", "gt")

tab4, tab5 = st.tabs(["Step 4 — ANN training", "Step 5 — Similitude prediction"])

with tab4:
    st.subheader("Train ANN surrogates")
    use_log = st.checkbox("Use log10 transform (recommended)", value=True)
    scaler = st.selectbox("Scaler", ["standard", "minmax"], index=0)
    n_layers = st.number_input("Hidden layers", 1, 5, 2)
    neurons = st.text_input("Neurons (e.g., 64 or 64,32)", "64,32")
    lr = st.number_input("Learning rate", min_value=1e-6, value=0.005, format="%.6f")
    alpha = st.number_input("Alpha", min_value=0.0, value=0.001, format="%.6f")

    if st.button("Train TS1 & TS2", type="primary"):
        if ts1_df is None or ts2_df is None:
            st.error("Upload TS1 and TS2 first.")
        else:
            try:
                hl = parse_hidden_layers(int(n_layers), neurons)
                ts1_curve, ts1_plot = train_ann_curve(ts1_df, ts1_x, ts1_y, use_log=use_log, scaler_type=scaler, hidden_layers=hl, lr=lr, alpha=alpha)
                ts2_curve, ts2_plot = train_ann_curve(ts2_df, ts2_x, ts2_y, use_log=use_log, scaler_type=scaler, hidden_layers=hl, lr=lr, alpha=alpha)

                st.session_state["ts1_curve"] = ts1_curve
                st.session_state["ts2_curve"] = ts2_curve
                st.session_state["ts1_plot"] = ts1_plot
                st.session_state["ts2_plot"] = ts2_plot
                st.success("Training done.")
            except Exception as e:
                st.error("Training failed.")
                st.exception(e)

    if "ts1_plot" in st.session_state:
        st.pyplot(render_diagnostics(st.session_state["ts1_plot"], "TS1"), clear_figure=True)
    if "ts2_plot" in st.session_state:
        st.pyplot(render_diagnostics(st.session_state["ts2_plot"], "TS2"), clear_figure=True)

with tab5:
    st.subheader("Similitude prediction")

    if st.button("Run similitude", type="secondary"):
        if "ts1_curve" not in st.session_state or "ts2_curve" not in st.session_state:
            st.error("Train ANN first.")
        else:
            out, met = first_order_similitude(
                st.session_state["ts1_curve"],
                st.session_state["ts2_curve"],
                beta1, beta2,
                gt_df=gt_df, gt_x=gt_x, gt_y=gt_y
            )
            st.session_state["out"] = out
            st.session_state["met"] = met
            st.success("Similitude computed. Choose ΔK range to plot.")

    if "out" in st.session_state:
        out = st.session_state["out"]
        st.write("Metrics:", st.session_state.get("met", {}))
        st.dataframe(out.head(15), use_container_width=True)

        xmin, xmax = float(out["del_K"].min()), float(out["del_K"].max())
        x0, x1 = st.slider("ΔK plot range", xmin, xmax, (xmin, xmax))
        plot_df = out[(out["del_K"] >= x0) & (out["del_K"] <= x1)]

        fig = plt.figure(figsize=(9, 5))
        plt.plot(plot_df["del_K"], plot_df["AI_W36_p"], label="AI_W36", linestyle="--")
        plt.plot(plot_df["del_K"], plot_df["AI_W18_q"], label="AI_W18", linestyle="--")
        if "GT_W72" in plot_df.columns:
            plt.plot(plot_df["del_K"], plot_df["GT_W72"], label="GT_W72", linestyle="-")
        plt.plot(plot_df["del_K"], plot_df["AI_similitude_W72"], label="AI_similitude_W72", linestyle="-.")
        plt.yscale("log")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        # Downloads
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download results (CSV)", csv_bytes, "matched_results.csv", "text/csv")

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            out.to_excel(writer, index=False, sheet_name="matched")
        st.download_button("Download results (Excel)", buffer.getvalue(), "matched_results.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
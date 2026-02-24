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


# ---------------- Sidebar ----------------
st.sidebar.header("Controls")

problem = st.sidebar.selectbox(
    "Problem template",
    ["Crack growth (ΔK → da/dN) — Demo"]
)

order = st.sidebar.selectbox("Order of similitude", [0, 1, 2], index=1)

st.sidebar.subheader("Scale parameters (β)")
beta1 = st.sidebar.number_input("β1", value=0.5, min_value=1e-6, format="%.6f")
beta2 = st.sidebar.number_input("β2", value=0.25, min_value=1e-6, format="%.6f")

st.sidebar.subheader("Step 3: Upload experimental data")
ts1_file = st.sidebar.file_uploader("TS1 (β1) data (CSV/XLSX)", type=["csv", "xlsx"], key="ts1")
ts2_file = st.sidebar.file_uploader("TS2 (β2) data (CSV/XLSX)", type=["csv", "xlsx"], key="ts2")
gt_file  = st.sidebar.file_uploader("GT (optional) data (CSV/XLSX)", type=["csv", "xlsx"], key="gt")


# ---------------- Helpers ----------------
def load_table(uploaded_file):
    """Load CSV or XLSX into a dataframe."""
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload CSV or XLSX.")
            return None
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None


def parse_hidden_layers(n_layers: int, neurons_text: str):
    """
    neurons_text can be:
    - "64" (repeated for n_layers)
    - "64,32,16" (must match n_layers)
    """
    neurons_text = (neurons_text or "").strip()
    if neurons_text == "":
        return (64, 32) if n_layers == 2 else tuple([64] * n_layers)

    parts = [p.strip() for p in neurons_text.split(",") if p.strip() != ""]
    nums = [int(p) for p in parts]

    if len(nums) == 1:
        return tuple([nums[0]] * n_layers)

    if len(nums) != n_layers:
        raise ValueError(
            f"Neurons list length ({len(nums)}) must equal number of layers ({n_layers}), "
            f"or provide a single value."
        )

    return tuple(nums)


def safe_log10(arr, name="value"):
    """Log10 with safety checks."""
    arr = np.asarray(arr, dtype=float)
    if np.any(arr <= 0):
        raise ValueError(f"{name} contains non-positive values; cannot apply log10.")
    return np.log10(arr)


def train_ann_curve(df, xcol, ycol, *,
                    scaler_type="standard",
                    use_log=True,
                    hidden_layers=(64, 32),
                    learning_rate_init=0.005,
                    alpha=0.001,
                    test_size=0.2,
                    early_stopping=True,
                    random_state=42):
    """
    Train ANN on (x,y) and return:
    - model
    - plot_data (loss curve + test pred/actual)  [to re-render later]
    - curve_df with columns: del_K, y_pred (ANN predicted on all x)
    - metrics dict (R2, RMSE on test in original space)
    """
    x = df[xcol].to_numpy(dtype=float).reshape(-1, 1)
    y = df[ycol].to_numpy(dtype=float).reshape(-1, 1)

    # log transform if enabled (recommended for da/dN vs ΔK)
    if use_log:
        x_t = safe_log10(x, name=xcol)
        y_t = safe_log10(y, name=ycol)
    else:
        x_t = x
        y_t = y

    # scalers
    if scaler_type == "minmax":
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
    else:
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

    Xs = x_scaler.fit_transform(x_t)
    ys = y_scaler.fit_transform(y_t).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, ys, test_size=test_size, random_state=random_state
    )

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        early_stopping=early_stopping,
        validation_fraction=0.2 if early_stopping else 0.0,
        max_iter=10000,
        random_state=random_state,
        verbose=False
    )
    model.fit(X_train, y_train)

    # --- Evaluate on test set (convert back to original space) ---
    y_pred_test_s = model.predict(X_test).reshape(-1, 1)
    y_test_s = y_test.reshape(-1, 1)

    y_pred_test_t = y_scaler.inverse_transform(y_pred_test_s)
    y_test_t = y_scaler.inverse_transform(y_test_s)

    if use_log:
        y_pred_test = (10 ** y_pred_test_t).ravel()
        y_test_orig = (10 ** y_test_t).ravel()
    else:
        y_pred_test = y_pred_test_t.ravel()
        y_test_orig = y_test_t.ravel()

    r2 = r2_score(y_test_orig, y_pred_test)
    rmse = float(np.sqrt(mean_squared_error(y_test_orig, y_pred_test)))

    # --- Predict on all points (for curve used in similitude) ---
    y_pred_all_s = model.predict(Xs).reshape(-1, 1)
    y_pred_all_t = y_scaler.inverse_transform(y_pred_all_s)

    if use_log:
        y_pred_all = (10 ** y_pred_all_t).ravel()
    else:
        y_pred_all = y_pred_all_t.ravel()

    curve_df = pd.DataFrame({
        "del_K": df[xcol].to_numpy(dtype=float),
        "y_pred": y_pred_all
    }).sort_values("del_K").reset_index(drop=True)

    plot_data = {
        "loss_curve": list(model.loss_curve_),
        "y_test_actual": y_test_orig.tolist(),
        "y_test_pred": y_pred_test.tolist(),
        "r2": float(r2),
        "rmse": float(rmse),
    }

    metrics = {"r2": float(r2), "rmse": float(rmse)}
    return model, plot_data, curve_df, metrics


def render_ann_diagnostics(plot_data, title="ANN"):
    loss_curve = np.asarray(plot_data["loss_curve"], dtype=float)
    y_true = np.asarray(plot_data["y_test_actual"], dtype=float)
    y_pred = np.asarray(plot_data["y_test_pred"], dtype=float)
    r2 = float(plot_data["r2"])
    rmse = float(plot_data["rmse"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Loss
    axes[0].plot(loss_curve)
    axes[0].set_title(f"{title} — Training Loss")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    # Pred vs actual
    axes[1].scatter(y_true, y_pred, s=8, alpha=0.7)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    axes[1].plot([lo, hi], [lo, hi], linestyle="--")
    axes[1].set_title(f"{title} — Pred vs Actual\nR²={r2:.4f}, RMSE={rmse:.3e}")
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].grid(True)

    fig.tight_layout()
    return fig


def compute_first_order_similitude(ts1_curve, ts2_curve, *,
                                  beta1, beta2,
                                  gt_df=None,
                                  gt_xcol=None, gt_ycol=None,
                                  phys_grid_n=2000):
    """
    Paper-consistent:
      ΔK_ts = sqrt(beta)*ΔK_phys
      overlap clamp
      p = beta1^(-3/2)*da_ts1(mapped)
      q = beta2^(-3/2)*da_ts2(mapped)
      result = p + R*(p-q)
    Returns matched_df and (optional) metrics vs GT.
    """
    R = (1 - (1 / beta1)) / ((1 / beta1) - (1 / beta2))

    w36 = ts1_curve.sort_values("del_K")
    w18 = ts2_curve.sort_values("del_K")

    x36_min, x36_max = w36["del_K"].min(), w36["del_K"].max()
    x18_min, x18_max = w18["del_K"].min(), w18["del_K"].max()

    # Choose physical grid
    if gt_df is not None and gt_xcol is not None and gt_ycol is not None:
        delK_phys_all = gt_df[gt_xcol].to_numpy(dtype=float)
        GT_all = gt_df[gt_ycol].to_numpy(dtype=float)
    else:
        xmin_phys = max(x36_min / np.sqrt(beta1), x18_min / np.sqrt(beta2))
        xmax_phys = min(x36_max / np.sqrt(beta1), x18_max / np.sqrt(beta2))
        delK_phys_all = np.linspace(xmin_phys, xmax_phys, phys_grid_n)
        GT_all = None

    # Overlap clamp (ensures no extrapolation)
    xmin_phys = max(
        delK_phys_all.min(),
        x36_min / np.sqrt(beta1),
        x18_min / np.sqrt(beta2)
    )
    xmax_phys = min(
        delK_phys_all.max(),
        x36_max / np.sqrt(beta1),
        x18_max / np.sqrt(beta2)
    )

    mask = (delK_phys_all >= xmin_phys) & (delK_phys_all <= xmax_phys)
    delK_phys = delK_phys_all[mask]
    GT = GT_all[mask] if GT_all is not None else None

    # Map to trial ΔK
    delK_ts1 = np.sqrt(beta1) * delK_phys
    delK_ts2 = np.sqrt(beta2) * delK_phys

    # Interpolate ANN-predicted trial curves at mapped points
    da_ts1 = np.interp(delK_ts1, w36["del_K"].to_numpy(), w36["y_pred"].to_numpy())
    da_ts2 = np.interp(delK_ts2, w18["del_K"].to_numpy(), w18["y_pred"].to_numpy())

    # Scale + blend
    p = (beta1 ** (-1.5)) * da_ts1
    q = (beta2 ** (-1.5)) * da_ts2
    result = p + R * (p - q)

    matched_df = pd.DataFrame({
        "del_K": delK_phys,
        "AI_W36_p": p,
        "AI_W18_q": q,
        "AI_similitude_W72": result,
    })

    if GT is not None:
        matched_df["GT_W72"] = GT
        r2v = r2_score(GT, result)
        rmse = float(np.sqrt(mean_squared_error(GT, result)))
        metrics = {"R": float(R), "r2_vs_gt": float(r2v), "rmse_vs_gt": float(rmse),
                   "xmin_phys": float(xmin_phys), "xmax_phys": float(xmax_phys)}
    else:
        metrics = {"R": float(R), "xmin_phys": float(xmin_phys), "xmax_phys": float(xmax_phys)}

    return matched_df, metrics


# ---------------- Load uploaded data ----------------
ts1_df = load_table(ts1_file)
ts2_df = load_table(ts2_file)
gt_df = load_table(gt_file)

# ---------------- Main layout ----------------
colL, colR = st.columns([1, 1.35], gap="large")

with colR:
    st.subheader("Similitude rule")
    if order == 1:
        st.latex(r"R_1 = \frac{1-\beta_1^{-1}}{\beta_1^{-1}-\beta_2^{-1}}")
        st.latex(
            r"\left(\frac{da}{dN}\right)_{\mathrm{1}}"
            r"=\beta_1^{-3/2}\left(\frac{da}{dN}\right)_{ts1}"
            r"+R_1\left[\beta_1^{-3/2}\left(\frac{da}{dN}\right)_{ts1}-\beta_2^{-3/2}\left(\frac{da}{dN}\right)_{ts2}\right]"
        )
        st.latex(r"\Delta K_{ts} = \sqrt{\beta}\,\Delta K_{\beta}")
    else:
        st.info("For MVP, we implement first-order (order = 1).")

with colL:
    st.subheader("Step 3: Upload + column mapping (experimental data)")
    st.write("Upload TS1, TS2 and optional GT (CSV/XLSX). Then pick x (ΔK) and y (da/dN) columns.")

    def mapping_ui(df, label, key_prefix):
        if df is None:
            st.info(f"Upload {label} to continue.")
            return None, None
        st.write(f"**{label} preview**")
        st.dataframe(df.head(10), use_container_width=True)
        cols = list(df.columns)
        xcol = st.selectbox(f"{label} x-column (ΔK)", cols, index=0, key=f"{key_prefix}_x")
        ycol = st.selectbox(f"{label} y-column (da/dN)", cols, index=min(1, len(cols)-1), key=f"{key_prefix}_y")
        return xcol, ycol

    ts1_xcol, ts1_ycol = mapping_ui(ts1_df, "TS1 (β1)", "ts1")
    ts2_xcol, ts2_ycol = mapping_ui(ts2_df, "TS2 (β2)", "ts2")
    gt_xcol, gt_ycol   = mapping_ui(gt_df,  "GT (optional)", "gt")

# -------- Tabs for Step 4 / Step 5 --------
tab4, tab5 = st.tabs(["Step 4 — ANN training", "Step 5 — Similitude prediction"])

with tab4:
    st.subheader("Step 4: Train ANN surrogates (MLPRegressor)")

    use_log = st.checkbox("Use log10 transform on ΔK and da/dN (recommended)", value=True)
    scaler_type = st.selectbox("Scaler type", ["standard", "minmax"], index=0)

    n_layers = st.number_input("Number of hidden layers", min_value=1, max_value=5, value=2, step=1)
    neurons_text = st.text_input("Neurons per layer (e.g., 64 or 64,32)", value="64,32")

    learning_rate_init = st.number_input("Learning rate", min_value=1e-6, value=0.005, format="%.6f")
    alpha = st.number_input("L2 regularization (alpha)", min_value=0.0, value=0.001, format="%.6f")

    test_size = st.slider("Test split (%)", min_value=10, max_value=40, value=20, step=5) / 100.0
    early_stopping = st.checkbox("Early stopping", value=True)

    train_btn = st.button("Train ANN for TS1 & TS2", type="primary")

    if train_btn:
        if ts1_df is None or ts2_df is None or ts1_xcol is None or ts2_xcol is None:
            st.error("Please upload TS1 and TS2 and select their columns before training.")
        else:
            try:
                hidden_layers = parse_hidden_layers(int(n_layers), neurons_text)
            except Exception as e:
                st.error(f"Hidden layer config error: {e}")
                hidden_layers = None

            if hidden_layers is not None:
                with st.spinner("Training TS1 ANN..."):
                    try:
                        m1, plot1, curve1, met1 = train_ann_curve(
                            ts1_df, ts1_xcol, ts1_ycol,
                            scaler_type=scaler_type,
                            use_log=use_log,
                            hidden_layers=hidden_layers,
                            learning_rate_init=learning_rate_init,
                            alpha=alpha,
                            test_size=test_size,
                            early_stopping=early_stopping
                        )
                        st.session_state["ts1_curve"] = curve1
                        st.session_state["ts1_metrics"] = met1
                        st.session_state["ts1_plot"] = plot1
                    except Exception as e:
                        st.error(f"TS1 training failed: {e}")

                with st.spinner("Training TS2 ANN..."):
                    try:
                        m2, plot2, curve2, met2 = train_ann_curve(
                            ts2_df, ts2_xcol, ts2_ycol,
                            scaler_type=scaler_type,
                            use_log=use_log,
                            hidden_layers=hidden_layers,
                            learning_rate_init=learning_rate_init,
                            alpha=alpha,
                            test_size=test_size,
                            early_stopping=early_stopping
                        )
                        st.session_state["ts2_curve"] = curve2
                        st.session_state["ts2_metrics"] = met2
                        st.session_state["ts2_plot"] = plot2
                    except Exception as e:
                        st.error(f"TS2 training failed: {e}")

                st.success("Training completed (if no errors shown).")

    # Re-render diagnostics every rerun (so they don't disappear)
    if "ts1_plot" in st.session_state:
        st.write("### TS1 ANN diagnostics")
        fig1 = render_ann_diagnostics(st.session_state["ts1_plot"], title="TS1")
        st.pyplot(fig1, clear_figure=True)
        st.write("TS1 metrics:", st.session_state.get("ts1_metrics", {}))

    if "ts2_plot" in st.session_state:
        st.write("### TS2 ANN diagnostics")
        fig2 = render_ann_diagnostics(st.session_state["ts2_plot"], title="TS2")
        st.pyplot(fig2, clear_figure=True)
        st.write("TS2 metrics:", st.session_state.get("ts2_metrics", {}))


with tab5:
    st.subheader("Step 5: First-order similitude prediction + plots + download")

    run_sim = st.button("Run similitude prediction", type="secondary")

    if run_sim:
        if order != 1:
            st.error("For MVP, set Order of similitude = 1.")
        elif ("ts1_curve" not in st.session_state) or ("ts2_curve" not in st.session_state):
            st.error("Train ANN first (Step 4) so we have predicted trial curves.")
        else:
            matched_df, sim_metrics = compute_first_order_similitude(
                st.session_state["ts1_curve"],
                st.session_state["ts2_curve"],
                beta1=beta1,
                beta2=beta2,
                gt_df=gt_df, gt_xcol=gt_xcol, gt_ycol=gt_ycol,
                phys_grid_n=2000
            )
            st.session_state["matched_df"] = matched_df
            st.session_state["sim_metrics"] = sim_metrics
            # reset plotting gate so user picks range again
            st.session_state["plot_ready"] = False
            st.success("Similitude prediction computed. Now select ΔK interval for plotting.")

    if "matched_df" in st.session_state:
        matched_df = st.session_state["matched_df"]
        sim_metrics = st.session_state.get("sim_metrics", {})

        st.write("Similitude metrics:", sim_metrics)
        st.write("Preview of results:")
        st.dataframe(matched_df.head(15), use_container_width=True)

        # Ask ΔK interval BEFORE plotting
        st.markdown("### Choose ΔK interval for plotting")
        xmin_data = float(matched_df["del_K"].min())
        xmax_data = float(matched_df["del_K"].max())

        x_min_plot, x_max_plot = st.slider(
            "ΔK plotting range",
            min_value=xmin_data,
            max_value=xmax_data,
            value=(xmin_data, xmax_data),
        )

        plot_now = st.button("Plot results in selected ΔK range")

        if plot_now:
            st.session_state["plot_ready"] = True
            st.session_state["x_min_plot"] = x_min_plot
            st.session_state["x_max_plot"] = x_max_plot

        if st.session_state.get("plot_ready", False):
            x_min_plot = st.session_state["x_min_plot"]
            x_max_plot = st.session_state["x_max_plot"]

            plot_df = matched_df[(matched_df["del_K"] >= x_min_plot) & (matched_df["del_K"] <= x_max_plot)]

            fig = plt.figure(figsize=(9, 5))
            plt.plot(plot_df["del_K"], plot_df["AI_W36_p"], label="AI_W36", linestyle="--")
            plt.plot(plot_df["del_K"], plot_df["AI_W18_q"], label="AI_W18", linestyle="--")

            if "GT_W72" in plot_df.columns:
                plt.plot(plot_df["del_K"], plot_df["GT_W72"], label="GT_W72", linestyle="-")

            plt.plot(plot_df["del_K"], plot_df["AI_similitude_W72"], label="AI_similitude_W72", linestyle="-.")

            plt.yscale("log")
            plt.xlabel("ΔK")
            plt.ylabel("da/dN (log scale)")
            plt.title(f"First-order similitude prediction (ΔK in [{x_min_plot:.3f}, {x_max_plot:.3f}])")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

        # Downloads unchanged (full matched_df)
        csv_bytes = matched_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="matched_results.csv",
            mime="text/csv"
        )

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            matched_df.to_excel(writer, index=False, sheet_name="matched")

        st.download_button(
            "Download results as Excel",
            data=buffer.getvalue(),
            file_name="matched_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
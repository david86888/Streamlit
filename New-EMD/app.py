import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import zipfile
import io
import os

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 350px;
        max-width: 350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Parameters")

data_mode = st.sidebar.segmented_control(
    "Data Source",
    # ["Synthetic Data", "Real-World Data", "Signal Comparison"],
    ["Synthetic Data", "Signal Comparison"],
    default="Synthetic Data"
)

st.sidebar.markdown("---")

h_start_input = st.sidebar.text_input("h_start", value="0.1")
try:
    h_start = float(h_start_input)
except:
    st.sidebar.error("Invalid input for h_start")
    st.stop()

h_min_input = st.sidebar.text_input("h_min", value="0.02")
try:
    h_min = float(h_min_input)
except:
    st.sidebar.error("Invalid input for h_min")
    st.stop()

a_input = st.sidebar.text_input("a (Python expression)", value="np.sqrt(2)")
try:
    a_val = float(eval(a_input, {"np": np, "sqrt": np.sqrt}))
except:
    st.sidebar.error("Invalid input for a")
    st.stop()

view_mode = "3 x 2"
num_real_columns = 1
custom_ylim_val = None

if data_mode == "Synthetic Data":
    # st.sidebar.markdown("---")
    st.sidebar.markdown("# Synthetic Data Options")
    # view_mode = st.sidebar.selectbox("Display Mode", ["3 x 2", "6 x 1"])

elif data_mode == "Real-World Data":
    # st.sidebar.markdown("---")
    st.sidebar.markdown("# Real-World Data Options")

    if 'real_cols_count' not in st.session_state:
        st.session_state.real_cols_count = 1

    st.session_state.real_cols_count = st.sidebar.number_input(
        "Number of Columns",
        min_value=1,
        step=1,
        value=st.session_state.real_cols_count
    )

    num_real_columns = int(st.session_state.real_cols_count)

elif data_mode == "Signal Comparison":
    # st.sidebar.markdown("---")
    st.sidebar.markdown("# Comparison Options")
    
    # view_mode = st.sidebar.selectbox("Display Mode", ["3 x 2", "6 x 1"])
    
    y_limit_input = st.sidebar.number_input("Y-Axis Limit (+/-)", value=4.0, step=0.1)
    custom_ylim_val = float(y_limit_input)


@st.cache_data
def emd_decompose(y_input, x, h_start, h_min, a):
    test_h = h_start
    test_cnt = 0
    while test_h >= h_min:
        test_cnt += 1
        test_h /= a
        if test_cnt > 100:
            st.error("Error: Iteration limit exceeded (>100). Please adjust parameters.")
            st.stop()

    residuals_list = []
    components = []
    current_residuals = y_input.copy()
    h_curr = h_start
    
    if len(x) > 1:
        dx = (x[-1] - x[0]) / (len(x) - 1)
    else:
        dx = 1.0

    while h_curr >= h_min:
        n_window = int(np.ceil(h_curr / dx))
        if n_window < 1: n_window = 1
        
        k_x = np.arange(-n_window, n_window + 1) * dx
        
        if h_curr == 0:
            weights = np.zeros_like(k_x)
            weights[len(weights)//2] = 1
        else:
            weights = 0.75 * (1 - (k_x / h_curr)**2)
            weights[weights < 0] = 0
            
        if weights.sum() > 0:
            weights = weights / weights.sum()

        numerator = fftconvolve(current_residuals, weights, mode='same')
        denominator = fftconvolve(np.ones_like(current_residuals), weights, mode='same')
        
        y_smooth = numerator / (denominator + 1e-10)
            
        components.append(y_smooth)
        current_residuals = current_residuals - y_smooth
        residuals_list.append(current_residuals.copy())
        h_curr /= a
        
    return components, residuals_list

@st.cache_data
def emd_decompose_local(y_input, x, h_start, h_min, a):
    residuals_list = []
    components = []
    current_residuals = y_input.copy()
    h_curr = h_start
    
    while h_curr >= h_min:
        n = len(x)
        y_smooth = np.zeros(n)
        
        for i in range(n):
            t = x[i]
            indices = np.where((x >= t - h_curr) & (x <= t + h_curr))[0]
            local_res = current_residuals[indices]
            
            if len(indices) > 0:
                u = (x[indices] - t) / h_curr
                weights = 0.75 * (1 - u**2)
                weights[weights < 0] = 0
                w_sum = weights.sum()
                if w_sum > 0:
                    weights = weights / w_sum
                    
                    def loss_function(m):
                        return np.sum(np.abs(local_res - m) * weights)
                    
                    result = minimize_scalar(loss_function)
                    if result.success:
                        y_smooth[i] = result.x
                else:
                    y_smooth[i] = current_residuals[i]
            else:
                 y_smooth[i] = current_residuals[i]

        components.append(y_smooth)
        current_residuals = current_residuals - y_smooth
        residuals_list.append(current_residuals.copy())
        h_curr /= a

    return components, residuals_list

import matplotlib.patches as patches

import matplotlib.patches as patches

@st.dialog("MSE Simulation Results", width="large")
def show_mse_window(y_clean_syn, x_syn, h_start, h_min, a_val, outlier_prob, outlier_mul, target_sigma, current_iter):
    st.write("Running MSE Simulation (B=100)...")
    progress_bar = st.progress(0)
    
    B = 100
    sigma_values = np.linspace(0, 2 * target_sigma, 11)
    
    n_points = len(x_syn)
    
    if len(x_syn) > 1:
        dx = (x_syn[-1] - x_syn[0]) / (len(x_syn) - 1)
    else:
        dx = 1.0

    components_clean_ref = []
    current_residuals_clean_ref = y_clean_syn.copy()
    h_curr = h_start

    while h_curr >= h_min:
        n_window = int(np.ceil(h_curr / dx))
        if n_window < 1: n_window = 1
        
        k_x = np.arange(-n_window, n_window + 1) * dx
        
        if h_curr == 0:
            weights = np.zeros_like(k_x)
            weights[len(weights)//2] = 1
        else:
            weights = 0.75 * (1 - (k_x / h_curr)**2)
            weights[weights < 0] = 0
            
        if weights.sum() > 0:
            weights = weights / weights.sum()

        denominator = fftconvolve(np.ones_like(current_residuals_clean_ref), weights, mode='same')
        denominator_safe = denominator + 1e-10
        numerator = fftconvolve(current_residuals_clean_ref, weights, mode='same')
        y_smooth = numerator / denominator_safe
        
        components_clean_ref.append(y_smooth)
        current_residuals_clean_ref = current_residuals_clean_ref - y_smooth
        h_curr /= a_val

    n_iterations = len(components_clean_ref)
    sse_results = np.zeros((n_iterations, len(sigma_values)))
    
    for j, sigma_val in enumerate(sigma_values):
        for b in range(B):
            np.random.seed(b + j * B)
            
            is_outlier = np.random.rand(n_points) < outlier_prob
            noise_b = np.zeros(n_points)
            
            if sigma_val > 0:
                noise_b[~is_outlier] = np.random.normal(scale=sigma_val, size=np.sum(~is_outlier))
                outliers_b = np.random.exponential(scale=sigma_val * outlier_mul, size=np.sum(is_outlier))
                outliers_b *= np.random.choice([-1, 1], size=len(outliers_b))
                noise_b[is_outlier] = outliers_b
            
            y_noisy_b = y_clean_syn + noise_b
            
            components_noisy_b = []
            current_residuals_noisy_b = y_noisy_b.copy()
            h_curr_b = h_start
            
            while h_curr_b >= h_min:
                n_window = int(np.ceil(h_curr_b / dx))
                if n_window < 1: n_window = 1
                
                k_x = np.arange(-n_window, n_window + 1) * dx
                
                if h_curr_b == 0:
                    weights = np.zeros_like(k_x)
                    weights[len(weights)//2] = 1
                else:
                    weights = 0.75 * (1 - (k_x / h_curr_b)**2)
                    weights[weights < 0] = 0
                    
                if weights.sum() > 0:
                    weights = weights / weights.sum()

                denominator = fftconvolve(np.ones_like(current_residuals_noisy_b), weights, mode='same')
                denominator_safe = denominator + 1e-10
                numerator = fftconvolve(current_residuals_noisy_b, weights, mode='same')
                y_smooth = numerator / denominator_safe
                
                components_noisy_b.append(y_smooth)
                current_residuals_noisy_b = current_residuals_noisy_b - y_smooth
                h_curr_b /= a_val
            
            n_comps = min(len(components_clean_ref), len(components_noisy_b))
            
            for k in range(n_comps):
                diff = components_noisy_b[k] - components_clean_ref[k]
                mse_k = np.mean(diff**2)
                sse_results[k, j] += mse_k
                
        sse_results[:, j] /= B
        progress_bar.progress((j + 1) / len(sigma_values))

    df_sse = pd.DataFrame(
        sse_results, 
        index=[f"{i}" for i in range(n_iterations)],
        columns=[f"{s:.2f}" for s in sigma_values]
    )

    col_table, col_plot = st.columns([1, 1])

    with col_table:
        st.markdown("### MSE Table")
        
        def highlight_target(x):
            df_color = pd.DataFrame('', index=x.index, columns=x.columns)
            # Find the column closest to target_sigma (index 5)
            target_col_name = df_sse.columns[5]
            target_row_name = f"{current_iter - 1}"
            
            if target_col_name in df_color.columns and target_row_name in df_color.index:
                df_color.at[target_row_name, target_col_name] = 'border: 2px solid red; color: red; font-weight: bold;'
            return df_color

        st.dataframe(df_sse.style.apply(highlight_target, axis=None).format("{:.4f}"))

    with col_plot:
        st.markdown("### MSE Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(sse_results, aspect='auto', cmap='viridis')
        
        target_col_idx = 5
        target_row_idx = current_iter - 1
        
        if 0 <= target_row_idx < n_iterations:
            rect = patches.Rectangle((target_col_idx - 0.5, target_row_idx - 0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        fig.colorbar(im)
        ax.set_ylabel('Iteration Order')
        ax.set_xlabel(r'Noise $\sigma$')
        ax.set_title(f'MSE, a={a_val:.3f}')
        
        ax.set_yticks(np.arange(n_iterations))
        ax.set_xticks(np.arange(len(sigma_values)))
        ax.set_xticklabels([f"{s:.2f}" for s in sigma_values], rotation=45, ha='right')
        
        st.pyplot(fig)

def generate_figure(i, mode, x_axis, 
                    input_noisy, input_clean, 
                    comps_noisy, comps_clean, 
                    res_noisy, res_clean, col_num=1, custom_ylim=None):
    
    if i == 0:
        curr_input_noisy = input_noisy
        curr_input_clean = input_clean if input_clean is not None else input_noisy 
    else:
        curr_input_noisy = res_noisy[i-1]
        curr_input_clean = res_clean[i-1] if res_clean is not None else None

    if custom_ylim is not None:
        ylim_lower = -abs(custom_ylim)
        ylim_upper = abs(custom_ylim)
    else:
        valid_data = input_noisy[~np.isnan(input_noisy)]
        if len(valid_data) > 0:
            y_min = np.min(valid_data)
            y_max = np.max(valid_data)
        else:
            y_min, y_max = 0, 1
            
        y_range = y_max - y_min
        if y_range == 0: y_range = 1.0
        ylim_lower = y_min - 0.1 * y_range
        ylim_upper = y_max + 0.1 * y_range

    if mode == "6 x 1":
        fig, axes = plt.subplots(6, 1, figsize=(10, 9), sharex=True)
        
        data_list = [
            curr_input_noisy,
            curr_input_clean if curr_input_clean is not None else np.nan * np.zeros_like(x_axis),
            comps_noisy[i],
            comps_clean[i] if comps_clean is not None else np.nan * np.zeros_like(x_axis),
            res_noisy[i],
            res_clean[i] if res_clean is not None else np.nan * np.zeros_like(x_axis)
        ]
        
        colors = ['gray', 'black', 'royalblue', 'royalblue', 'salmon', 'salmon']
        alphas = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0]
        widths = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]

        for ax, data, col, alph, wid in zip(axes, data_list, colors, alphas, widths):
            if not np.all(np.isnan(data)):
                ax.plot(x_axis, data, color=col, linewidth=wid, alpha=alph)
            ax.set_ylim(ylim_lower, ylim_upper)
    
    elif mode == "3 x 1":
        fig, axes = plt.subplots(3, 1, figsize=(10, 2+col_num*3), sharex=True)
        
        axes[0].plot(x_axis, curr_input_noisy, color='gray', linewidth=col_num*1.5, alpha=0.5)
        axes[1].plot(x_axis, comps_noisy[i], color='royalblue', linestyle='-', linewidth=col_num*1.5)
        axes[2].plot(x_axis, res_noisy[i], color='salmon', linestyle='-', linewidth=col_num*1.5)
        
        for ax in axes:
             if custom_ylim is not None:
                ax.set_ylim(ylim_lower, ylim_upper)

    else: 
        fig, axes = plt.subplots(3, 2, figsize=(10, 4.5), sharex=True)

        axes[0, 0].plot(x_axis, curr_input_noisy, color='gray', linewidth=1.5, alpha=0.5)
        axes[1, 0].plot(x_axis, comps_noisy[i], color='royalblue', linestyle='-', linewidth=1.5)
        axes[2, 0].plot(x_axis, res_noisy[i], color='salmon', linestyle='-', linewidth=1.5)

        if input_clean is not None:
            if i == 0:
                axes[0, 1].plot(x_axis, curr_input_clean, color='black', linestyle='-', linewidth=1.5)
            elif curr_input_clean is not None:
                axes[0, 1].plot(x_axis, curr_input_clean, color='black', linestyle='-', linewidth=1.5)
            
            if comps_clean is not None:
                axes[1, 1].plot(x_axis, comps_clean[i], color='royalblue', linestyle='-', linewidth=1.5)
            if res_clean is not None:
                axes[2, 1].plot(x_axis, res_clean[i], color='salmon', linestyle='-', linewidth=1.5)

        for ax_row in axes:
            for ax in ax_row:
                ax.set_ylim(ylim_lower, ylim_upper)
    
    if hasattr(x_axis, 'dtype') and np.issubdtype(x_axis.dtype, np.datetime64):
        fig.autofmt_xdate()
    
    plt.tight_layout()
    return fig

def render_controls(key_suffix, n_iters, h_start, a_val, container):
    iter_key = f'iter_index_{key_suffix}'
    if iter_key not in st.session_state:
        st.session_state[iter_key] = 1

    with container:
        col_prev, col_slide, col_next = st.columns([1, 4, 1.5])

        with col_prev:
            st.button(
                "◀",
                key=f"btn_prev_{key_suffix}",
                on_click=lambda: st.session_state.__setitem__(
                    iter_key,
                    max(1, int(st.session_state.get(iter_key, 1)) - 1)
                )
            )

        with col_slide:
            selected_iter = st.slider(
                "Select Iteration",
                1,
                n_iters,
                key=iter_key,
                label_visibility="collapsed",
            )

        with col_next:
            st.button(
                "▶",
                key=f"btn_next_{key_suffix}",
                on_click=lambda: st.session_state.__setitem__(
                    iter_key,
                    min(n_iters, int(st.session_state.get(iter_key, 1)) + 1)
                )
            )
    
    current_h = h_start / (a_val ** (selected_iter - 1))
    return selected_iter, current_h

if data_mode == "Synthetic Data":
    st.header("Synthetic Data Analysis")
    
    noise_str = st.sidebar.text_input("sigma (noise_std)", value="0.2", key="noise_input")
    try:
        noise_std = float(noise_str)
    except:
        st.sidebar.error("Invalid input for sigma")
        st.stop()
    
    outlier_prob = st.sidebar.number_input("Outlier Probability", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    outlier_mul = st.sidebar.number_input("Outlier Multiplier", value=5.0)
    outlier_dist = st.sidebar.selectbox("Outlier Distribution", ["exponential"])
    
    n_points = 2000
    eps = 1e-3
    x_syn = np.linspace(eps, 1.0, n_points)
    y_clean_syn = (1 + 0.5 * np.sin(4 * np.pi * x_syn)) * np.sin(2 * np.pi * (6 * x_syn + 12 * x_syn**3))

    np.random.seed(42)
    
    is_outlier = np.random.rand(n_points) < outlier_prob
    noise = np.zeros(n_points)
    
    noise[~is_outlier] = np.random.normal(scale=noise_std, size=np.sum(~is_outlier))
    
    if outlier_dist == "exponential":
        outliers = np.random.exponential(scale=noise_std * outlier_mul, size=np.sum(is_outlier))
        outliers *= np.random.choice([-1, 1], size=len(outliers))
        noise[is_outlier] = outliers

    y_noisy_syn = y_clean_syn + noise

    with st.spinner('Decomposing synthetic signals...'):
        components_syn, residuals_list_syn = emd_decompose(y_noisy_syn, x_syn, h_start, h_min, a_val)
        components_clean_syn, residuals_list_clean_syn = emd_decompose(y_clean_syn, x_syn, h_start, h_min, a_val)

    n_iterations_syn = len(components_syn)

    if n_iterations_syn > 0:
        header_syn = st.container()
        with header_syn:
            c1, c2, c3, c4 = st.columns([8, 2, 4, 6], vertical_alignment="center")
            
            sel_iter_syn, curr_h_syn = render_controls("syn", n_iterations_syn, h_start, a_val, c4)

            c1.markdown(f"# Total Iterations: {n_iterations_syn}")
            c2.metric("Iteration", f"{sel_iter_syn}")
            c3.metric("h (bandwidth)", f"{curr_h_syn:.4f}")

        st.markdown("\n")
        fig_syn = generate_figure(
            sel_iter_syn - 1, view_mode, x_syn,
            y_noisy_syn, y_clean_syn,
            components_syn, components_clean_syn,
            residuals_list_syn, residuals_list_clean_syn
        )
        st.pyplot(fig_syn)

        st.markdown("\n")
        st.progress((sel_iter_syn - 1) / max(n_iterations_syn - 1, 1))

        col_zip, col_mse = st.columns([1, 1], vertical_alignment="center")
        
        with col_zip:
            if st.button("Save Synthetic Iterations to ZIP", key="save_syn", use_container_width=True):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for k in range(n_iterations_syn):
                        f = generate_figure(k, view_mode, x_syn, y_noisy_syn, y_clean_syn, components_syn, components_clean_syn, residuals_list_syn, residuals_list_clean_syn)
                        img_bytes = io.BytesIO()
                        f.savefig(img_bytes, format='png', dpi=144, transparent=True)
                        plt.close(f)
                        zf.writestr(f"synthetic_iteration_{k:03d}.png", img_bytes.getvalue())
                
                st.download_button("Download ZIP", zip_buffer.getvalue(), "synthetic_iterations.zip", "application/zip", key="dl_zip_btn")

        with col_mse:
            if st.button("Show MSE Analysis", key="show_mse", use_container_width=True):
                # Added noise_std and sel_iter_syn to the call
                show_mse_window(y_clean_syn, x_syn, h_start, h_min, a_val, outlier_prob, outlier_mul, noise_std, sel_iter_syn)

elif data_mode == "Real-World Data":
    st.header("Real-World Data Analysis")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'Cap_10.csv')
    
    try:
        if not os.path.exists(file_path):
            st.warning("File 'Cap_10.csv' not found.")
        
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df = np.log10(df)
        col_list = df.columns.tolist()

        selected_cols_data = []
        for i in range(num_real_columns):
            key_name = f"real_asset_select_{i}"
            default_val = col_list[i % len(col_list)]
            current_val = st.session_state.get(key_name, default_val)
            
            if current_val in df.columns:
                y_r = df[current_val].values
                x_calc = np.linspace(0, 1, len(y_r))
                x_plot = df.index
                
                selected_cols_data.append({
                    "name": current_val,
                    "y": y_r,
                    "x_calc": x_calc,
                    "x_plot": x_plot
                })

        max_iters = 0
        n = len(df.index)
        if selected_cols_data:
            with st.spinner('Pre-calculating iterations...'):
                for item in selected_cols_data:
                    c_tmp, r_tmp = emd_decompose(item["y"], item["x_calc"], h_start, h_min, a_val)
                    item["comps"] = c_tmp
                    item["res"] = r_tmp
                    item["iters"] = len(c_tmp)
                    if item["iters"] > max_iters:
                        max_iters = item["iters"]

        if max_iters > 0:
            header_real = st.container()
            with header_real:
                c1, c2, c3, c4 = st.columns([8, 2, 6, 6], vertical_alignment="bottom")
                sel_iter_real, curr_h_real = render_controls("real_shared", max_iters, h_start, a_val, c4)
                
                c1.markdown(f"# Max Iterations: {max_iters}")
                c2.metric("Iteration", f"{sel_iter_real}")
                c3.metric("h (bandwidth)", f"{curr_h_real:.3f}  ({curr_h_real * n / 24:.0f} days)")

        # st.markdown("---\n")
        real_cols_ui = st.columns(num_real_columns)
        
        for i, col_ui in enumerate(real_cols_ui):
            with col_ui:
                default_idx = i % len(col_list)
                selected_col = st.selectbox(
                    f"Select Column {i+1}", 
                    col_list, 
                    index=default_idx, 
                    key=f"real_asset_select_{i}"
                )
                
                # st.markdown("---")

                if selected_col and max_iters > 0:
                    data_item = next((item for item in selected_cols_data if item["name"] == selected_col), None)
                    
                    if data_item:
                        actual_idx = min(sel_iter_real, data_item["iters"]) - 1
                        if actual_idx < 0: actual_idx = 0
                        
                        fig_real = generate_figure(
                            actual_idx, "3 x 1", data_item["x_plot"],
                            data_item["y"], None,
                            data_item["comps"], None,
                            data_item["res"], None,
                            col_num=num_real_columns
                        )
                        st.pyplot(fig_real)

    except Exception as e:
        st.error(f"Error loading or processing data: {e}")

elif data_mode == "Signal Comparison":
    st.header("Signal Comparison (Bird vs Mavic)")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_bird = os.path.join(current_dir, "bird_roi_abs.csv")
    file_mavic = os.path.join(current_dir, "mavic_roi_abs.csv")
    
    try:
        if not os.path.exists(file_bird) or not os.path.exists(file_mavic):
            st.error("Error: Required CSV files (bird_roi_abs.csv, mavic_roi_abs.csv) not found.")
        else:
            df1 = pd.read_csv(file_bird, index_col=0)
            df2 = pd.read_csv(file_mavic, index_col=0)
            
            df1 = (df1 - df1.mean()) / df1.std()
            df2 = (df2 - df2.mean()) / df2.std()
            
            y_clean = df1.iloc[:, 0].to_numpy() 
            y_noisy = df2.iloc[:, 0].to_numpy() 
            
            x_plot = df1.index.to_numpy()
            x_calc = (x_plot - x_plot.min()) / (x_plot.max() - x_plot.min())
            
            with st.spinner('Decomposing signals using Local EMD (minimize_scalar)... This may take a while.'):
                components_comp, residuals_list_comp = emd_decompose_local(y_noisy, x_calc, h_start, h_min, a_val)
                components_clean_comp, residuals_list_clean_comp = emd_decompose_local(y_clean, x_calc, h_start, h_min, a_val)
            
            n_iterations_comp = len(components_comp)
            
            if n_iterations_comp > 0:
                header_comp = st.container()
                with header_comp:
                    c1, c2, c3, c4 = st.columns([8, 2, 4, 6], vertical_alignment="center")
                    sel_iter_comp, curr_h_comp = render_controls("comp", n_iterations_comp, h_start, a_val, c4)
                    
                    c1.markdown(f"# Total Iterations: {n_iterations_comp}")
                    c2.metric("Iteration", f"{sel_iter_comp}")
                    c3.metric("h (bandwidth)", f"{curr_h_comp:.4f}")
                
                st.markdown("\n")
                
                fig_comp = generate_figure(
                    sel_iter_comp - 1, view_mode, x_plot,
                    y_noisy, y_clean,
                    components_comp, components_clean_comp,
                    residuals_list_comp, residuals_list_clean_comp,
                    col_num=1,
                    custom_ylim=custom_ylim_val
                )
                st.pyplot(fig_comp)
                
                st.markdown("\n")
                st.progress((sel_iter_comp - 1) / max(n_iterations_comp - 1, 1))

    except Exception as e:
        st.error(f"Error in Signal Comparison: {e}")
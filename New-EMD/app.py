import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import zipfile
import io
import os

st.set_page_config(layout="wide")

st.sidebar.title("Parameters")

data_mode = st.sidebar.segmented_control(
    "Data Source",
    ["Synthetic Data", "Real-World Data"],
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

if data_mode == "Synthetic Data":
    st.sidebar.markdown("---")
    st.sidebar.markdown("# Synthetic Data Options")
    view_mode = st.sidebar.selectbox("Display Mode", ["3 x 2", "6 x 1"])

elif data_mode == "Real-World Data":
    st.sidebar.markdown("---")
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

def generate_figure(i, mode, x_axis, 
                    input_noisy, input_clean, 
                    comps_noisy, comps_clean, 
                    res_noisy, res_clean, col_num=1):
    
    if i == 0:
        curr_input_noisy = input_noisy
        curr_input_clean = input_clean if input_clean is not None else input_noisy 
    else:
        curr_input_noisy = res_noisy[i-1]
        curr_input_clean = res_clean[i-1] if res_clean is not None else None

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
        
        colors = ['gray', 'black', 'salmon', 'salmon', 'salmon', 'salmon']
        alphas = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0]
        widths = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]

        for ax, data, col, alph, wid in zip(axes, data_list, colors, alphas, widths):
            if not np.all(np.isnan(data)):
                ax.plot(x_axis, data, color=col, linewidth=wid, alpha=alph)
            ax.set_ylim(ylim_lower, ylim_upper)
    
    elif mode == "3 x 1":
        fig, axes = plt.subplots(3, 1, figsize=(10, 2+col_num*3), sharex=True)
        
        axes[0].plot(x_axis, curr_input_noisy, color='gray', linewidth=col_num*1.5, alpha=0.5)
        axes[1].plot(x_axis, comps_noisy[i], color='salmon', linestyle='-', linewidth=col_num*1.5)
        axes[2].plot(x_axis, res_noisy[i], color='salmon', linestyle='-', linewidth=col_num*1.5)

        # for ax in axes:
        #     ax.set_ylim(ylim_lower, ylim_upper)

    else: # 3 x 2
        fig, axes = plt.subplots(3, 2, figsize=(10, 4.5), sharex=True)

        axes[0, 0].plot(x_axis, curr_input_noisy, color='gray', linewidth=1.5, alpha=0.5)
        axes[1, 0].plot(x_axis, comps_noisy[i], color='salmon', linestyle='-', linewidth=1.5)
        axes[2, 0].plot(x_axis, res_noisy[i], color='salmon', linestyle='-', linewidth=1.5)

        if input_clean is not None:
            if i == 0:
                axes[0, 1].plot(x_axis, curr_input_clean, color='black', linestyle='-', linewidth=1.5)
            elif curr_input_clean is not None:
                axes[0, 1].plot(x_axis, curr_input_clean, color='black', linestyle='-', linewidth=1.5)
            
            if comps_clean is not None:
                axes[1, 1].plot(x_axis, comps_clean[i], color='salmon', linestyle='-', linewidth=1.5)
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
    
    n_points = 2000
    eps = 1e-3
    x_syn = np.linspace(eps, 1.0, n_points)
    y_clean_syn = (1 + 0.5 * np.sin(4 * np.pi * x_syn)) * np.sin(2 * np.pi * (6 * x_syn + 12 * x_syn**3))

    np.random.seed(42)
    noise = np.random.normal(scale=noise_std, size=n_points)
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

        if st.button("Save Synthetic Iterations to ZIP", key="save_syn"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for k in range(n_iterations_syn):
                    f = generate_figure(k, view_mode, x_syn, y_noisy_syn, y_clean_syn, components_syn, components_clean_syn, residuals_list_syn, residuals_list_clean_syn)
                    img_bytes = io.BytesIO()
                    f.savefig(img_bytes, format='png', dpi=144, transparent=True)
                    plt.close(f)
                    zf.writestr(f"synthetic_iteration_{k:03d}.png", img_bytes.getvalue())
            
            st.download_button("Download ZIP", zip_buffer.getvalue(), "synthetic_iterations.zip", "application/zip")

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
                c1, c2, c3, c4 = st.columns([8, 2, 4, 6], vertical_alignment="bottom")
                sel_iter_real, curr_h_real = render_controls("real_shared", max_iters, h_start, a_val, c4)
                
                c1.markdown(f"# Max Iterations: {max_iters}")
                c2.metric("Iteration", f"{sel_iter_real}")
                c3.metric("h (bandwidth)", f"{curr_h_real:.4f}")

        st.markdown("---\n")
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
                
                st.markdown("---")

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
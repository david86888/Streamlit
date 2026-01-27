import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import zipfile
import io

st.set_page_config(layout="wide")

st.sidebar.title("Parameters")

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

a_input = st.sidebar.text_input("a (Python expression, e.g., np.sqrt(2))", value="np.sqrt(2)")
try:
    a_val = float(eval(a_input, {"np": np, "sqrt": np.sqrt}))
except:
    st.sidebar.error("Invalid input for a")
    st.stop()

noise_std = st.sidebar.number_input("sigma (noise_std)", value=0.2, step=0.05)
view_mode = st.sidebar.selectbox("Display Mode", ["3 x 2", "6 x 1"])

n_points = 2000
eps = 1e-3

x = np.linspace(eps, 1.0, n_points)
y_clean = (1 + 0.5 * np.sin(4 * np.pi * x)) * np.sin(2 * np.pi * (6 * x + 12 * x**3))

np.random.seed(42)
noise = np.random.normal(scale=noise_std, size=n_points)
y_noisy = y_clean + noise

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
    
    dx = x[1] - x[0]

    while h_curr >= h_min:
        n_window = int(np.ceil(h_curr / dx))
        k_x = np.arange(-n_window, n_window + 1) * dx
        
        weights = 0.75 * (1 - (k_x / h_curr)**2)
        weights[weights < 0] = 0
        weights = weights / weights.sum()

        numerator = fftconvolve(current_residuals, weights, mode='same')
        denominator = fftconvolve(np.ones_like(current_residuals), weights, mode='same')
        
        y_smooth = numerator / (denominator + 1e-10)
            
        components.append(y_smooth)
        current_residuals = current_residuals - y_smooth
        residuals_list.append(current_residuals.copy())
        h_curr /= a
        
    return components, residuals_list

with st.spinner('Decomposing signals...'):
    components, residuals_list = emd_decompose(y_noisy, x, h_start, h_min, a_val)
    components_clean, residuals_list_clean = emd_decompose(y_clean, x, h_start, h_min, a_val)

n_iterations = len(components)

with st.spinner('Decomposing signals...'):
    components, residuals_list = emd_decompose(y_noisy, x, h_start, h_min, a_val)
    components_clean, residuals_list_clean = emd_decompose(y_clean, x, h_start, h_min, a_val)

n_iterations = len(components)

def generate_figure(i, mode):
    if i == 0:
        curr_input_noisy = y_noisy
        curr_input_clean = y_clean
    else:
        curr_input_noisy = residuals_list[i-1]
        curr_input_clean = residuals_list_clean[i-1]

    if mode == "6 x 1":
        fig, axes = plt.subplots(6, 1, figsize=(10, 18), sharex=True)
        
        data_list = [
            curr_input_noisy,
            curr_input_clean,
            components[i],
            components_clean[i],
            residuals_list[i],
            residuals_list_clean[i]
        ]
        
        colors = ['gray', 'black', 'salmon', 'salmon', 'salmon', 'salmon']
        alphas = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0]
        widths = [1.5, 2.0, 2.0, 2.0, 2.0, 2.0]

        for ax, data, col, alph, wid in zip(axes, data_list, colors, alphas, widths):
            ax.plot(x, data, color=col, linewidth=wid, alpha=alph)
            ax.set_ylim(-2.5, 2.5)

    else:
        fig, axes = plt.subplots(3, 2, figsize=(16,6), sharex=True)

        axes[0, 0].plot(x, curr_input_noisy, color='gray', linewidth=1.5, alpha=0.5)
        axes[0, 1].plot(x, curr_input_clean, color='black', linestyle='-', linewidth=2)

        axes[1, 0].plot(x, components[i], color='salmon', linestyle='-', linewidth=2)
        axes[1, 1].plot(x, components_clean[i], color='salmon', linestyle='-', linewidth=2)

        axes[2, 0].plot(x, residuals_list[i], color='salmon', linestyle='-', linewidth=2)
        axes[2, 1].plot(x, residuals_list_clean[i], color='salmon', linestyle='-', linewidth=2)

        for ax_row in axes:
            for ax in ax_row:
                ax.set_ylim(-2.5, 2.5)
    
    plt.tight_layout()
    return fig

if n_iterations > 0:
    st.sidebar.markdown("---")
    
    if 'iter_index' not in st.session_state:
        st.session_state.iter_index = 1

    col_prev, col_slide, col_next = st.sidebar.columns([1, 4, 1])

    with col_prev:
        st.button(
            "◀",
            key="iter_prev",
            on_click=lambda: st.session_state.__setitem__(
                'iter_index',
                max(1, int(st.session_state.get('iter_index', 1)) - 1)
            )
        )

    with col_slide:
        selected_iter = st.slider(
            "Select Iteration to View",
            1,
            n_iterations,
            key="iter_index",
            label_visibility="collapsed",
        )

    with col_next:
        st.button(
            "▶",
            key="iter_next",
            on_click=lambda: st.session_state.__setitem__(
                'iter_index',
                min(n_iterations, int(st.session_state.get('iter_index', 1)) + 1)
            )
        )

    st.sidebar.markdown("---")

    if st.sidebar.button("Save All Iterations to ZIP"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for k in range(n_iterations):
                fig = generate_figure(k, view_mode)
                img_bytes = io.BytesIO()
                fig.savefig(img_bytes, format='png', dpi=144, transparent=True)
                plt.close(fig)
                zf.writestr(f"iteration_{k:03d}.png", img_bytes.getvalue())
        
        st.sidebar.download_button(
            label="Download ZIP",
            data=zip_buffer.getvalue(),
            file_name="iterations.zip",
            mime="application/zip"
        )

    current_h = h_start / (a_val ** (selected_iter - 1))

    if selected_iter % 100 in [11, 12, 13]:
        iter_suffix = "th"
    elif selected_iter % 10 == 1:
        iter_suffix = "st"
    elif selected_iter % 10 == 2:
        iter_suffix = "nd"
    elif selected_iter % 10 == 3:
        iter_suffix = "rd"
    else:
        iter_suffix = "th"
    iter_label = f"{selected_iter}{iter_suffix}"

    header = st.container()
    with header:
        tcol, m1, m2 = st.columns([6, 3, 3])
        with tcol:
            st.markdown(f"# Total Iterations: {n_iterations}")
        with m1:
            st.metric("Iteration", f"{iter_label}")
        with m2:
            st.metric("h (bandwidth)", f"{current_h:.4f}")

        denom = max(n_iterations - 1, 1)
        progress = (selected_iter - 1) / denom
        st.progress(progress)
        st.caption("Iteration progress")

    fig = generate_figure(selected_iter - 1, view_mode)
    st.pyplot(fig)
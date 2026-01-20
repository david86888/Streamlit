import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import zipfile
import io

st.set_page_config(layout="wide")

st.sidebar.title("Parameters")
h_start = st.sidebar.number_input("h_start", value=0.1, step=0.01, format="%.2f")
h_min = st.sidebar.number_input("h_min", value=0.02, step=0.01, format="%.2f")

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
    
    while h_curr >= h_min:
        n = len(x)
        y_smooth = np.zeros(n)
        
        for i in range(n):
            t = x[i]
            
            indices = np.where((x >= t - h_curr) & (x <= t + h_curr))[0]
            local_res = current_residuals[indices]
            
            u = (x[indices] - t) / h_curr
            weights = 0.75 * (1 - u**2)
            weights = weights / (weights.sum())
            
            def loss_function(m):
                return np.sum(np.abs(local_res - m) * weights)
            result = minimize_scalar(loss_function)
            
            if result.success:
                y_smooth[i] = result.x
            
        components.append(y_smooth)
        current_residuals = current_residuals - y_smooth
        residuals_list.append(current_residuals.copy())
        h_curr /= a
        
    return components, residuals_list

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

    if 'iter_index' not in st.session_state:
        st.session_state.iter_index = 0

    col_info, col_prev, col_slider, col_next = st.columns([5, 1, 8, 1])

    with col_info:
        st.markdown(f"# Total Iterations: {n_iterations}")

    with col_prev:
        if st.button("◀"):
            st.session_state.iter_index = max(1, st.session_state.iter_index - 1)
            
    with col_next:
        if st.button("▶"):
            st.session_state.iter_index = min(n_iterations, st.session_state.iter_index + 1)

    with col_slider:
        selected_iter = st.slider("Select Iteration to View", 1, n_iterations, key="iter_index")
    
    fig = generate_figure(selected_iter - 1, view_mode)
    st.pyplot(fig)
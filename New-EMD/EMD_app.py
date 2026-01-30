import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.sidebar.title("Parameters")
h_start = st.sidebar.number_input("h_start", value=0.030, step=0.001, format="%.4f")
h_min = st.sidebar.number_input("h_min", value=0.020, step=0.001, format="%.4f")
a_val = st.sidebar.number_input("a", value=np.sqrt(2), format="%.4f")
noise_std = st.sidebar.number_input("sigma (noise_std)", value=0.2, step=0.05)

n_points = 2000
eps = 1e-3

x = np.linspace(eps, 1.0, n_points)
y_clean = (1 + 0.5 * np.sin(4 * np.pi * x)) * np.sin(2 * np.pi * (6 * x + 12 * x**3))

np.random.seed(42)
noise = np.random.normal(scale=noise_std, size=n_points)
y_noisy = y_clean + noise

col1, col2 = st.columns(2)
with col1:
    st.write("Clean Signal")
    fig1, ax1 = plt.subplots(figsize=(10, 2))
    ax1.plot(x, y_clean, color='gray', linewidth=2, alpha=0.5)
    ax1.set_ylim(-2.5, 2.5)
    st.pyplot(fig1)

with col2:
    st.write("Noisy Signal")
    fig2, ax2 = plt.subplots(figsize=(10, 2))
    ax2.plot(x, y_noisy, color='gray', linewidth=2, alpha=0.5)
    ax2.set_ylim(-2.5, 2.5)
    st.pyplot(fig2)

@st.cache_data
def emd_decompose(y_input, x, h_start, h_min, a):
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
st.write(f"Total Iterations: {n_iterations}")

if n_iterations > 0:
    selected_iter = st.slider("Select Iteration to View", 0, n_iterations - 1, 0)
    
    i = selected_iter
    fig, axes = plt.subplots(6, 1, figsize=(16, 12), sharex=True)

    if i == 0:
        current_input_noisy = y_noisy
        current_input_clean = y_clean
    else:
        current_input_noisy = residuals_list[i-1]
        current_input_clean = residuals_list_clean[i-1]

    ax = axes[0]
    ax.plot(x, current_input_noisy, color='gray', linewidth=1.5, alpha=0.5)
    ax.set_ylim(-2.5, 2.5)

    ax = axes[1]
    ax.plot(x, current_input_clean, color='black', linestyle='-', linewidth=2)
    ax.set_ylim(-2.5, 2.5)

    ax = axes[2]
    ax.plot(x, components[i], color='salmon', linestyle='-', linewidth=3)
    ax.set_ylim(-2.5, 2.5)

    ax = axes[3]
    ax.plot(x, components_clean[i], color='salmon', linestyle='-', linewidth=3)
    ax.set_ylim(-2.5, 2.5)

    ax = axes[4]
    ax.plot(x, residuals_list[i], color='royalblue', linestyle='-', linewidth=3)
    ax.set_ylim(-2.5, 2.5)

    ax = axes[5]
    ax.plot(x, residuals_list_clean[i], color='royalblue', linestyle='-', linewidth=3)
    ax.set_ylim(-2.5, 2.5)

    plt.tight_layout()
    st.pyplot(fig)
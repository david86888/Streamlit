import streamlit as st
import os
from PIL import Image

# 設定結果資料夾的基礎路徑
base_path = "/Users/johnnyjheng/Documents/研究所/論文/20240722 JHE TEN Hae FRM@TW/Code/20240809 FRM-Taiwan"

# 設定公司類型選項和窗口選項
companies = ["Cap", "Ele", "Fin", "FinEle"]
windows = ["63", "126"]

# 選擇公司和窗口 (允許多選)
selected_companies = st.multiselect("Select Companies:", companies, default=companies[0])
selected_windows = st.multiselect("Select Windows:", windows, default=windows[0])

# 需要顯示的比對項目
compare_options = {
    "EDA": "05 Predict Recession/05-00 EDA/{company}_{window}/FRM vs SI (Recession).gif",
    "FRM (Mean)": "99 Results/{company}_{window}/FRM (Mean).png",
    "FRM (Median)": "99 Results/{company}_{window}/FRM (Median).png"
}

# 創建多選框讓使用者選擇要顯示的項目
selected_comparisons = st.multiselect("Select comparisons to display:", compare_options.keys())

# 生成 Company_Window 組合
company_window_combinations = [(company, window) for company in selected_companies for window in selected_windows]

# 迭代選擇的 Company_Window 組合，並排展示結果
if company_window_combinations and selected_comparisons:
    columns = st.columns(len(company_window_combinations))
    for i, (company, window) in enumerate(company_window_combinations):
        with columns[i]:
            st.header(f"{company} ({window})")
            for comparison in selected_comparisons:
                file_path = os.path.join(base_path, compare_options[comparison].format(company=company, window=window))
                if os.path.exists(file_path):
                    if file_path.endswith(".gif"):
                        st.image(file_path, caption=f"{comparison}")
                    elif file_path.endswith(".png"):
                        image = Image.open(file_path)
                        st.image(image, caption=f"{comparison}")
                    else:
                        st.write(f"File path: {file_path}")
                else:
                    st.write(f"File not found: {file_path}")
else:
    st.write("Please select at least one company, one window, and one comparison to display.")
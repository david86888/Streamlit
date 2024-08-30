import streamlit as st
import Results
import Table

st.title("FRM@Taiwan")

if st.button('Score Results'):
    Results.show_results()

if st.button('Table Results'):
    Table.show_table()
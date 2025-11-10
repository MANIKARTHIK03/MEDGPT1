import pandas as pd
import streamlit as st

def load_data(uploaded_file):
    """Load uploaded CSV into a pandas DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def show_basic_info(df):
    """Display dataset summary info."""
    st.subheader("📋 Dataset Overview")
    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
    st.dataframe(df.head())

    st.subheader("🔍 Column Information")
    st.write(df.describe())

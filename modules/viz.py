import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def show_visualizations(df):
    """Generate basic charts."""
    st.subheader("📊 Data Visualizations")

    # Numeric distribution
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        selected_col = st.selectbox("Select a numeric column:", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, ax=ax)
        st.pyplot(fig)

    # Correlation heatmap
    if st.checkbox("Show correlation heatmap"):
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

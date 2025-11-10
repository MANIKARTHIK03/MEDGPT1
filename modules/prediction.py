import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_predict(df):
    """Train a simple ML model for disease prediction."""
    st.subheader("🤖 Machine Learning Prediction")

    target = st.selectbox("Select target column (label):", df.columns)
    features = st.multiselect("Select feature columns:", [col for col in df.columns if col != target])

    if len(features) == 0:
        st.warning("Please select feature columns.")
        return

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model Accuracy: {acc:.2f}")

    st.write("Feature Importances:")
    st.bar_chart(model.feature_importances_)

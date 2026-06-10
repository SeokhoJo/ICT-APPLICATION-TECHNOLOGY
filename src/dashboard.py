import os
import glob
import random

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# =========================================================
# Path Configuration
# =========================================================

BASE_DIR = r"F:\My Folder\study\2026\ICT\project_Data"
SPLIT_DIR = os.path.join(BASE_DIR, "processed", "dl_split_chunked")
TEST_DIR = os.path.join(SPLIT_DIR, "test")

# Final sklearn MLP files
MODEL_PATH = os.path.join(SPLIT_DIR, "mlp_chunked_model.pkl")
SCALER_PATH = os.path.join(SPLIT_DIR, "mlp_chunked_scaler.pkl")
FEATURE_COLS_PATH = os.path.join(SPLIT_DIR, "mlp_feature_cols.pkl")

# Final threshold selected by threshold tuning
ATTACK_THRESHOLD = 0.25


# =========================================================
# Streamlit Page Configuration
# =========================================================

st.set_page_config(
    page_title="ICT Network Anomaly Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# Loading Utilities
# =========================================================

@st.cache_data
def load_test_data(max_parts=10):
    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.parquet")))
    if len(files) == 0:
        raise FileNotFoundError(f"No parquet files found in {TEST_DIR}")

    selected_files = files[:max_parts]
    dfs = [pd.read_parquet(f) for f in selected_files]
    return pd.concat(dfs, ignore_index=True)


@st.cache_resource
def load_model_scaler_features():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)

    model_info = {
        "backend": "sklearn MLPClassifier",
        "hidden_layers": "128 → 64 → 32",
        "optimizer": "Adam",
        "loss": "Cross-entropy loss",
        "threshold": ATTACK_THRESHOLD,
    }

    return model, scaler, feature_cols, model_info


# =========================================================
# Prediction Utilities
# =========================================================

def prepare_x(df, feature_cols):
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.astype(np.float32)
    return X


def label_name(x):
    return "BENIGN" if int(x) == 0 else "ATTACK"


def confidence_level(attack_prob: float):
    """
    Confidence is interpreted separately from the final prediction.
    A flow can be classified as ATTACK but still be a borderline case
    when its ATTACK probability is close to the threshold.
    """
    if attack_prob >= ATTACK_THRESHOLD:
        if attack_prob >= 0.75:
            return "High-confidence ATTACK"
        elif attack_prob >= 0.50:
            return "Medium-confidence ATTACK"
        else:
            return "Low-confidence / Borderline ATTACK"
    else:
        if attack_prob <= 0.05:
            return "High-confidence BENIGN"
        elif attack_prob <= 0.15:
            return "Medium-confidence BENIGN"
        else:
            return "Low-confidence / Borderline BENIGN"


def predict_with_threshold(model, scaler, df, feature_cols):
    """
    sklearn model.predict() uses the default 0.5 decision threshold.
    This project uses predict_proba() and applies the tuned threshold 0.25.
    """
    X = prepare_x(df, feature_cols)
    X_scaled = scaler.transform(X).astype(np.float32)

    proba = model.predict_proba(X_scaled)
    attack_prob = proba[:, 1]
    pred = (attack_prob >= ATTACK_THRESHOLD).astype(np.int8)

    return pred, attack_prob, proba


def predict_single(model, scaler, row_df, feature_cols):
    pred, attack_prob, proba = predict_with_threshold(
        model=model,
        scaler=scaler,
        df=row_df,
        feature_cols=feature_cols,
    )

    benign_prob = float(proba[0, 0])
    attack_prob_value = float(proba[0, 1])
    conf = confidence_level(attack_prob_value)

    return int(pred[0]), benign_prob, attack_prob_value, conf


# =========================================================
# Scenario Utilities
# =========================================================

def source_to_scenario_name(source_file):
    name = source_file.lower()

    if "syn" in name:
        return "SYN Attack"
    if "udp" in name:
        return "UDP / DDoS Attack"
    if "ddos" in name or "dos" in name:
        return "DoS / DDoS Attack"
    if "portscan" in name:
        return "PortScan Attack"
    if "web" in name:
        return "Web Attack"
    if "bot" in name:
        return "Botnet Attack"
    if "infiltration" in name:
        return "Infiltration Attack"
    if "brute" in name:
        return "Brute Force Attack"
    if "morning" in name or "monday" in name:
        return "Mixed / Normal Traffic"

    return source_file


def make_unique_scenario_map(source_files):
    scenario_map = {}

    for src in source_files:
        base_name = source_to_scenario_name(src)
        if base_name not in scenario_map:
            scenario_map[base_name] = src
        else:
            scenario_map[f"{base_name} ({src})"] = src

    return scenario_map


# =========================================================
# Main
# =========================================================

st.title("🛡️ ICT Network Anomaly Detection Dashboard")
st.caption("Deep Learning-Based Network Anomaly Detection System using CIC-IDS 2017, 2018, and 2019")

st.sidebar.header("Data Loading Settings")

max_parts = st.sidebar.slider(
    "Number of test parquet parts to load",
    min_value=1,
    max_value=187,
    value=10,
    step=1
)

st.sidebar.caption(
    "The full test set consists of 187 parquet parts. "
    "You can load all parts, but it may take longer and use more memory."
)

try:
    test_df = load_test_data(max_parts=max_parts)
    model, scaler, feature_cols, model_info = load_model_scaler_features()
except FileNotFoundError as e:
    st.error("Required model or data file was not found.")
    st.code(str(e))
    st.stop()
except Exception as e:
    st.error("An error occurred while loading the data or model.")
    st.code(str(e))
    st.stop()

missing_features = [c for c in feature_cols if c not in test_df.columns]
if missing_features:
    st.error("Some training features are missing from the test data.")
    st.write(missing_features)
    st.stop()


# =========================================================
# Sidebar Evaluation Settings
# =========================================================

st.sidebar.markdown("---")
st.sidebar.header("Evaluation Settings")

sample_size = st.sidebar.slider(
    "Sample size for model evaluation",
    min_value=1000,
    max_value=len(test_df),
    value=min(20000, len(test_df)),
    step=10000 if len(test_df) > 100000 else 1000
)

random_state = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=9999,
    value=42
)

st.sidebar.markdown("---")
st.sidebar.write("Model")
st.sidebar.code(model_info["backend"])

st.sidebar.write("Hidden Layers")
st.sidebar.code(str(model_info["hidden_layers"]))

st.sidebar.write("Optimizer / Loss")
st.sidebar.code(f"{model_info.get('optimizer', 'N/A')} / {model_info.get('loss', 'N/A')}")

st.sidebar.write("Classification Type")
st.sidebar.code("BENIGN / ATTACK")

st.sidebar.write("Decision Threshold")
st.sidebar.code(f"ATTACK probability >= {ATTACK_THRESHOLD}")


# =========================================================
# 1. Dataset Overview
# =========================================================

st.header("1. Test Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Loaded Test Flows", f"{len(test_df):,}")
with col2:
    st.metric("Input Features", len(feature_cols))
with col3:
    attack_count = int((test_df["Label"] == 1).sum())
    st.metric("ATTACK Flows", f"{attack_count:,}")
with col4:
    benign_count = int((test_df["Label"] == 0).sum())
    st.metric("BENIGN Flows", f"{benign_count:,}")

left, right = st.columns(2)

with left:
    st.subheader("Label Distribution")
    label_counts = test_df["Label"].map({0: "BENIGN", 1: "ATTACK"}).value_counts()

    fig, ax = plt.subplots()
    ax.bar(label_counts.index, label_counts.values)
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.set_title("BENIGN vs ATTACK")
    st.pyplot(fig)

with right:
    st.subheader("Dataset Year Distribution")
    if "DatasetYear" in test_df.columns:
        year_counts = test_df["DatasetYear"].value_counts().sort_index()

        fig, ax = plt.subplots()
        ax.bar(year_counts.index.astype(str), year_counts.values)
        ax.set_xlabel("Dataset Year")
        ax.set_ylabel("Count")
        ax.set_title("Test Data by Year")
        st.pyplot(fig)
    else:
        st.warning("The DatasetYear column is not available.")


# =========================================================
# 2. Network Feature Visualization
# =========================================================

st.header("2. Network Traffic Feature Visualization")

numeric_features = [
    c for c in feature_cols
    if c in test_df.columns and pd.api.types.is_numeric_dtype(test_df[c])
]

default_features = [
    f for f in ["Flow Duration", "Flow Byts/s", "Flow Pkts/s", "Tot Fwd Pkts", "Tot Bwd Pkts"]
    if f in numeric_features
]

if len(numeric_features) == 0:
    st.warning("No numeric feature is available for visualization.")
else:
    selected_feature = st.selectbox(
        "Select a network feature to visualize",
        options=numeric_features,
        index=numeric_features.index(default_features[0]) if default_features else 0
    )

    plot_df = test_df.sample(n=min(10000, len(test_df)), random_state=int(random_state))

    st.subheader(f"{selected_feature} Distribution")
    fig, ax = plt.subplots()
    ax.hist(plot_df[selected_feature].replace([np.inf, -np.inf], np.nan).dropna(), bins=50)
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {selected_feature}")
    st.pyplot(fig)


# =========================================================
# 3. AI Model Evaluation
# =========================================================

st.header("3. AI Model Evaluation on Test Sample")

st.write(
    f"""
    The current system uses **{model_info['backend']}** and classifies a network flow as **ATTACK** when its ATTACK probability is greater than or equal to `{ATTACK_THRESHOLD}`.
    This threshold was selected based on the threshold tuning experiment.
    """
)

if st.button("Run AI Evaluation on Sample Data"):
    eval_df = test_df.sample(n=sample_size, random_state=int(random_state)).reset_index(drop=True)
    y_true = eval_df["Label"].astype(np.int8)

    y_pred, attack_prob, _ = predict_with_threshold(
        model=model,
        scaler=scaler,
        df=eval_df,
        feature_cols=feature_cols,
    )

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Sample Accuracy", f"{acc:.4f}")
    with c2:
        st.metric("Predicted ATTACK", f"{int((y_pred == 1).sum()):,}")
    with c3:
        st.metric("Predicted BENIGN", f"{int((y_pred == 0).sum()):,}")
    with c4:
        st.metric("Average ATTACK Probability", f"{float(np.mean(attack_prob)):.4f}")

    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(
        cm,
        index=["Actual BENIGN", "Actual ATTACK"],
        columns=["Predicted BENIGN", "Predicted ATTACK"]
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Classification Report")
    report = classification_report(
        y_true,
        y_pred,
        target_names=["BENIGN", "ATTACK"],
        output_dict=True,
        zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    st.subheader("ATTACK Probability Distribution")
    fig, ax = plt.subplots()
    ax.hist(attack_prob, bins=50)
    ax.axvline(ATTACK_THRESHOLD, linestyle="--")
    ax.set_xlabel("ATTACK Probability")
    ax.set_ylabel("Frequency")
    ax.set_title(f"ATTACK Probability Distribution, Threshold = {ATTACK_THRESHOLD}")
    st.pyplot(fig)


# =========================================================
# 4. Attack Scenario Test
# =========================================================

st.header("4. Attack Scenario Test")

st.write(
    """
    Select an attack or traffic scenario below.  
    The dashboard samples one actual network flow from the selected source file and checks whether the AI model classifies it as BENIGN or ATTACK.
    """
)

if "SourceFile" not in test_df.columns:
    st.warning("The SourceFile column is not available. Scenario testing cannot be used.")
else:
    source_files = sorted(test_df["SourceFile"].unique().tolist())
    scenario_map = make_unique_scenario_map(source_files)
    scenario_names = list(scenario_map.keys())

    selected_scenario = st.selectbox("Select an attack or traffic scenario", options=scenario_names)
    selected_source = scenario_map[selected_scenario]
    st.info(f"Selected source file: {selected_source}")

    scenario_df = test_df[test_df["SourceFile"] == selected_source].copy()

    actual_label_filter = st.radio(
        "Select sample type",
        options=["Random", "Actual ATTACK only", "Actual BENIGN only"],
        horizontal=True
    )

    if actual_label_filter == "Actual ATTACK only":
        scenario_df = scenario_df[scenario_df["Label"] == 1]
    elif actual_label_filter == "Actual BENIGN only":
        scenario_df = scenario_df[scenario_df["Label"] == 0]

    if len(scenario_df) == 0:
        st.warning("No data is available for the selected condition.")
    else:
        if st.button("Detect This Scenario with AI"):
            one_row = scenario_df.sample(n=1, random_state=random.randint(0, 999999)).reset_index(drop=True)

            pred, benign_prob, attack_prob, conf = predict_single(
                model=model,
                scaler=scaler,
                row_df=one_row,
                feature_cols=feature_cols,
            )

            actual = int(one_row["Label"].iloc[0])

            st.subheader("AI Detection Result")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("Selected Scenario", selected_scenario)
            with c2:
                st.metric("Actual Label", label_name(actual))
            with c3:
                st.metric("AI Prediction", label_name(pred))
            with c4:
                st.metric("ATTACK Probability", f"{attack_prob:.4f}")
            with c5:
                st.metric("Confidence", conf)

            if pred == actual:
                st.success("The AI prediction matches the actual label.")
            else:
                st.error("The AI prediction does not match the actual label.")

            if "Borderline" in conf:
                st.warning(
                    "This is a borderline decision. The prediction is close to the decision threshold, "
                    "so it should be interpreted with lower confidence."
                )

            st.subheader("Prediction Probability")
            prob_df = pd.DataFrame({
                "Class": ["BENIGN", "ATTACK"],
                "Probability": [benign_prob, attack_prob]
            })

            fig, ax = plt.subplots()
            ax.bar(prob_df["Class"], prob_df["Probability"])
            ax.axhline(ATTACK_THRESHOLD, linestyle="--")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title("AI Prediction Probability")
            st.pyplot(fig)

            st.dataframe(prob_df, use_container_width=True)

            st.subheader("Selected Network Flow Features")
            display_cols = [c for c in ["DatasetYear", "SourceFile", "Label"] if c in one_row.columns] + feature_cols[:20]
            temp_row = one_row[display_cols].copy()
            if "Label" in temp_row.columns:
                temp_row["Label"] = temp_row["Label"].map({0: "BENIGN", 1: "ATTACK"})
            st.dataframe(temp_row, use_container_width=True)


# =========================================================
# 5. System Interpretation
# =========================================================

st.header("5. System Interpretation")

st.markdown(
    f"""
    ### System Overview

    This dashboard presents a deep learning-based network anomaly detection system using CIC-IDS 2017, 2018, and 2019 flow-level traffic data.

    The selected model classifies each network flow as either **BENIGN** or **ATTACK** based on {len(feature_cols)} network traffic features.

    ### Model Configuration

    - Model Backend: {model_info['backend']}
    - Hidden Layers: {model_info['hidden_layers']}
    - Optimizer: {model_info.get('optimizer', 'N/A')}
    - Loss: {model_info.get('loss', 'N/A')}
    - Input Features: {len(feature_cols)}
    - Decision Threshold: ATTACK probability >= {ATTACK_THRESHOLD}

    ### Why sklearn MLP Is Used as the Final Model

    A PyTorch MLP with weighted loss was also tested to improve attack recall. Although it slightly improved attack recall, it significantly increased false positives and reduced overall accuracy. Therefore, the tuned sklearn MLP was selected as the final model because it provided the best overall balance.

    ### Confidence Interpretation

    Confidence is interpreted based on the ATTACK probability. A flow can be classified as **ATTACK** but still be marked as a **borderline attack** if its attack probability is close to the threshold.

    - High-confidence ATTACK: ATTACK probability >= 0.75
    - Medium-confidence ATTACK: 0.50 <= ATTACK probability < 0.75
    - Low-confidence / Borderline ATTACK: {ATTACK_THRESHOLD} <= ATTACK probability < 0.50
    - Low-confidence / Borderline BENIGN: 0.15 < ATTACK probability < {ATTACK_THRESHOLD}
    - Medium-confidence BENIGN: 0.05 < ATTACK probability <= 0.15
    - High-confidence BENIGN: ATTACK probability <= 0.05

    ### Limitation

    This model is a binary classification model. It does not directly classify the specific attack type, such as SYN, DDoS, PortScan, or Web Attack.

    When a user selects a scenario such as **SYN Attack**, the system samples a flow from the corresponding source file and checks whether the AI model detects it as **ATTACK** or **BENIGN**.

    ### Future Improvement

    To classify specific attack types, the label structure should be changed from binary labels to multi-class labels such as **BENIGN**, **SYN**, **DDoS**, **PortScan**, **Web Attack**, **Botnet**, and **Infiltration**.
    """
)

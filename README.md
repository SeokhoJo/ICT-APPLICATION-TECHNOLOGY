# Network Anomaly Detection System using CIC-IDS Datasets

## 1. Project Overview

This project implements a network anomaly detection system that classifies network traffic flows as either **BENIGN** or **ATTACK** using flow-level statistical features.

The system was built as an ICT project to demonstrate the full pipeline of network security data processing:

1. Collect and integrate public intrusion detection datasets
2. Preprocess and normalize large-scale network flow data
3. Train a deep learning-based binary classification model
4. Tune the decision threshold to reduce missed attacks
5. Visualize the final result through an interactive web dashboard

Unlike payload inspection systems, this project does **not** analyze packet contents.
Instead, it uses numerical flow-level features such as packet length, flow duration, byte count, packet rate, and flag count.

---

## 2. Repository Structure

```text
ICT-Network-Anomaly-Detection/
│
├── src/
│   ├── preprocess_all_split_70_20_10.py
│   ├── train_deep_mlp_chunked.py
│   ├── evaluate_thresholds_chunked.py
│   └── utils/
│
├── dashboard/
│   └── dashboard.py
│
├── data/
│   ├── sample/
│   └── synthetic/
│
├── models/
│   └── trained_model.pt
│
├── results/
│   ├── threshold_results.csv
│   ├── confusion_matrix.png
│   └── evaluation_summary.txt
│
├── README.md
├── TROUBLESHOOTING.md
└── requirements.txt
```

### Folder Description

| Folder / File        | Description                                                   |
| -------------------- | ------------------------------------------------------------- |
| `/src`               | Source code for preprocessing, model training, and evaluation |
| `/dashboard`         | Streamlit-based web GUI code                                  |
| `/data`              | Sample or synthetic data used for demonstration               |
| `/models`            | Saved trained model files                                     |
| `/results`           | Evaluation results, threshold comparison, and visual outputs  |
| `README.md`          | System overview, logic, execution guide, and video link       |
| `TROUBLESHOOTING.md` | Major bugs faced during development and solutions             |
| `requirements.txt`   | Python package dependencies                                   |

---

## 3. Dataset

This project uses public network intrusion detection datasets:

* CIC-IDS2017
* CSE-CIC-IDS2018
* CIC-DDoS2019

The original datasets contain network flow records extracted from packet captures.
For this project, multiple CSV files from different years were integrated into one unified binary classification dataset.

### Final Dataset Summary

| Item               |                 Value |
| ------------------ | --------------------: |
| Total flows        |            34,296,907 |
| Training set       |            24,010,781 |
| Validation set     |             6,858,734 |
| Test set           |             3,427,392 |
| Number of features |                    76 |
| Task               | Binary classification |
| Labels             |       BENIGN / ATTACK |

### Label Distribution

| Label  |      Count |
| ------ | ---------: |
| ATTACK | 18,800,618 |
| BENIGN | 15,496,289 |

---

## 4. Preprocessing Logic

The preprocessing step was designed to make different CIC datasets compatible with one another.

Main preprocessing steps:

1. Load multiple CSV files from CIC-IDS2017, CSE-CIC-IDS2018, and CIC-DDoS2019
2. Standardize column names across datasets
3. Remove unnecessary identifier columns
4. Remove IP addresses, ports, flow IDs, and timestamps
5. Convert labels into binary classes
6. Keep only numerical flow-level features
7. Split the integrated dataset into train, validation, and test sets
8. Save the processed data as chunked Parquet files

### Removed Columns

The following types of columns were removed:

* Flow ID
* Source IP
* Destination IP
* Source Port
* Destination Port
* Timestamp

These columns were removed because they can cause the model to memorize dataset-specific identifiers instead of learning general traffic behavior.
The goal of this system is to detect abnormal flow patterns, not to depend on specific IP addresses or port numbers.

---

## 5. Model Design

The final model is a deep learning-based Multi-Layer Perceptron, or MLP.

### Model Architecture

```text
Input Layer: 76 features
Hidden Layer 1: 128 neurons + ReLU
Hidden Layer 2: 64 neurons + ReLU
Hidden Layer 3: 32 neurons + ReLU
Output Layer: 1 neuron
Activation: BCEWithLogitsLoss
Optimizer: Adam
Epochs: 5
```

MLP was selected because the input data is tabular numerical flow-level data.
Since the features are not images or sequential text, a fully connected neural network is suitable for learning relationships among numerical traffic features.

---

## 6. Threshold Tuning

The model outputs a probability-like score for each flow.
Instead of using the default threshold of `0.5`, several thresholds were tested on the validation set.

The final threshold was set to:

```text
Threshold = 0.25
```

This threshold was selected because, in network security, missing an actual attack can be more dangerous than generating a false alarm.
Therefore, the project prioritizes improving **attack recall** and reducing false negatives.

---

## 7. Final Evaluation Result

The final model was evaluated on the test set.

| Metric        | Result |
| ------------- | -----: |
| Accuracy      | 94.62% |
| Attack Recall | 95.05% |
| Threshold     |   0.25 |

The result shows that the model can classify large-scale network flows with high accuracy while maintaining strong attack detection performance.

---

## 8. Dashboard

The project includes an interactive web dashboard built with Streamlit.

The dashboard provides the following functions:

1. Load processed test data
2. Select dataset year
3. Control sample size
4. Adjust classification threshold
5. Display dataset overview
6. Show model evaluation results
7. Visualize confusion matrix and classification report
8. Run scenario-based single-flow prediction

### Dashboard Example Features

* Test Parquet file count selection
* Sample size selection
* Threshold slider
* Dataset year filter
* Attack-only filter
* Random seed control
* Model prediction result
* Confusion matrix
* Classification report

---

## 9. How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### 4. Install Requirements

```bash
pip install -r requirements.txt
```

### 5. Run Preprocessing

```bash
python src/preprocess_all_split_70_20_10.py
```

### 6. Train Model

```bash
python src/train_deep_mlp_chunked.py
```

### 7. Evaluate Thresholds

```bash
python src/evaluate_thresholds_chunked.py
```

### 8. Run Dashboard

```bash
streamlit run dashboard/dashboard.py
```

---

## 10. System Pipeline

```text
Raw CIC Dataset
      ↓
Column Standardization
      ↓
Feature Selection
      ↓
Train / Validation / Test Split
      ↓
Standard Scaling
      ↓
MLP Model Training
      ↓
Threshold Tuning
      ↓
Final Evaluation
      ↓
Streamlit Dashboard
```

---

## 11. Key Design Decisions

### 1. Flow-Level Feature-Based Detection

This project does not inspect packet payloads.
It uses flow-level statistical features, which makes the system more privacy-friendly and practical for large-scale traffic analysis.

### 2. Removing IP and Port Information

IP addresses and port numbers were removed to reduce overfitting.
If these identifiers remain, the model may learn specific dataset patterns instead of general attack behavior.

### 3. Binary Classification

The current system predicts only whether a flow is BENIGN or ATTACK.
This design makes the system simple and suitable for an initial anomaly detection pipeline.

### 4. Threshold Adjustment

The threshold was adjusted from 0.5 to 0.25 because attack recall is more important than overall accuracy in security systems.

---

## 12. Limitations

This project has several limitations:

1. The model performs binary classification only
2. It does not identify the exact attack type
3. It does not process real-time packet streams directly
4. It uses pre-extracted flow-level features
5. The dashboard is designed for demonstration and analysis, not production deployment

---

## 13. Future Work

Future improvements may include:

1. Multi-class attack classification
2. Real-time packet capture and flow extraction
3. Additional model comparison with Random Forest, XGBoost, or LightGBM
4. Improved class imbalance handling
5. Deployment as a real-time monitoring system
6. More detailed attack scenario visualization

---

## 14. Demo Video

Project demonstration video:

```text
Video Link: https://YOUR_VIDEO_LINK_HERE
```

The video shows:

1. Project overview
2. Dataset and preprocessing explanation
3. Model evaluation result
4. Dashboard demonstration
5. Scenario-based prediction example

---

## 15. Tech Stack

| Category             | Technology                                 |
| -------------------- | ------------------------------------------ |
| Programming Language | Python                                     |
| Data Processing      | pandas, numpy, pyarrow                     |
| Machine Learning     | PyTorch, scikit-learn                      |
| Visualization        | matplotlib, seaborn                        |
| Dashboard            | Streamlit                                  |
| Data Format          | CSV, Parquet                               |
| Dataset              | CIC-IDS2017, CSE-CIC-IDS2018, CIC-DDoS2019 |

---

## 16. Conclusion

This project demonstrates an end-to-end network anomaly detection system using large-scale public intrusion detection datasets.

The system integrates multiple CIC datasets, preprocesses them into a unified structure, trains an MLP-based binary classifier, tunes the decision threshold to improve attack recall, and provides an interactive dashboard for result analysis.

The final model achieved **94.62% accuracy** and **95.05% attack recall** on the test set, showing that flow-level statistical features can be effectively used for network attack detection.

# Network Anomaly Detection Dashboard

## 1. Project Overview

This project is a machine learning-based network anomaly detection system.

The system classifies network traffic flows into two categories:

* **BENIGN**
* **ATTACK**

The project was developed for the ICT Application Technology course. It demonstrates a complete security data analysis pipeline, including dataset preprocessing, model training, threshold tuning, evaluation, and dashboard visualization.

This system does **not** inspect packet payloads or private message contents. Instead, it uses flow-level statistical features extracted from network traffic, such as packet length, flow duration, byte count, packet rate, and flag count.

The main goal of this project is to detect abnormal traffic patterns from large-scale network flow data and visualize the result through an interactive dashboard.

---

## 2. Repository Structure

```text
ICT-APPLICATION-TECHNOLOGY/
│
├── src/
│   ├── preprocess_all_split_70_20_10.py
│   ├── train_deep_mlp_chunked.py
│   ├── train_pytorch_mlp_chunked.py
│   └── evaluate_thresholds_chunked.py
│
├── dashboard/
│   └── dashboard.py
│
├── data/
│   └── sample_network_flows.csv
│
├── README.md
├── TROUBLESHOOTING.md
└── requirements.txt
```

## 3. Folder Description

| Path                 | Description                                                             |
| -------------------- | ----------------------------------------------------------------------- |
| `/src`               | Source code for preprocessing, model training, and threshold evaluation |
| `/dashboard`         | Streamlit-based web GUI code                                            |
| `/data`              | Sample network flow data for demonstration                              |
| `README.md`          | Project overview, system logic, execution guide, and demo video link    |
| `TROUBLESHOOTING.md` | Major bugs faced during development and how they were solved            |
| `requirements.txt`   | Python package dependencies                                             |

---

## 4. Dataset

This project uses public network intrusion detection datasets from the CIC dataset family.

The datasets used in the full project are:

* CIC-IDS2017
* CSE-CIC-IDS2018
* CIC-DDoS2019

The original datasets contain network flow records extracted from network traffic. Each row represents one network flow and contains numerical features that describe the behavior of that flow.

Because the full processed dataset is very large, this GitHub repository only includes a sample data file:

```text
data/sample_network_flows.csv
```

The sample file is provided to show the data structure and to make the repository easier to review.

---

## 5. Full Dataset Summary

The full processed dataset used in the project contains approximately 34.3 million network flows.

| Split      | Number of Rows |
| ---------- | -------------: |
| Train      |     24,010,781 |
| Validation |      6,858,734 |
| Test       |      3,427,392 |
| Total      |     34,296,907 |

The final dataset contains 76 numerical flow-level features.

| Label  | Meaning                               |
| ------ | ------------------------------------- |
| BENIGN | Normal network traffic                |
| ATTACK | Malicious or abnormal network traffic |

---

## 6. Preprocessing Logic

The preprocessing step was designed to integrate multiple CIC datasets into one unified format.

Main preprocessing steps:

1. Load raw CSV files from CIC-IDS2017, CSE-CIC-IDS2018, and CIC-DDoS2019
2. Standardize column names
3. Remove unnecessary identifier columns
4. Remove IP addresses, ports, flow IDs, and timestamps
5. Keep only numerical flow-level features
6. Convert original labels into binary labels
7. Split the dataset into train, validation, and test sets
8. Save the processed data as chunked Parquet files

### Removed Columns

The following types of columns were removed:

* Flow ID
* Source IP
* Destination IP
* Source Port
* Destination Port
* Timestamp

These columns were removed to reduce overfitting and data leakage. If IP addresses or port numbers remain in the dataset, the model may memorize dataset-specific identifiers instead of learning general attack behavior.

The purpose of this project is to detect abnormal traffic behavior using statistical patterns, not to depend on specific IP addresses or port values.

---

## 7. Model Design

The final model is a deep learning-based Multi-Layer Perceptron, or MLP.

MLP was selected because the input data is tabular numerical data. Since the dataset is not image data or sequential text data, a fully connected neural network is suitable for learning relationships among flow-level features.

### Model Architecture

```text
Input Layer: 76 features
Hidden Layer 1: 128 neurons + ReLU
Hidden Layer 2: 64 neurons + ReLU
Hidden Layer 3: 32 neurons + ReLU
Output Layer: 1 neuron
Loss Function: BCEWithLogitsLoss
Optimizer: Adam
Epochs: 5
```

The model performs binary classification and outputs a score indicating whether each network flow is likely to be an attack.

---

## 8. Threshold Tuning

In binary classification, the default threshold is usually 0.5.

However, in network security, missing an actual attack can be more dangerous than generating a false alarm. Therefore, this project tested several threshold values using the validation set.

The final threshold was selected as:

```text
Threshold = 0.25
```

This threshold was selected to improve attack recall and reduce false negatives.

---

## 9. Final Evaluation Result

The final model was evaluated on the test set.

| Metric        | Result |
| ------------- | -----: |
| Accuracy      | 94.62% |
| Attack Recall | 95.05% |
| Threshold     |   0.25 |

The result shows that the model can classify large-scale network flows with high accuracy while maintaining strong attack detection performance.

---

## 10. Dashboard

The project includes an interactive dashboard built with Streamlit.

The dashboard provides the following functions:

* Load sample or processed test data
* Select the number of test files
* Adjust sample size
* Set classification threshold
* Filter data by year
* Show attack-only data
* Control random seed
* Display dataset overview
* Show model evaluation results
* Display confusion matrix
* Display classification report
* Run scenario-based single-flow prediction

The dashboard helps users understand the model result visually and interactively.

---

## 11. System Pipeline

```text
Raw CIC Dataset
        ↓
Column Standardization
        ↓
Feature Selection
        ↓
Identifier Column Removal
        ↓
Train / Validation / Test Split
        ↓
Feature Scaling
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

## 12. How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/SeokhoJo/ICT-APPLICATION-TECHNOLOGY.git
cd ICT-APPLICATION-TECHNOLOGY
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run Preprocessing

```bash
python src/preprocess_all_split_70_20_10.py
```

### 6. Train the Model

```bash
python src/train_deep_mlp_chunked.py
```

### 7. Evaluate Thresholds

```bash
python src/evaluate_thresholds_chunked.py
```

### 8. Run the Dashboard

```bash
streamlit run dashboard/dashboard.py
```

---

## 13. Sample Data

This repository includes a sample data file:

```text
data/sample_network_flows.csv
```

The sample data is included for demonstration and repository review. The full CIC datasets are not included because they are too large for GitHub.

---

## 14. Key Design Decisions

### Flow-Level Detection

This system uses flow-level statistical features rather than raw packet payloads. This makes the system suitable for large-scale traffic analysis and avoids directly inspecting private message contents.

### Identifier Removal

IP addresses, ports, timestamps, and flow IDs were removed to reduce overfitting and data leakage.

### Binary Classification

The project focuses on binary classification: BENIGN or ATTACK. This design makes the first version of the system simple and easy to interpret.

### Threshold Adjustment

The final threshold was adjusted to 0.25 because attack recall is more important than simple accuracy in a security detection system.

---

## 15. Limitations

This project has several limitations:

1. The system performs binary classification only.
2. It does not classify the exact attack type.
3. It does not analyze packet payloads.
4. It uses pre-extracted flow-level features.
5. It is a demonstration system, not a production-level intrusion detection system.
6. Real-time packet capture is not included in the current version.

---

## 16. Future Work

Possible future improvements include:

1. Multi-class attack classification
2. Real-time packet capture and flow extraction
3. Comparison with other models such as Random Forest, XGBoost, and LightGBM
4. Improved class imbalance handling
5. Real-time monitoring dashboard
6. More detailed attack scenario visualization
7. Model calibration for more reliable prediction scores

---

## 17. Troubleshooting

Major bugs and solutions are documented in:

```text
TROUBLESHOOTING.md
```

The troubleshooting document includes issues such as:

* Python environment problems
* Dataset path errors
* Large file processing problems
* Parquet loading errors
* Sample data generation issues
* Dashboard path configuration issues
* Threshold mismatch between evaluation and dashboard

---

## 18. Demo Video

Project demonstration video:

```text
Video Link: https://YOUR_VIDEO_LINK_HERE
```

The video demonstrates:

1. Project overview
2. Dataset and preprocessing logic
3. Model design
4. Evaluation result
5. Streamlit dashboard
6. Scenario-based prediction

---

## 19. Tech Stack

| Category             | Technology                                 |
| -------------------- | ------------------------------------------ |
| Programming Language | Python                                     |
| Data Processing      | pandas, numpy, pyarrow                     |
| Machine Learning     | PyTorch, scikit-learn                      |
| Dashboard            | Streamlit                                  |
| Data Format          | CSV, Parquet                               |
| Dataset              | CIC-IDS2017, CSE-CIC-IDS2018, CIC-DDoS2019 |

---

## 20. Conclusion

This project demonstrates an end-to-end network anomaly detection system using large-scale public network intrusion detection datasets.

The system integrates multiple CIC datasets, preprocesses them into a unified feature structure, trains an MLP-based binary classifier, tunes the decision threshold to improve attack recall, and visualizes the final result through a Streamlit dashboard.

The final model achieved 94.62% accuracy and 95.05% attack recall on the test set, showing that flow-level statistical features can be effectively used for network anomaly detection.

# Troubleshooting

This document summarizes the major problems faced during the development of the Network Anomaly Detection Dashboard project and how they were solved.

---

## 1. Python Environment Issue

### Problem

During the project, some Python packages did not work correctly because the Python environment was not consistent.

For example, packages such as `pandas`, `pyarrow`, `torch`, and `streamlit` had to be installed in the same virtual environment. In some cases, the terminal was using a different Python interpreter than expected.

### Cause

The project was developed using a local virtual environment. However, if the virtual environment was not activated correctly, Python executed commands using the global environment instead of the project environment.

### Solution

A virtual environment was created and activated before running the project.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

After activating the environment, the required packages were installed.

```bash
pip install -r requirements.txt
```

### Lesson Learned

Using a virtual environment is important for keeping project dependencies consistent and avoiding package conflicts.

---

## 2. Dataset Path Error

### Problem

Some scripts failed because the dataset path was different depending on the local computer environment.

For example, the project data path was originally set to a local absolute path such as:

```text
D:\my folder\study\ICT\project_Data
```

If the same script was executed on another computer, the file path did not exist.

### Cause

The scripts used absolute paths for dataset loading. Absolute paths are easy to use locally, but they reduce portability.

### Solution

The base directory was clearly defined in each script using a `BASE_DIR` variable. The path settings were checked and corrected in the main scripts:

```text
preprocess_all_split_70_20_10.py
train_deep_mlp_chunked.py
evaluate_thresholds_chunked.py
dashboard.py
```

For GitHub submission, only sample data is included in the repository, and the README explains that the full dataset must be downloaded separately.

### Lesson Learned

For public repositories, relative paths or clearly documented path settings are better than hard-coded local paths.

---

## 3. Large CSV File Processing Problem

### Problem

The original CIC datasets were too large to process at once.

The full processed dataset contained more than 34 million rows. Loading all raw CSV files into memory at once could cause high memory usage or slow execution.

### Cause

CSV files are large, and the project integrates multiple datasets from CIC-IDS2017, CSE-CIC-IDS2018, and CIC-DDoS2019. Processing all rows at once is inefficient for a local development environment.

### Solution

The dataset was processed and saved as chunked Parquet files.

Parquet was used because it is more efficient than CSV for large tabular data. The dataset was split into train, validation, and test sets, and each split was stored in multiple Parquet chunks.

Final split:

```text
Train: 24,010,781 rows
Validation: 6,858,734 rows
Test: 3,427,392 rows
```

### Lesson Learned

For large-scale machine learning projects, chunk-based processing and efficient file formats such as Parquet are necessary.

---

## 4. Column Name Inconsistency

### Problem

Different CIC datasets used slightly different column names.

For example, some datasets used different spacing, capitalization, or naming styles for similar features.

### Cause

The project integrated datasets from different years and sources. Even though they belong to the CIC dataset family, their column formats were not perfectly identical.

### Solution

Column names were standardized during preprocessing.

The preprocessing script cleaned and unified column names before feature selection and label conversion.

### Lesson Learned

When combining multiple datasets, schema unification is one of the most important preprocessing steps.

---

## 5. Label Format Issue

### Problem

The original datasets contained multiple attack names, while the project required binary classification.

Examples of original labels include:

```text
BENIGN
DDoS
PortScan
Web Attack
UDP
```

However, the model was designed to classify traffic into only two classes:

```text
BENIGN
ATTACK
```

### Cause

The original datasets were designed for multi-class intrusion detection, but this project focused on binary anomaly detection.

### Solution

All non-BENIGN labels were converted into `ATTACK`.

```text
BENIGN → 0
All attack labels → 1
```

### Lesson Learned

The label structure must match the project goal. Since this project focused on anomaly detection, binary classification was appropriate.

---

## 6. IP and Port Overfitting Issue

### Problem

At first, there was a concern that using IP addresses or port numbers could make the model overfit to dataset-specific information.

For example, if a certain destination IP or port appears frequently in attack samples, the model may memorize that identifier instead of learning general attack behavior.

### Cause

IP addresses, port numbers, timestamps, and flow IDs can act as identifiers. These values may not generalize well to other network environments.

### Solution

The following columns were removed during preprocessing:

```text
Flow ID
Source IP
Destination IP
Source Port
Destination Port
Timestamp
```

The final model uses numerical flow-level statistical features instead.

### Lesson Learned

Removing identifier-like columns is important for reducing data leakage and improving generalization.

---

## 7. Threshold Mismatch Between Evaluation and Dashboard

### Problem

The model evaluation and the dashboard could show different results if they used different classification thresholds.

For example, evaluation might use a threshold of `0.25`, while the dashboard might still use the default threshold of `0.5`.

### Cause

Binary classification models output a score, and this score must be converted into a final class using a threshold. If the threshold is not consistent across scripts, the reported performance can be different.

### Solution

The threshold was tuned using the validation set, and the final threshold was set to:

```text
0.25
```

The same default threshold was applied to both evaluation and dashboard settings.

### Lesson Learned

Threshold values must be managed consistently across training, evaluation, and visualization.

---

## 8. Sample Data Generation Issue

### Problem

A sample data generation script was used to create a small demonstration file for GitHub.

However, when the sample size was too small, the generated sample could contain only one class, such as only `ATTACK` or only `BENIGN`.

### Cause

The original dataset is large and imbalanced depending on the selected file chunk. If the sampling logic selects too few rows from one chunk, the sample may not represent both classes.

### Solution

The sampling logic was checked to ensure that the sample data could include representative rows. The GitHub repository includes only a small sample file for structure demonstration, not full model training.

Final sample file:

```text
data/sample_network_flows.csv
```

### Lesson Learned

Sample data for GitHub should be small, but it should still represent the expected data structure clearly.

---

## 9. Dashboard File Location Issue

### Problem

The dashboard code needed to be placed in the correct folder for the GitHub repository structure.

The project requirement specified:

```text
/dashboard: Web GUI code
```

However, if the dashboard file is placed in another folder or duplicated in multiple folders, the repository structure may become confusing.

### Cause

During development, the dashboard file was tested in different locations.

### Solution

The final repository structure should place the Streamlit dashboard code in:

```text
dashboard/dashboard.py
```

The dashboard can be executed with:

```bash
streamlit run dashboard/dashboard.py
```

### Lesson Learned

A clean repository structure makes the project easier to review and execute.

---

## 10. GitHub Large File Limitation

### Problem

The full processed dataset is too large to upload to GitHub.

The complete dataset contains more than 34 million rows, and the processed Parquet files are also large.

### Cause

GitHub is not suitable for storing very large machine learning datasets directly in a normal repository.

### Solution

Only a small sample file was uploaded to the repository.

```text
data/sample_network_flows.csv
```

The README explains that the full CIC datasets should be downloaded separately from the original public dataset sources.

### Lesson Learned

For machine learning projects, GitHub should contain source code, documentation, and sample data, while large datasets should be stored or referenced separately.

---

## 11. Summary

The main challenges in this project were related to large-scale data processing, dataset integration, environment configuration, and consistency between evaluation and dashboard visualization.

Through these troubleshooting steps, the project was improved in the following ways:

* More stable Python environment
* Clearer dataset path management
* Efficient chunk-based Parquet processing
* Unified column structure
* Reduced overfitting risk by removing identifiers
* Consistent threshold usage
* Cleaner GitHub repository structure
* Small sample data for public submission

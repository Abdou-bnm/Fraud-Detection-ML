<div align="center">

# Credit Card Fraud Detection ML

**A Machine Learning approach to detect fraudulent credit card transactions using Random Forest Classifier**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)  
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)  

</div>

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Analysis & Visualizations](#analysis--visualizations)
- [Results](#results)
- [Author](#author)

---

## Project Overview
This project implements a **Machine Learning solution** for detecting fraudulent credit card transactions. Using a comprehensive dataset of credit card transactions, we've built a **Random Forest Classifier** that can accurately identify potentially fraudulent activities while minimizing false positives.

### Problem Statement
Credit card fraud is a significant concern in the financial industry, causing billions of dollars in losses annually. Traditional rule-based systems often fail to adapt to new fraud patterns. This project leverages machine learning to:
- Detect fraudulent transactions in real-time  
- Reduce false positives to maintain customer satisfaction  
- Provide interpretable results for fraud investigation teams  

---

## Features
- **High Accuracy**: Achieves excellent performance metrics with Random Forest  
- **Comprehensive Analysis**: Detailed exploratory data analysis and visualization  
- **Feature Importance**: Identifies the most important factors in fraud detection  
- **Performance Metrics**: ROC curves, confusion matrices, and classification reports  
- **Interactive Visualizations**: Beautiful plots using Matplotlib and Seaborn  
- **Scalable**: Easily deployable for real-world applications  

---

## Technologies Used
<div align="center">

| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) | Core programming language |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) | Data manipulation and analysis |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) | Numerical computing |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) | Machine learning algorithms |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white) | Data visualization |
| ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white) | Statistical data visualization |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white) | Interactive development environment |

</div>

---

## Project Structure

```bash
Fraud-Detection-ML/
│
├── data/                        # Data directory (ignored in git)
│   ├── .gitkeep                  # placeholder to keep the folder
│   └── sample_small.csv          # small demo file (optional, tracked in git)
│
├── notebook/
│   └── fraud_detection.ipynb     # Main Jupyter notebook with analysis
│
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## Installation

### Prerequisites
- Python 3.7 or higher  
- pip package manager  

### Step-by-step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Abdou-bnm/Fraud-Detection-ML.git
   cd Fraud-Detection-ML
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv fraud_detection_env
   ```

   **Activate the environment**
   - On Windows:
     ```bash
     fraud_detection_env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source fraud_detection_env/bin/activate
     ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **Open the analysis notebook**
   Navigate to:  
   ```
   notebook/fraud_detection.ipynb
   ```

---

## Data  

This project uses the **Credit Card Fraud Detection Dataset 2023** from Kaggle.  

- **Dataset link**: [Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/data)  
- **Size**: ~550,000 transactions (~150 MB)  
- **Source**: Kaggle (uploaded by [nelgiriyewithana](https://www.kaggle.com/nelgiriyewithana))  
- **Note**: The dataset is **not included** in this repository due to size and licensing. Please download it yourself if you’d like to run the full analysis.  

### Dataset Description
This dataset contains **credit card transactions made by European cardholders in the year 2023**.  
It comprises **over 550,000 records**, and the data has been **anonymized** to protect cardholder identities.  

The primary objective of the dataset is to support the **development of fraud detection algorithms and models** that can identify potentially fraudulent transactions.  

### How to Use It
1. Download the dataset from the Kaggle link above.  
2. Move the CSV file into the `data/` folder.  

Expected structure:
```
Fraud-Detection-ML/
└── data/
    └── creditcard_2023.csv   # Kaggle dataset file (not tracked in git)
```

👉 For quick tests, the repo includes small `sample_*.csv` files (kept in git) so you can try out the notebooks without downloading the full dataset.  

### Demo data
For lightweight testing, a small demo dataset is already included in this repository:  

- **File**: `data/sample_small.csv`  
- **Size**: first 5,000 rows of the full dataset  

This sample is small enough to be tracked in git and lets you run the notebooks instantly, even if you don’t download the full dataset from Kaggle.  

---

## Usage

### Running the Analysis

1. **Data Loading & Exploration**
   - Load the credit card dataset (from `data/creditcard_2023.csv` or `data/sample_small.csv`)  
   - Perform exploratory data analysis  
   - Check for missing values and data quality  

2. **Data Preprocessing**
   - Feature selection and engineering  
   - Data scaling using `StandardScaler`  
   - Train-test split for model validation  

3. **Model Training**
   - Train a Random Forest Classifier  
   - Perform cross-validation  
   - Hyperparameter optimization  

4. **Model Evaluation**
   - Generate predictions on the test set  
   - Create confusion matrix and classification report  
   - Plot ROC curve and calculate AUC  

5. **Visualization & Insights**
   - Feature importance analysis  
   - Correlation matrix visualization  
   - Performance metrics visualization  

---

## Model Performance

### Key Metrics

| Metric       | Score                          |
|--------------|--------------------------------|
| **Accuracy** | ~99.9%                         |
| **Precision**| High precision for fraud cases |
| **Recall**   | Excellent fraud capture rate   |
| **F1-Score** | Balanced performance metric    |
| **AUC-ROC**  | Near-perfect classification    |

### Cross-Validation Results
- **5-Fold Cross-Validation** implemented  
- **Consistent performance** across all folds  
- **Robust model** with minimal overfitting  

---

## Analysis & Visualizations

The project includes comprehensive visualizations:

- **Confusion Matrix**: Visual representation of model predictions  
- **ROC Curve**: Model discrimination capability  
- **Feature Importance**: Most influential features for fraud detection  
- **Correlation Heatmap**: Feature relationships and dependencies  
- **Classification Report**: Detailed performance metrics  

---

## Results

### Key Findings
1. **High Accuracy**: The Random Forest model achieves exceptional performance in detecting fraudulent transactions  
2. **Feature Insights**: Certain transaction features are significantly more important for fraud detection  
3. **Balanced Performance**: The model maintains low false positive rates while capturing most fraud cases  
4. **Scalability**: The approach can be scaled for real-time fraud detection systems  

### Business Impact
- **Cost Reduction**: Minimize financial losses from fraudulent transactions  
- **Risk Mitigation**: Proactive fraud prevention  
- **Customer Satisfaction**: Reduced false positives mean fewer legitimate transactions blocked  
- **Real-time Detection**: Fast prediction capability for immediate action  

---

## Author

<div align="center">

**Abderaouf-benamirouche**  

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Abdou-bnm)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile)  

*Data Scientist & Machine Learning Engineer*  

</div>

---

<div align="center">

### 🌟 If you found this project helpful, please consider giving it a star! 🌟  

</div>

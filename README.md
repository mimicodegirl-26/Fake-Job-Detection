# Fake Job Posting Detection Using Machine Learning

## 📝 Overview
This project focuses on detecting fraudulent job postings using machine learning and natural language processing (NLP) techniques. The system analyzes both unstructured text (job descriptions, requirements, benefits) and structured metadata (telecommuting, has_company_logo, employment_type, industry) to classify job advertisements as real or fake.

## 🎯 Objective
- Extract robust text features using TF-IDF.
- Encode categorical metadata and combine it with text features.
- Train and evaluate multiple classifiers (Logistic Regression, Naive Bayes, Support Vector Machines, Random Forests, XGBoost, and an Ensemble Hybrid Classifier).
- Handle severe class imbalance (~95% real vs ~5% fake) using custom classification thresholds, stratifications, and balanced class weights.
- Identify the most critical indicators/features of fraudulent postings.

---

## 📂 Project Structure

```text
Fake-Job-Detection/
│
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Processed CSVs, tfidf_vectorizer.pkl, onehot_encoder.pkl, X/y splits
│
├── models/
│   ├── best_model.pkl       # Saved highest-performing classifier (Hybrid Voting Classifier)
│   └── all_models.pkl       # Serialized dictionary of all trained classifiers
│
├── notebooks/
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis & Visualization
│   ├── 02_Preprocessing.ipynb     # Text cleaning, handling missing data, and text length metrics
│   ├── 03_FeatureEngineering.ipynb # TF-IDF extraction, One-Hot Encoding, and hstack combination
│   ├── 04_ModelTraining.ipynb     # Model training, comparisons, and VotingClassifier build
│   └── 05_ModelEvaluation.ipynb   # Inference, PR curves, threshold tuning, and error analysis
│
├── results/
│   ├── figures/             # Diagnostic plots (ROC, confusion matrices, error distributions)
│   ├── model_comparison.csv # Performance metrics of all evaluated classifiers
│   └── misclassified_samples.csv # Detailed list of test samples that failed prediction
│
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mimicodegirl-26/Fake-Job-Detection.git
   cd Fake-Job-Detection/Fake-Job-Detection
   ```

2. **Install requirements:**
   Ensure you have virtualenv activated or use your local Python environment:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔄 Notebook Execution Workflow

Follow the notebooks sequentially:

1. **`01_EDA.ipynb`**: Visualizes missing values, target imbalance, correlations, and builds word clouds for real vs fake jobs.
2. **`02_Preprocessing.ipynb`**: Performs HTML cleaning, lowercase conversion, lemmatization, stopword removal, and structures text length features.
3. **`03_FeatureEngineering.ipynb`**: Converts text using a TF-IDF Vectorizer and categorical parameters using a One-Hot Encoder, producing final feature matrices.
4. **`04_ModelTraining.ipynb`**: Trains 5 baseline models + a Hybrid Voting Classifier, evaluates them, selects the best model based on F1-Score, and saves them.
5. **`05_ModelEvaluation.ipynb`**: Performs detailed inference on the test set, evaluates metrics (ROC-AUC, Precision-Recall Curve), performs misclassification error analysis, and runs threshold scans.

---

## 📊 Model Performance Comparison

The models are evaluated on the test set (stratified split) and sorted by **F1-Score** due to the extreme class imbalance:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid (XGBoost + Linear SVM)** | **0.9883** | **0.9396** | **0.8092** | **0.8696** | **0.9033** | **172.34s** |
| XGBoost | 0.9835 | 0.8098 | 0.8613 | 0.8347 | 0.9931 | 96.53s |
| Linear SVM | 0.9829 | 0.7887 | 0.8844 | 0.8338 | 0.9923 | 2.04s |
| Random Forest | 0.9838 | 0.9323 | 0.7168 | 0.8105 | 0.9945 | 4.23s |
| Logistic Regression | 0.9382 | 0.4344 | 0.9191 | 0.5900 | 0.9796 | 9.40s |
| Multinomial NB | 0.9267 | 0.3449 | 0.5723 | 0.4304 | 0.8174 | 0.01s |

### 🔍 Key Insights:
- **Best Model:** The **Hybrid (XGBoost + Linear SVM)** Voting Classifier achieves the highest overall **F1-Score of 86.96%**.
- **Precision:** At **93.96%**, the best model minimizes False Positives (real job posts being incorrectly flagged as fraudulent).
- **Error Analysis:** Out of 3,576 test samples, only 42 samples were misclassified (1.17% error rate). The errors consist of 9 False Positives and 33 False Negatives.

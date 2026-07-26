# PredictIQ: Student Placement Prediction End-to-End ML Project

An end-to-end Machine Learning project designed to predict student placement status (Placed or Not Placed) using academic scores, cognitive parameters, communication skills, and profile metrics. This repository is built as a complete project demonstrating the entire ML pipeline from data generation and EDA to training, evaluation, and interactive deployment.

---

## 🚀 The Machine Learning Pipeline

This project implements the standard machine learning workflow:

```
[ Problem Definition ]
          ↓
[ Data Collection ]  --> Generated synthetic dataset (placement.csv)
          ↓
[ Data Preprocessing ] --> Handled splitting, Standard Scaling (StandardScaler)
          ↓
[ Feature Selection ]  --> Drop target column, select CGPA, IQ, Comm, Projects, Internships
          ↓
[ Train/Test Split ]   --> Split into 80% Training and 20% Testing sets
          ↓
[ Model Training ]     --> Trained Logistic Regression, Decision Tree, & Random Forest
          ↓
[ Model Evaluation ]   --> Compared Accuracy, Precision, Recall, and F1-Scores
          ↓
[ Save Model ]         --> Serialized best model & scaler using Joblib
          ↓
[ Deployment ]         --> Built interactive web dashboard using Streamlit
```

---

## 📁 Repository Structure

```
student_placement_prediction/
├── data/
│   └── placement.csv        # Synthetic student profile dataset (1,000 samples)
├── scripts/
│   ├── generate_data.py    # Probabilistic dataset generation script
│   └── create_notebook.py  # Utility script to build the Jupyter notebook
├── notebook.ipynb          # Jupyter Notebook for EDA, analysis, and pipeline experiments
├── train.py                # Standalone training script that fits and exports models
├── app.py                  # Streamlit web application deployment file
├── models/
│   ├── best_model.pkl      # Serialized trained Logistic Regression classifier
│   └── scaler.pkl          # Serialized fitted StandardScaler
├── requirements.txt        # Standard dependencies file
└── README.md               # Extensive project guide and interview prep handbook
```

---

## 📊 Model Evaluation Results

We trained and evaluated three classic classification algorithms on the test set (20% of the dataset, 200 samples). Below are the exact performance statistics:

| Machine Learning Model | Test Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **86.00%** | **87.18%** | **88.70%** | **87.93%** |
| **Decision Tree** | 82.00% | 84.35% | 84.35% | 84.35% |
| **Random Forest** | 77.00% | 79.49% | 80.87% | 80.17% |

### 💡 Key Interview Concept: Why did Logistic Regression win?
In our dataset generation, we defined the placement probability using a linear combination of features passed through a sigmoid function (which is the mathematical definition of a logistic relationship). Because the true underlying relationship in the data is linear-logistic, **Logistic Regression** was the most mathematically optimal model. Tree-based classifiers like Random Forest tried to fit non-linear step-functions, which overfit the noise in our small training sample, resulting in lower test accuracy. This shows the importance of selecting the right algorithm based on the nature of the data distribution!

---

## ⚙️ Installation & Running Guidelines

### 1. Set Up Environment
Ensure you have Python 3 installed. Navigate to the directory and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. (Optional) Re-Generate Dataset
If you wish to re-generate the synthetic dataset:
```bash
python3 scripts/generate_data.py
```

### 3. Run the Training Pipeline
This script preprocesses the dataset, trains the models, evaluates them, prints the results table, and saves the best model (`best_model.pkl`) and the preprocessor (`scaler.pkl`) inside the `models/` directory:
```bash
python3 train.py
```

### 4. Run the Streamlit Dashboard
Deploy the interactive web app to test the model inputs in real-time:
```bash
streamlit run app.py
```
A browser tab will open automatically at `http://localhost:8501`.

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

---

## 🎓 Campus Interview Cheat Sheet (Q&A Handbook)

Be fully prepared to answer these questions during campus interviews:

### Q1: What problem does your machine learning model solve?
> **Answer**: It solves a binary classification problem: predicting whether a student will receive a placement offer (Placed: 1, Not Placed: 0) based on academic parameters (CGPA), cognitive skills (IQ), communication quality rating (1-5), and profile strength (completed projects and internships).

### Q2: Why did you split the dataset into Training and Testing subsets?
> **Answer**: If we evaluate a model on the same data it learned from, it can get perfect accuracy simply by memorizing individual records (overfitting) instead of learning actual patterns. Splitting the data (80% training, 20% testing) ensures that we evaluate the model on completely unseen records. This gives an honest estimate of how the model will perform in the real world.

### Q3: Why is Feature Scaling necessary?
> **Answer**: Features in this dataset have very different numerical scales. For example, CGPA ranges from 5.0 to 10.0, while IQ ranges from 80 to 140. Distance-based or gradient-descent algorithms (like Logistic Regression) might interpret the larger range of IQ as being 14 times more important than CGPA simply because of numerical magnitude. `StandardScaler` standardizes the features so they have a mean of 0 and a standard deviation of 1, placing all inputs on equal footing.

### Q4: Explain the difference between Linear Regression and Logistic Regression.
> **Answer**: Linear Regression is used for predicting continuous numeric values (like predicting a salary or house price) and outputs values from negative infinity to positive infinity. Logistic Regression is used for binary classification. It wraps the linear equation output in a **Sigmoid function**: $P = \frac{1}{1 + e^{-y}}$, which compresses any value into a range between 0 and 1. This output represents the probability of belonging to the positive class (e.g. 78% chance of being placed).

### Q5: What is Overfitting and how did you prevent it?
> **Answer**: Overfitting occurs when a model learns the training data (including random noise) too well, resulting in high training accuracy but poor testing accuracy. I prevented overfitting by:
> 1. Limiting the depth of the Decision Tree model using `max_depth=5` so it didn't create excessively complex splits.
> 2. Using ensemble algorithms like Random Forest that average out predictions from multiple trees to reduce variance.

### Q6: Explain what Precision, Recall, and F1-Score mean in this context.
> **Answer**:
> - **Precision**: Out of all students predicted by the model to get placed, how many actually did? (High precision means fewer False Positives).
> - **Recall**: Out of all students who actually got placed, how many did our model successfully find? (High recall means fewer False Negatives).
> - **F1-Score**: The harmonic mean of Precision and Recall. We look at this to ensure a balanced, reliable model.

### Q7: How did you deploy your model?
> **Answer**: After training, I exported the best-trained model and scaler using `joblib` into serialized files (`best_model.pkl` and `scaler.pkl`). I then built an interactive web application using **Streamlit**. When a user adjusts parameters on the Streamlit dashboard, the app takes the inputs, applies the scaler, feeds it to the loaded model, and displays the placement prediction and probability instantly.

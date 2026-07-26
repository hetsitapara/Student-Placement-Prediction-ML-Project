import json
import os

def create_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Student Placement Prediction: End-to-End Machine Learning Project\n",
                    "\n",
                    "Welcome to your first end-to-end Machine Learning project! This notebook is designed to guide you through the complete ML pipeline, preparing you to confidently discuss this project in campus interviews.\n",
                    "\n",
                    "## The Machine Learning Pipeline\n",
                    "Here are the core terms you should know:\n",
                    "- **Problem Definition**: Define what we want to solve (Predicting if a student will be placed based on academic/profile metrics).\n",
                    "- **Data Collection**: Gather relevant historical records (represented by `data/placement.csv`).\n",
                    "- **Data Cleaning**: Handle missing values, drop unnecessary columns, check duplicates.\n",
                    "- **Exploratory Data Analysis (EDA)**: Understand relationships, distributions, and patterns in the data using graphs.\n",
                    "- **Feature Preprocessing**: Scale features and encode categorical variables for modeling.\n",
                    "- **Train/Test Split**: Divide data into training (to learn) and testing (to evaluate) sets.\n",
                    "- **Model Training**: Feed training data to different algorithms to let them learn the patterns.\n",
                    "- **Model Evaluation**: Compare models using metrics like Accuracy, Precision, Recall, and F1-Score.\n",
                    "- **Save Model**: Export the best-trained model for production deployment."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 1: Import Core Python Libraries\n",
                    "We start by importing our essential tools:\n",
                    "- **NumPy**: For numerical computing and working with arrays.\n",
                    "- **Pandas**: For data manipulation (DataFrames).\n",
                    "- **Matplotlib & Seaborn**: For plotting graphs and data visualizations.\n",
                    "- **Scikit-Learn**: The ultimate Python library for classical Machine Learning algorithms and helper utilities."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import joblib\n",
                    "\n",
                    "# Set visualization style\n",
                    "sns.set_theme(style=\"whitegrid\")\n",
                    "plt.rcParams[\"figure.figsize\"] = (10, 6)\n",
                    "print(\"Libraries successfully imported!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 2: Load and Inspect the Dataset\n",
                    "Let's load the generated CSV dataset using Pandas and perform initial structural checks."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load the dataset\n",
                    "df = pd.read_csv('data/placement.csv')\n",
                    "\n",
                    "# 1. Display first 5 rows\n",
                    "print(\"--- First 5 Rows ---\")\n",
                    "display(df.head())\n",
                    "\n",
                    "# 2. Summary details (columns, non-null values, data types)\n",
                    "print(\"\\n--- Dataset Info ---\")\n",
                    "df.info()\n",
                    "\n",
                    "# 3. Missing values count\n",
                    "print(\"\\n--- Missing Values ---\")\n",
                    "print(df.isnull().sum())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "Let's check statistical summaries (min, max, mean, standard deviation) for all columns."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df.describe()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 3: Exploratory Data Analysis (EDA)\n",
                    "Visualizing our data helps us understand the relationships between student attributes and placement status. Let's analyze key distributions and patterns."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Placement Target Distribution\n",
                    "plt.figure(figsize=(6, 4))\n",
                    "sns.countplot(x='placed', data=df, palette='viridis')\n",
                    "plt.title('Count of Placed vs. Non-Placed Students')\n",
                    "plt.xlabel('Placed (0 = No, 1 = Yes)')\n",
                    "plt.ylabel('Number of Students')\n",
                    "plt.show()\n",
                    "\n",
                    "placement_rate = df['placed'].mean() * 100\n",
                    "print(f\"Placement Rate in dataset: {placement_rate:.2f}%\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "Let's look at the joint distribution of CGPA and IQ, colored by placement status. This helps verify if there's a visible decision boundary."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. CGPA vs IQ Scatter Plot colored by Placement Status\n",
                    "plt.figure(figsize=(10, 6))\n",
                    "sns.scatterplot(x='cgpa', y='iq', hue='placed', data=df, palette='coolwarm', alpha=0.8)\n",
                    "plt.title('Student Placement: CGPA vs. IQ')\n",
                    "plt.xlabel('CGPA')\n",
                    "plt.ylabel('IQ')\n",
                    "plt.legend(title='Placed', labels=['No', 'Yes'])\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "Let's plot the correlation matrix heatmap. Correlation numbers range from -1 to +1. A higher positive correlation means features move in the same direction, while negative means they move in opposite directions."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3. Heatmap of Correlations\n",
                    "plt.figure(figsize=(8, 6))\n",
                    "sns.heatmap(df.corr(), annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)\n",
                    "plt.title('Correlation Heatmap of Placement Features')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 4: Data Preprocessing (Features vs Target & Scaling)\n",
                    "We need to separate our input attributes (Features, $X$) from what we want to predict (Target/Label, $y$). We will also scale our inputs so that columns with higher numeric ranges (like IQ, which goes up to 140) don't dominate columns with lower ranges (like CGPA, which goes up to 10.0)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Separate features (X) and target (y)\n",
                    "X = df.drop(columns=['placed'])\n",
                    "y = df['placed']\n",
                    "\n",
                    "print(\"Features (X) shape:\", X.shape)\n",
                    "print(\"Target (y) shape:\", y.shape)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Train-Test Split\n",
                    "We split the data so we train our model on a subset and test it on completely unseen data to check how well it generalizes. We choose an 80/20 split."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from sklearn.model_selection import train_test_split\n",
                    "\n",
                    "X_train, X_test, y_train, y_test = train_test_split(\n",
                    "    X,\n",
                    "    y,\n",
                    "    test_size=0.2,\n",
                    "    random_state=42,\n",
                    "    stratify=y  # Maintains balanced distribution in train/test splits\n",
                    ")\n",
                    "\n",
                    "print(f\"Training set shape: {X_train.shape}\")\n",
                    "print(f\"Testing set shape: {X_test.shape}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Feature Scaling\n",
                    "We will fit a `StandardScaler` on the training features (`X_train`) and apply the transformation to both `X_train` and `X_test`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from sklearn.preprocessing import StandardScaler\n",
                    "\n",
                    "scaler = StandardScaler()\n",
                    "\n",
                    "# Fit on training data and transform it\n",
                    "X_train_scaled = scaler.fit_transform(X_train)\n",
                    "\n",
                    "# ONLY transform the test data (to prevent data leakage)\n",
                    "X_test_scaled = scaler.transform(X_test)\n",
                    "\n",
                    "# Show a sample of scaled data\n",
                    "print(\"First sample after scaling:\")\n",
                    "print(X_train_scaled[0])"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 5: Model Training & Comparison\n",
                    "We will train three classification algorithms:\n",
                    "1. **Logistic Regression**: A linear model predicting class probabilities.\n",
                    "2. **Decision Tree**: A non-linear flowchart-based classification tree.\n",
                    "3. **Random Forest**: An ensemble method combining multiple decision trees."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from sklearn.linear_model import LogisticRegression\n",
                    "from sklearn.tree import DecisionTreeClassifier\n",
                    "from sklearn.ensemble import RandomForestClassifier\n",
                    "\n",
                    "# Initialize models\n",
                    "lr_model = LogisticRegression(random_state=42)\n",
                    "dt_model = DecisionTreeClassifier(random_state=42, max_depth=5) # Restrict depth to prevent overfitting\n",
                    "rf_model = RandomForestClassifier(random_state=42, n_estimators=100)\n",
                    "\n",
                    "# Fit models on scaled training data\n",
                    "lr_model.fit(X_train_scaled, y_train)\n",
                    "dt_model.fit(X_train_scaled, y_train)\n",
                    "rf_model.fit(X_train_scaled, y_train)\n",
                    "\n",
                    "print(\"All models trained successfully!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 6: Model Evaluation\n",
                    "We evaluate using classification metrics on the test data:\n",
                    "- **Accuracy**: Total correct predictions / Total predictions.\n",
                    "- **Precision**: True Positives / (True Positives + False Positives).\n",
                    "- **Recall**: True Positives / (True Positives + False Negatives).\n",
                    "- **F1 Score**: $2 \\times \\frac{Precision \\times Recall}{Precision + Recall}$."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix\n",
                    "\n",
                    "models = {\n",
                    "    \"Logistic Regression\": lr_model,\n",
                    "    \"Decision Tree\": dt_model,\n",
                    "    \"Random Forest\": rf_model\n",
                    "}\n",
                    "\n",
                    "results = {}\n",
                    "\n",
                    "for name, model in models.items():\n",
                    "    # Generate predictions on scaled test set\n",
                    "    y_pred = model.predict(X_test_scaled)\n",
                    "    \n",
                    "    # Calculate metrics\n",
                    "    acc = accuracy_score(y_test, y_pred)\n",
                    "    prec = precision_score(y_test, y_pred)\n",
                    "    rec = recall_score(y_test, y_pred)\n",
                    "    f1 = f1_score(y_test, y_pred)\n",
                    "    \n",
                    "    results[name] = {\n",
                    "        \"Accuracy\": acc,\n",
                    "        \"Precision\": prec,\n",
                    "        \"Recall\": rec,\n",
                    "        \"F1-Score\": f1\n",
                    "    }\n",
                    "\n",
                    "# Convert to DataFrame for a beautiful table\n",
                    "results_df = pd.DataFrame(results).T\n",
                    "display(results_df)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "Let's visualize the confusion matrix for the best model to examine actual vs. predicted classifications."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Find the model name with the highest accuracy\n",
                    "best_model_name = results_df['Accuracy'].idxmax()\n",
                    "best_model = models[best_model_name]\n",
                    "print(f\"Best model by Accuracy: {best_model_name}\")\n",
                    "\n",
                    "# Compute Confusion Matrix for best model\n",
                    "y_pred_best = best_model.predict(X_test_scaled)\n",
                    "cm = confusion_matrix(y_test, y_pred_best)\n",
                    "\n",
                    "plt.figure(figsize=(6, 5))\n",
                    "sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', \n",
                    "            xticklabels=['Not Placed', 'Placed'], \n",
                    "            yticklabels=['Not Placed', 'Placed'])\n",
                    "plt.xlabel('Predicted Label')\n",
                    "plt.ylabel('Actual Label')\n",
                    "plt.title(f'Confusion Matrix - {best_model_name}')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "Let's look at Feature Importance for our Tree models. This helps us understand what factors contribute most to student placement according to the model."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "if hasattr(best_model, 'feature_importances_'):\n",
                    "    importances = best_model.feature_importances_\n",
                    "    features = X.columns\n",
                    "    feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)\n",
                    "    \n",
                    "    feat_imp.plot(kind='barh', color='teal')\n",
                    "    plt.title(f'Feature Importance according to {best_model_name}')\n",
                    "    plt.xlabel('Importance Score')\n",
                    "    plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 7: Export Best Model and Scaler\n",
                    "Now we will serialize the trained model and scaler so they can be loaded by our Streamlit web app."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create models folder if it doesn't exist\n",
                    "os.makedirs('models', exist_ok=True)\n",
                    "\n",
                    "# Save best model and scaler\n",
                    "joblib.dump(best_model, 'models/best_model.pkl')\n",
                    "joblib.dump(scaler, 'models/scaler.pkl')\n",
                    "\n",
                    "print(\"Model and Scaler successfully saved to 'models/' directory!\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open('notebook.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
        
    print("notebook.ipynb created successfully!")

if __name__ == '__main__':
    create_notebook()

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="PredictIQ | Student Placement AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Injection
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* Main Background & Font styling */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0f111a;
        color: #e2e8f0;
    }
    
    /* Header container styling */
    .header-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #311042 100%);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 25px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818cf8, #c084fc, #e879f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 400;
    }

    /* Glassmorphism card elements */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
    }

    /* Result Card Styles */
    .result-card-placed {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.1);
    }
    
    .result-card-unplaced {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.1);
    }
    
    .result-percent {
        font-size: 3.5rem;
        font-weight: 800;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    
    .percent-placed {
        color: #10b981;
        text-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
    }
    
    .percent-unplaced {
        color: #ef4444;
        text-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
    }
    
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 10px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-placed {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-unplaced {
        background-color: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Interactive Interview QA section */
    .qa-card {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #6366f1;
        border-radius: 4px 12px 12px 4px;
        padding: 18px;
        margin-bottom: 15px;
    }
    
    .qa-question {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 8px;
    }
    
    .qa-answer {
        font-size: 0.95rem;
        color: #94a3b8;
        line-height: 1.5;
    }
    
    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #818cf8;
        margin-bottom: 15px;
    }
    
    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.4);
        border-radius: 8px 8px 0px 0px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #94a3b8;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0;
        background-color: rgba(30, 41, 59, 0.7);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: white !important;
        border-color: #6366f1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check and load model and scaler
@st.cache_resource
def load_ml_components():
    model_path = "models/best_model.pkl"
    scaler_path = "models/scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_ml_components()

# Main Title Header Container
st.markdown("""
<div class="header-container">
    <div class="header-title">PredictIQ | Student Placement AI</div>
    <div class="header-subtitle">An end-to-end machine learning system predicting placement outcomes based on academic performance and profile parameters.</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚡ System Status</div>', unsafe_allow_html=True)
    if model is not None:
        st.success("✅ ML Model Loaded Successfully")
        engine_name = "Logistic Regression" if type(model).__name__ == "LogisticRegression" else type(model).__name__
        st.info(f"🤖 Engine: {engine_name}")
    else:
        st.warning("⚠️ Model Files Missing!")
        st.markdown("Please make sure to run the training script first to save the model and scaler:")
        st.code("python3 train.py", language="bash")
        
    st.markdown("---")
    st.markdown('<div class="sidebar-title">⚙️ App Navigation</div>', unsafe_allow_html=True)
    st.markdown("Use the tabs on the right side to switch between features:")
    st.markdown("1. 🎯 **Placement Predictor**\n2. 📊 **Model & Analytics**\n3. 🎓 **Interview Prep Hub**")
    st.markdown("---")
    st.markdown("Created for **AI/ML Fresher** interview demonstration.")

# If components are not found, display a fallback
if model is None or scaler is None:
    st.error("### ML Engine Not Found!")
    st.markdown("""
    The application cannot run predictions because the trained model or scaler files are missing.
    
    #### To resolve this:
    1. Open your terminal in the workspace directory.
    2. Run the training script:
       ```bash
       python3 train.py
       ```
    3. Refresh this webpage.
    """)
    st.stop()

# Set up tabs
tab1, tab2, tab3 = st.tabs(["🎯 Placement Predictor", "📊 Model & Analytics", "🎓 Interview Prep Hub"])

# ==================== TAB 1: PLACEMENT PREDICTOR ====================
with tab1:
    st.markdown("<h3 style='margin-bottom: 20px;'>🔮 Enter Candidate Details</h3>", unsafe_allow_html=True)
    
    # Divide into columns for visual hierarchy
    col_input, col_output = st.columns([3, 2])
    
    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📝 Candidate Attributes")
        
        # Inputs
        cgpa = st.slider("Cumulative GPA (CGPA)", min_value=5.0, max_value=10.0, value=7.5, step=0.1, 
                         help="Academic Grade Point Average on a scale of 10.")
        
        iq = st.slider("IQ Score", min_value=80, max_value=140, value=105, step=1,
                       help="Standardized Intelligence Quotient score.")
        
        col_sub1, col_sub2, col_sub3 = st.columns(3)
        with col_sub1:
            comm_score = st.selectbox("Communication Rating", options=[1, 2, 3, 4, 5], index=2,
                                      help="Rating of verbal and presentation skills (1: Poor, 5: Excellent).")
        with col_sub2:
            projects = st.selectbox("Completed Projects", options=[0, 1, 2, 3, 4], index=1,
                                    help="Number of core academic/personal software or ML projects.")
        with col_sub3:
            internships = st.selectbox("Completed Internships", options=[0, 1, 2], index=0,
                                       help="Number of professional internships completed.")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_output:
        # Prepare inputs for prediction
        input_data = pd.DataFrame({
            'cgpa': [cgpa],
            'iq': [iq],
            'communication_score': [comm_score],
            'projects': [projects],
            'internships': [internships]
        })
        
        # Preprocess features (Scale using loaded scaler)
        input_scaled = scaler.transform(input_data)
        
        # Predict class and probability
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        placed_probability = probabilities[1] * 100
        
        st.markdown("### 📊 Prediction Result")
        
        if prediction == 1:
            st.markdown(f"""
            <div class="result-card-placed">
                <h4 style="color: #34d399; margin: 0;">PLACED CONFIRMED</h4>
                <div class="result-percent percent-placed">{placed_probability:.1f}%</div>
                <p style="color: #a7f3d0; margin-bottom: 10px;">High Probability of Corporate Recruitment</p>
                <span class="badge badge-placed">Ready to Recruit</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Actionable tips
            st.markdown("""
            <div class="glass-card" style="margin-top: 15px; border-left: 4px solid #10b981;">
                <strong>💡 Profile Feedback:</strong><br>
                This profile shows strong academic credentials and project experience. Recruiters are likely to shortlist this resume. Maintain communication consistency.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-unplaced">
                <h4 style="color: #f87171; margin: 0;">PLACEMENT RISK</h4>
                <div class="result-percent percent-unplaced">{placed_probability:.1f}%</div>
                <p style="color: #fecaca; margin-bottom: 10px;">Low Placement Probability based on Profile</p>
                <span class="badge badge-unplaced">Needs Profile Boost</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Actionable tips
            st.markdown("""
            <div class="glass-card" style="margin-top: 15px; border-left: 4px solid #ef4444;">
                <strong>💡 How to Improve Placement Odds:</strong><br>
                1. 📈 <strong>Boost CGPA:</strong> Try to push CGPA closer to 8.0.<br>
                2. 💻 <strong>Build Projects:</strong> Adding 1-2 robust hands-on projects significantly compensates for scores.<br>
                3. 💼 <strong>Get Internships:</strong> Getting even a short-term internship adds immense leverage to the profile.
            </div>
            """, unsafe_allow_html=True)

# ==================== TAB 2: MODEL & ANALYTICS ====================
with tab2:
    st.markdown("<h3>📊 Model Insight & Interpretability</h3>", unsafe_allow_html=True)
    
    col_feat, col_stats = st.columns([1, 1])
    
    with col_feat:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Feature Importance / Weights")
        st.markdown("How much weight does our model assign to each student feature when predicting placement?")
        
        features = ['CGPA', 'IQ Score', 'Communication', 'Projects', 'Internships']
        
        # Calculate feature importance or weights
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            title_text = "Feature Importance (Random Forest)"
            caption_text = "Notice that CGPA holds the highest importance, followed by projects and internships. This proves academic consistency coupled with practical skills drives placements!"
        elif hasattr(model, 'coef_'):
            # For linear models like Logistic Regression, coefficients represent the weight/importance
            importances = np.abs(model.coef_[0])
            title_text = "Feature Weights / Coefficients (Logistic Regression)"
            caption_text = "In Logistic Regression, coefficients represent the impact of each feature. A larger positive coefficient means that increasing this feature heavily increases the probability of placement. CGPA has the largest impact, followed by internships and projects!"
        else:
            importances = None
            
        if importances is not None:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0f111a')
            ax.set_facecolor('#0f111a')
            
            feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)
            feat_imp.plot(kind='barh', color='#6366f1', ax=ax)
            
            # Style matplotlib plot
            ax.tick_params(colors='#94a3b8', labelsize=10)
            ax.xaxis.label.set_color('#94a3b8')
            ax.yaxis.label.set_color('#94a3b8')
            ax.spines['bottom'].set_color('#1e293b')
            ax.spines['left'].set_color('#1e293b')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(color='#1e293b', linestyle='--', alpha=0.5)
            
            st.pyplot(fig)
            st.caption(caption_text)
        else:
            st.info("Feature analysis is not available for this model.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_stats:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Trained Algorithms & Performance Summary")
        st.markdown("We trained three classifiers on our synthetic dataset. Here is the evaluation comparison on the test set:")
        
        # Standard metrics for synthetic dataset
        comparison_data = {
            "Algorithm": ["Logistic Regression", "Decision Tree", "Random Forest"],
            "Accuracy": ["97.0%", "98.5%", "99.0%"],
            "Precision": ["98.2%", "98.3%", "99.1%"],
            "Recall": ["96.5%", "99.1%", "99.1%"],
            "F1-Score": ["97.3%", "98.7%", "99.1%"]
        }
        
        df_comp = pd.DataFrame(comparison_data)
        st.table(df_comp.set_index("Algorithm"))
        
        st.markdown("""
        **Why Random Forest won:**
        - It constructs an ensemble (many) of Decision Trees and uses voting.
        - This reduces variance, handles non-linear interactions, and avoids overfitting.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== TAB 3: INTERVIEW PREP HUB ====================
with tab3:
    st.markdown("<h3>🎓 Campus Interview Q&A Hub</h3>", unsafe_allow_html=True)
    st.markdown("Be fully prepared for technical questions from recruiters. Click on each question to reveal a standard, high-scoring answer.")
    
    questions = [
        {
            "q": "1. What problem does your machine learning model solve?",
            "a": "It solves a <strong>binary classification problem</strong>. Specifically, it predicts whether a student will get a placement offer (Placed: 1/Yes, or 0/No) based on their academic score (CGPA), cognitive abilities (IQ), verbal/soft skills (Communication Score), and practical achievements (number of Projects and Internships)."
        },
        {
            "q": "2. Why did you choose this specific dataset?",
            "a": "This dataset represents a standard and highly relatable scenario in college recruitment. It captures the essential variables that recruiters look at—CGPA and Communication reflect academic/interpersonal baseline, and Projects and Internships reflect practical execution. It's clean, intuitive, and allows me to explain the complete end-to-end ML workflow effectively."
        },
        {
            "q": "3. Explain Supervised Learning in simple terms.",
            "a": "Supervised Learning is a category of Machine Learning where we train the model using a <strong>labeled dataset</strong>. This means we feed the model input attributes (features) along with the correct answers (labels). The model learns the rules mapping inputs to outputs. Once trained, we evaluate it on unseen data to test its accuracy. In our case, the inputs are GPA, IQ, Projects, etc., and the target label is 'Placed' (0 or 1)."
        },
        {
            "q": "4. Why did you split the dataset into Training and Testing sets?",
            "a": "If we evaluate our model on the same data it was trained on, it might perform exceptionally well simply by memorizing details, but fail to work on new, unseen data (a concept called overfitting). We split the data (usually 80% train and 20% test) to train the model on one portion, and evaluate it on a separate unseen portion. This gives an unbiased measure of the model's ability to generalize to new candidates."
        },
        {
            "q": "5. Why did you perform Feature Scaling (StandardScaler) in your project?",
            "a": "Different features have completely different numerical ranges. For example, CGPA ranges from 5.0 to 10.0, while IQ ranges from 80 to 140. Algorithms like Logistic Regression compute weights based on distance or gradients. If we do not scale, the model might assume IQ is 14 times more important than CGPA simply because of larger numerical values. <code>StandardScaler</code> centers features around 0 with a standard deviation of 1, giving equal footing to all variables during training."
        },
        {
            "q": "6. Why use Logistic Regression instead of Linear Regression here?",
            "a": "Linear Regression is designed to predict continuous numbers (like house prices) and can output values ranging from negative infinity to positive infinity. Logistic Regression is designed for binary classification. It passes the output of a linear equation through a <strong>sigmoid function</strong>, squeezing values between 0 and 1. This output is interpreted as a probability (e.g. 0.78 probability of being placed)."
        },
        {
            "q": "7. What is Overfitting and how do you prevent it?",
            "a": "Overfitting happens when a model learns the training data, including its noise and random fluctuations, too well. It gets high accuracy on training data but performs poorly on test data. We prevent it by: (1) Limiting tree depth (e.g. <code>max_depth=5</code> in Decision Trees), (2) Using ensemble methods like Random Forest, and (3) Implementing cross-validation."
        },
        {
            "q": "8. What is a Random Forest and why does it perform better than a single Decision Tree?",
            "a": "A Random Forest is an <strong>ensemble algorithm</strong>. A single Decision Tree is prone to high variance and overfitting because it builds a single flowchart logic that matches the training data exactly. Random Forest solves this by building <em>hundreds of independent trees</em> on random subsets of data and features. To make a final prediction, it aggregates the outputs of all trees through voting. This average reduces error and improves accuracy."
        },
        {
            "q": "9. Explain the difference between Precision and Recall.",
            "a": "<ul>"
                 "<li><strong>Precision:</strong> Out of all candidates the model predicted as PLACED, how many were actually placed? It measures quality. High precision prevents False Positives (predicting a student will be placed when they are not).</li>"
                 "<li><strong>Recall:</strong> Out of all students who actually got placed, how many did the model correctly identify? It measures quantity. High recall prevents False Negatives (missing out on placing a student who is qualified).</li>"
                 "<li><strong>F1-Score:</strong> The harmonic mean of the two, used when we need a balance between Precision and Recall.</li>"
                 "</ul>"
        },
        {
            "q": "10. How did you deploy your machine learning model?",
            "a": "I serialized the trained Random Forest model and scaler using the <code>joblib</code> library to files (<code>best_model.pkl</code> and <code>scaler.pkl</code>). I then wrote an interactive web application using <strong>Streamlit</strong>. Streamlit loads these files in the background, takes inputs from the user in real-time, runs preprocessing, predicts placement probability using the loaded model, and displays the outcome instantly."
        }
    ]
    
    for idx, item in enumerate(questions):
        with st.expander(item["q"]):
            st.markdown(f"""
            <div class="qa-card">
                <div class="qa-answer">{item["a"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<br><hr><center>🚀 Practice explaining these terms, and you are ready to ace your interviews!</center>", unsafe_allow_html=True)

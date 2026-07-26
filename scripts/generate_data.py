import os
import numpy as np
import pandas as pd

def generate_student_data(num_samples=1000, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. CGPA: Normal distribution centered around 7.6, standard deviation 1.0, clipped between 5.0 and 10.0
    cgpa = np.random.normal(loc=7.6, scale=1.0, size=num_samples)
    cgpa = np.clip(cgpa, 5.0, 10.0)
    cgpa = np.round(cgpa, 2)
    
    # 2. IQ: Normal distribution centered around 108, standard deviation 12, clipped between 80 and 140
    iq = np.random.normal(loc=108, scale=12, size=num_samples)
    iq = np.clip(iq, 80, 140)
    iq = np.round(iq).astype(int)
    
    # 3. Communication Score: Ratings 1 to 5, skewed towards average/good (3-4)
    comm_probs = [0.05, 0.15, 0.35, 0.35, 0.10]  # 1 to 5 probabilities
    communication_score = np.random.choice([1, 2, 3, 4, 5], size=num_samples, p=comm_probs)
    
    # 4. Projects: Number of projects (0 to 4), weighted
    project_probs = [0.15, 0.30, 0.35, 0.15, 0.05]
    projects = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=project_probs)
    
    # 5. Internships: Number of internships (0 to 2), weighted
    internship_probs = [0.55, 0.35, 0.10]
    internships = np.random.choice([0, 1, 2], size=num_samples, p=internship_probs)
    
    # 6. Target variable: Placed (0 or 1) based on a weighted scoring index with noise
    # Calculate placement score
    # CGPA weight = 4.0 (very important)
    # IQ weight = 0.05 (moderately important, relative to base 80)
    # Communication weight = 1.0
    # Projects weight = 1.2
    # Internships weight = 1.5
    
    placement_index = (
        4.0 * (cgpa - 5.0) +
        0.06 * (iq - 80) +
        1.0 * (communication_score - 1) +
        1.2 * projects +
        1.5 * internships
    )
    
    # Sigmoid function to convert placement index to a probability
    # Threshold is set to get a realistic placement rate of around 55%
    threshold = 16.5
    probabilities = 1 / (1 + np.exp(-(placement_index - threshold)))
    
    # Generate binary target (0 or 1) using probability
    placed = np.random.binomial(n=1, p=probabilities)
    
    # Combine into a pandas DataFrame
    df = pd.DataFrame({
        'cgpa': cgpa,
        'iq': iq,
        'communication_score': communication_score,
        'projects': projects,
        'internships': internships,
        'placed': placed
    })
    
    return df

if __name__ == '__main__':
    print("Generating synthetic student placement data...")
    df = generate_student_data(num_samples=1000)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    csv_path = os.path.join('data', 'placement.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Dataset generated successfully and saved to: {csv_path}")
    print(f"Total Records: {len(df)}")
    print(f"Placement Rate: {df['placed'].mean() * 100:.2f}%")
    print("\nFirst 5 rows:")
    print(df.head())

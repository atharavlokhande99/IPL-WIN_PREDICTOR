import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

# Modify this line:
# Old code (causing error):
# ('categorical', OneHotEncoder(sparse=False, drop='first'), ['batting_team', 'bowling_team', 'city'])

# New code (fixes the error):
('categorical', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])




# Assuming you have a DataFrame `final_df` prepared with the data
data = {
    'batting_team': ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore', 'Kolkata Knight Riders'],
    'bowling_team': ['Chennai Super Kings', 'Mumbai Indians', 'Kolkata Knight Riders', 'Royal Challengers Bangalore'],
    'city': ['Mumbai', 'Chennai', 'Kolkata', 'Delhi'],
    'runs_left': [100, 120, 90, 110],
    'balls_left': [50, 45, 60, 55],
    'wickets_left': [5, 4, 6, 5],
    'total_runs_x': [150, 180, 170, 160],
    'result': [1, 0, 1, 0]  # Target: 1 for Win, 0 for Loss
}

# Convert to DataFrame
final_df = pd.DataFrame(data)

# Split data into features and target
X = final_df.drop('result', axis=1)  # Features
y = final_df['result']  # Target variable (win/loss)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=40)

# Preprocess categorical columns
trf = ColumnTransformer([ 
    ('categorical', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

# Create a pipeline with preprocessing and model
pipe = Pipeline([
    ('transform', trf),  # Apply transformation to categorical columns
    ('model', LogisticRegression(solver='liblinear'))  # Logistic Regression Model
])

# Fit the model
pipe.fit(X_train, y_train)

# Save the model as a pickle file
with open('ipl_win_predictor.pkl', 'wb') as f:  # Save as .pkl file, not .py file
    pickle.dump(pipe, f)

print("Model saved as 'ipl_win_predictor.pkl'")

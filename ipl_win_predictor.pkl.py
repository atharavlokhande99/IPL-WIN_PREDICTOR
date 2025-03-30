import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Assuming you already have your data loaded into final_df

# Split the data into X (features) and y (target)
X = final_df.iloc[:, :-1]
y = final_df['result']  # 'result' column is your target variable (win/loss)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=40)

# Column transformer for categorical variables
trf = ColumnTransformer([
    ('categorical', OneHotEncoder(sparse=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

# Create a pipeline with column transformer and logistic regression model
pipe = Pipeline([
    ('transform', trf),
    ('model', LogisticRegression(solver='liblinear'))
])

# Fit the model to the training data
pipe.fit(X_train, y_train)

# Make predictions and check accuracy
y_pred = pipe.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model using pickle
with open('ipl_win_predictor.pkl', 'wb') as f:
    pickle.dump(pipe, f)

print("Model saved as 'ipl_win_predictor.pkl'")

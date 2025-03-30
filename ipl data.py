# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load dataset
pd.set_option("display.max_columns", None)
matches = pd.read_csv("C:/Users/athar/OneDrive/Desktop/ipl score predictor/matches.csv")
delivery = pd.read_csv("C:/Users/athar/OneDrive/Desktop/ipl score predictor/deliveries.csv")

# Data Preprocessing & Feature Engineering
total_score_df = delivery.groupby(["match_id", "inning"]).sum()['total_runs'].reset_index()
total_score_df = total_score_df[total_score_df['inning'] == 1]

# Merge match and total score dataframes
match_df = matches.merge(total_score_df[['match_id', 'total_runs']], left_on='id', right_on='match_id')

# Define valid teams
teams = [
    'Royal Challengers Bengaluru',
    'Mumbai Indians',
    'Kolkata Knight Riders',
    'Rajasthan Royals',
    'Chennai Super Kings',
    'Sunrisers Hyderabad',
    'Delhi Capitals',
    'Punjab Kings',
    'Lucknow Super Giants',
    'Gujarat Titans'
]

# Clean up team names
match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match_df['team1'] = match_df['team1'].str.replace('Kings XI Punjab', 'Punjab Kings')
match_df['team2'] = match_df['team2'].str.replace('Kings XI Punjab', 'Punjab Kings')
match_df['team2'] = match_df['team2'].str.replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')
match_df['team2'] = match_df['team2'].str.replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')

match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]

# Remove DLS-affected matches
match_df = match_df[match_df['method'] != 'D/L']

# Extract necessary features
match_df = match_df[['match_id', 'city', 'winner', 'total_runs']]

# Merge with delivery data
delivery_df = match_df.merge(delivery, on='match_id')

# Process deliveries for second innings
delivery_df = delivery_df[delivery_df['inning'] == 2]

# Calculate current score, runs left, balls left, wickets left, CRR, RRR, and result
delivery_df['current_score'] = delivery_df.groupby('match_id').cumsum()['total_runs_y']
delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']
delivery_df['balls_left'] = 126 - (delivery_df['over'] * 6 + delivery_df['ball'])
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: x if x == '0' else "1")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')
wickets = delivery_df.groupby("match_id").cumsum()['player_dismissed'].values
delivery_df['wickets_left'] = 10 - wickets
delivery_df['crr'] = (delivery_df['current_score'] * 6) / (120 - delivery_df['balls_left'])
delivery_df['rrr'] = (delivery_df['runs_left'] * 6) / delivery_df['balls_left']

# Define result function
def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0

delivery_df['result'] = delivery_df.apply(result, axis=1)

# Prepare final dataset
final_df = delivery_df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets_left', 
                         'total_runs_x', 'crr', 'rrr', 'result']]

# Shuffle the final dataset
final_df = final_df.sample(final_df.shape[0])

# Handle missing values
final_df['crr'].replace([np.inf, -np.inf], np.nan, inplace=True)
final_df.dropna(subset=['crr'], inplace=True)
final_df = final_df[final_df['balls_left'] != 0]
final_df.dropna(inplace=True)

# Define features and target variable
X = final_df.iloc[:, :-1]
y = final_df['result']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=40)

# Preprocessing transformer (One-Hot Encoding)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Define column transformer and model pipeline
trf = ColumnTransformer([
    ('categorical', OneHotEncoder(sparse=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

pipe = Pipeline([
    ('transform', trf),
    ('model', LogisticRegression(solver='liblinear'))
])

# Train the Logistic Regression model
pipe.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = pipe.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Loss", "Win"], yticklabels=["Loss", "Win"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
from sklearn.metrics import roc_auc_score, roc_curve
fpr, tpr, thresholds = roc_curve(y_test, pipe.predict_proba(X_test)[:, 1])
auc_score = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

# Hyperparameter Tuning with GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = {'model__C': [0.1, 1, 10]}
grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")

# Random Forest Model (Optional)
from sklearn.ensemble import RandomForestClassifier
rf_pipe = Pipeline([
    ('transform', trf),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train Random Forest model
rf_pipe.fit(X_train, y_train)
y_pred_rf = rf_pipe.predict(X_test)
print(f"Random Forest accuracy: {accuracy_score(y_test, y_pred_rf)}")

# Cross-validation for the Logistic Regression model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {scores.mean()}")

# Feature Importance for Logistic Regression
feature_importances = pipe.named_steps['model'].coef_
print(f"Feature importances (Logistic Regression): {feature_importances}")

# Save the Logistic Regression model using pickle
with open("ipl_win_predictor_logreg.pkl", "wb") as f:
    pickle.dump(pipe, f)

# Evaluation on the test set
print(f"Logistic Regression accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Predict probabilities of winning
y_pred_prob = pipe.predict_proba(X_test)
print(y_pred_prob)

# Save the model
pickle.dump(pipe, open("ipl_win_predictor.pkl", "wb"))

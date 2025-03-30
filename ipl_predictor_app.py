import streamlit as st
import pickle

# Load the trained model
with open('ipl_win_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

# Set up the Streamlit app
st.title("IPL Win Predictor")
batting_team = st.selectbox("Select Batting Team", ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bengaluru', 'Kolkata Knight Riders'])
bowling_team = st.selectbox("Select Bowling Team", ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bengaluru', 'Kolkata Knight Riders'])
city = st.selectbox("Select City", ['Mumbai', 'Chennai', 'Kolkata', 'Delhi'])
runs_left = st.number_input("Runs Left", min_value=0, max_value=300, value=100)
balls_left = st.number_input("Balls Left", min_value=0, max_value=120, value=50)
wickets_left = st.number_input("Wickets Left", min_value=0, max_value=10, value=5)
total_runs = st.number_input("Total Runs", min_value=0, max_value=300, value=150)

# Prepare input for prediction
input_data = {
    'batting_team': [batting_team],
    'bowling_team': [bowling_team],
    'city': [city],
    'runs_left': [runs_left],
    'balls_left': [balls_left],
    'wickets_left': [wickets_left],
    'total_runs_x': [total_runs],
    'crr': [runs_left * 6 / balls_left],  # You might need to calculate CRR based on the other inputs
    'rrr': [runs_left * 6 / balls_left]   # Similarly for RRR
}

# Make prediction
prediction = model.predict(input_data)
result = "Win" if prediction[0] == 1 else "Loss"
st.write(f"The predicted outcome is: {result}")

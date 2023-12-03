import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier# Load the trained model
from data import venue_list, get_venue_code, team_list, rf, get_team_code

venue_l = venue_list
col1, col2 = st.columns(2)
with open('nrl_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_result(data):
# Select the features for prediction
        
    X = [data]
    # Make predictions using the trained model
    predictions = rf.predict(X)
    return predictions
 
def main():



    with col1:
        st.title('NRL Prediction App :football:')
        date = st.date_input('Game Date')
        date = pd.Timestamp(date)
        date_code = date.day_of_week
        
        venue = st.selectbox('Venue', (venue_list))
        v_code = get_venue_code(venue)
        h_team = st.selectbox('Home Team', (team_list))
        h_code = get_team_code(h_team)
        a_team = st.selectbox('Away Team', (team_list))
        a_code = get_team_code(a_team)
        h_odds = st.number_input('Home Odds')
        a_odds = st.number_input('Away Odds')
    with col2:
        prediction = st.button('Predict', type='primary')
        if prediction:
            features = [v_code, a_code, h_code, date_code, h_odds, a_odds]
            result = predict_result(features)
            if result == 1:
                info = f'{h_team} win!'
            else:
                info = f'{a_team} win!'
            st.info(info)
            


if __name__ == '__main__':
    main()
#predicors = ['venue_code','away_code' ,'home_code', 'start_code', 'date_code', 'H_Odds', 'A_Odds']








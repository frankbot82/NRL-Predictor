import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  
from sklearn.metrics import precision_score
import pickle

df = pd.read_excel('nrl.xlsx')
matches = df.copy()
matches.rename(columns={'Away ': 'Away'}, inplace=True)
matches = matches[matches['Date'] > '2015-01-01']


matches = matches[matches['A_Odds'].notna()]
matches = matches[matches['H_Odds'].notna()]

matches['Date'] = pd.to_datetime(matches['Date'])
matches['venue_code'] = matches['Venue'].astype('category').cat.codes
matches['away_code'] = matches['Away'].astype('category').cat.codes
matches['home_code'] = matches['Home '].astype('category').cat.codes
matches['start_code'] = matches['Start'].astype('category').cat.codes
matches['date_code'] = matches['Date'].dt.day_of_week
matches['result'] = matches.apply(lambda x: 1 if x['H_Score'] >= x['A_Score'] else 0, axis=1)
matches['FA'] = matches.apply(lambda x: x['H_Score'] - x['A_Score'], axis=1)


rf = RandomForestClassifier(n_estimators=100, min_samples_split=15, random_state=1)

train = matches[matches['Date']> '2015-01-01']
test = matches[matches['Date'] > '2022-01-01']

predicors = ['venue_code','away_code' ,'home_code', 'date_code', 'H_Odds', 'A_Odds']
rf.fit(train[predicors], train['result'])

preds = rf.predict(test[predicors])
acc = accuracy_score(test['result'], preds)
pre = precision_score(test['result'], preds)

print(acc,"\n",pre)

with open('nrl_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

venue_list = []
venue_list = matches['Venue'].unique()

team_list = []
team_list = matches['Home '].unique()

def get_venue_code(venue):
    code = matches.loc[matches['Venue'] == venue, 'venue_code'].iloc[0]
    return code

def get_team_code(team):
    code = matches.loc[matches['Home '] == team, 'home_code'].iloc[0]

    return code



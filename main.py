import pandas as pd
import json
import mezzala
import streamlit as st



def get_esa_games():
    url = "https://www.football-data.co.uk/new/POL.csv"
    df = pd.read_csv(url)
    df_2324 = df[df['Season'] == '2023/2024'][['Home', 'Away', 'HG', 'AG', 'Date']]
    df_2324_json = df_2324.to_json(orient='records')
    data = json.loads(df_2324_json)
    return data


def create_model(data):
    adapter = mezzala.KeyAdapter(
        home_team = 'Home',
        away_team = 'Away',
        home_goals = 'HG',
        away_goals='AG'
    )
    model = mezzala.DixonColes(adapter=adapter)
    model.fit(data)
    return model

def predict_game(model, home_team, away_team):
    match = {
        'Home': home_team,
        'Away': away_team
    }
    prediction = model.predict_one(match)
    return prediction

def over_25(prediction):
    total_probability = 0

    for scoreline in prediction:
        if (scoreline.home_goals + scoreline.away_goals) >= 2.5:
            total_probability += scoreline.probability
            

    return total_probability
    


data = get_esa_games()
model = create_model(data)
prediction = predict_game(model, 'Lech Poznan', 'Legia Warszawa')
st.write(over_25(prediction))






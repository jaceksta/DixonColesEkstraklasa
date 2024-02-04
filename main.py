import pandas as pd
import plotly.express as px
import json
import mezzala
import streamlit as st
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects



def get_esa_games():
    url = "https://www.football-data.co.uk/new/POL.csv"
    df = pd.read_csv(url)
    df_2324 = df[df['Season'] == '2023/2024'][['Home', 'Away', 'HG', 'AG', 'Date']]
    return df_2324

def turn_into_dict(df):
    df_2324_json = df.to_json(orient='records')
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
            

    total_probability = np.round(total_probability * 100, decimals=2)
    under_25 = 100 - total_probability

    labels = ['Over 2.5', 'Under 2.5']
    probs = [total_probability, under_25]
    df = pd.DataFrame({'result': labels, 'probs': probs}).reset_index()
    fig = px.bar(df, x='result', y='probs', text='probs', labels={'result': 'Over / Under 2.5', 'probs': 'Probability'}, color='result')
    fig.update_layout(showlegend=False)
    return fig

def create_array(prediction):
    filtered_scorelines = [scoreline for scoreline in prediction if scoreline.home_goals <= 5 and scoreline.away_goals <= 5]
    array_5x5 = np.zeros((6, 6))

    for scoreline in filtered_scorelines:
        home_goals = scoreline.home_goals
        away_goals = scoreline.away_goals
        probability = scoreline.probability
        array_5x5[home_goals, away_goals] = probability

    array_5x5 = np.round(array_5x5 * 100, decimals=2)
    
    
    return array_5x5
    

def create_heatmap(prediction, home_team, away_team):
    array_5x5 = create_array(prediction)
    df_5x5 = pd.DataFrame(array_5x5, columns=range(6), index=range(6))
    
    fig = px.imshow(df_5x5, labels=dict(x=away_team,
                                    y=home_team, color="Probability"),
                                    text_auto=True, color_continuous_scale='Blues')
    fig.update_traces(text=df_5x5.applymap(lambda x: f'{x}%'), texttemplate="%{text}")
    fig.update_layout(
        title={
            'text': f'{home_team} vs {away_team}',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title=away_team,
        yaxis_title=home_team,
        font=dict(
            family="sans-serif",
            size=18,
            color="#7f7f7f"
        ),
        showlegend=False  # Remove the legend
    )
    fig.update_traces(hovertemplate="<b>%{y}-%{x}</b> <br>Probability: %{text}")

    fig.update_coloraxes(showscale=False)
    
    return fig


def plot_probabilities(prediction, home_team, away_team):
    outcomes = mezzala.scorelines_to_outcomes(prediction)
    probabilities = [outcome.probability for outcome in outcomes.values()]
    home = np.round(probabilities[0]*100, decimals=2)
    draw = np.round(probabilities[1]*100, decimals=2)
    away = np.round(probabilities[2]*100, decimals=2)
    labels = [home_team, 'Draw', away_team]
    probs = [home, draw, away]
    df = pd.DataFrame({'result': labels, 'probs': probs}).reset_index()
    
    fig = px.bar(df, x='result', y='probs', text='probs', labels={'result': 'Result', 'probs': 'Probability'}, color='result')
    fig.update_layout(showlegend=False)
    return fig

def plot_expected_points(prediction, home_team, away_team):
    outcomes = mezzala.scorelines_to_outcomes(prediction)
    probabilities = [outcome.probability for outcome in outcomes.values()]
    home = np.round(probabilities[0], decimals=2)
    draw = np.round(probabilities[1], decimals=2)
    away = np.round(probabilities[2], decimals=2)
    home_pts = np.round(home*3 + draw, decimals=2)
    away_pts = np.round(away*3 + draw, decimals=2)
    labels = [home_team, away_team]
    probs = [home_pts, away_pts]
    df = pd.DataFrame({'result': labels, 'probs': probs}).reset_index()
    
    fig = px.bar(df, x='result', y='probs', text='probs', labels={'result': 'Team', 'probs': 'Expected Points'})
    return fig

def expected_goals(prediction, home_team, away_team):
    home = 0
    away = 0
    for scoreline in prediction:
        home += scoreline.home_goals * scoreline.probability
        away += scoreline.away_goals * scoreline.probability
    home = np.round(home, decimals = 2)
    away = np.round(away, decimals=2)
    labels = [home_team, away_team]
    df = pd.DataFrame({'Team': labels, 'Expected Goals': [home, away]})
    fig = px.bar(df, x='Team', y='Expected Goals', text='Expected Goals', labels={'Team': 'Team', 'Expected Goals': 'Expected Goals'})
    return fig    

def clean_sheet_probability(prediction, home_team, away_team):
    home = 0
    away = 0
    for scoreline in prediction:
        if scoreline.away_goals == 0:
            home += scoreline.probability
        if scoreline.home_goals == 0:
            away += scoreline.probability
    home = np.round(home*100, decimals = 2)
    away = np.round(away*100, decimals=2)
    labels = [home_team, away_team]
    df = pd.DataFrame({'Team': labels, 'Clean Sheet Probability': [home, away]})
    fig = px.bar(df, x='Team', y='Clean Sheet Probability', text='Clean Sheet Probability', labels={'Team': 'Team', 'Clean Sheet Probability': 'Clean Sheet Probability'})
    return fig

def both_teams_to_score(prediction):
    btts = 0
    for scoreline in prediction:
        if scoreline.away_goals > 0 and scoreline.home_goals > 0:
            btts += scoreline.probability
    
    btts = np.round(btts*100, decimals=2)
    labels = ['Yes', 'No']
    probs = [btts, 100-btts]
    df = pd.DataFrame({'result': labels, 'probs': probs}).reset_index()
    fig = px.bar(df, x='result', y='probs', text='probs', labels={'result': 'Both Teams to Score', 'probs': 'Probability'}, color='result')
    fig.update_layout(showlegend=False)
    return fig



df = get_esa_games()
data = turn_into_dict(df)
model = create_model(data)
st.title('ESA Predictor')
teams = df['Home'].unique()

home_team = st.selectbox('Home Team', teams, index = 1)
away_team = st.selectbox('Away Team', teams)

prediction = predict_game(model, home_team, away_team)
st.header('Expected Goals')
st.plotly_chart(expected_goals(prediction, home_team, away_team))
plot = create_heatmap(prediction,  home_team, away_team)
st.header('Scoreline Probability')
st.plotly_chart(plot)
st.header('Clean Sheet Probability')
st.plotly_chart(clean_sheet_probability(prediction, home_team, away_team))
st.header('Over / Under 2.5')
st.plotly_chart(over_25(prediction))
st.header('Both Teams to Score')
st.plotly_chart(both_teams_to_score(prediction))
st.header('Result Probability')
st.plotly_chart(plot_probabilities(prediction, home_team, away_team))
st.header('Expected Points')
st.plotly_chart(plot_expected_points(prediction, home_team, away_team))






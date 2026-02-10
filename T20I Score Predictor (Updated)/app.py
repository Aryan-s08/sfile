import streamlit as st
import pandas as pd
import joblib

def run():
    # ================= LOAD MODEL =================
    @st.cache_resource
    def load_model():
        import os

        BASE_DIR = os.path.dirname(__file__)

        model = joblib.load(os.path.join(BASE_DIR, "5tmodel.joblib"))
        features = joblib.load(os.path.join(BASE_DIR, "5tfeatures.joblib"))
        return model, features

    model, model_features = load_model()

    # ================= OPTIONS =================
    teams = ['Afghanistan', 'Australia', 'Bangladesh', 'Canada', 'England', 'India', 'Ireland', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Oman', 'Pakistan', 'Scotland', 'South Africa', 'Sri Lanka', 'United Arab Emirates', 'United States of America', 'West Indies', 'Zimbabwe']

    venues = ['AMI Stadium', 'Adelaide Oval', 'Al Amerat Cricket Ground Oman Cricket (Ministry Turf 1)', 'Al Amerat Cricket Ground Oman Cricket (Ministry Turf 2)', 'Arnos Vale Ground', 'Arun Jaitley Stadium', 'Barabati Stadium', 'Barsapara Cricket Stadium', 'Bay Oval', 'Bellerive Oval', 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium', 'Boland Park', 'Brabourne Stadium', 'Bready Cricket Club', 'Brian Lara Stadium', 'Brisbane Cricket Ground', 'Buffalo Park', 'Bulawayo Athletic Club', 'Carrara Oval', 'Castle Avenue', 'Central Broward Regional Park Stadium Turf Ground', 'Civil Service Cricket Club', 'Clontarf Cricket Club Ground', 'Coolidge Cricket Ground', 'County Ground', 'Darren Sammy National Cricket Stadium', 'De Beers Diamond Oval', 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium', 'Dubai International Cricket Stadium', 'Eden Gardens', 'Eden Park', 'Edgbaston', 'Feroz Shah Kotla', 'GMHBA Stadium', 'Gaddafi Stadium', 'Grange Cricket Club Ground', 'Greater Noida Sports Complex Ground', 'Green Park', 'Greenfield International Stadium', 'Gymkhana Club Ground', 'Hagley Oval', 'Harare Sports Club', 'Hazelaarweg', 'Headingley', 'Himachal Pradesh Cricket Association Stadium', 'Holkar Cricket Stadium', 'ICC Global Cricket Academy', 'Indian Association Ground', 'JSCA International Stadium Complex', 'Jade Stadium', 'John Davies Oval', 'Kennington Oval', 'Kensington Oval', 'Khan Shaheb Osman Ali Stadium', 'Kingsmead', "Lord's", 'M.Chinnaswamy Stadium', 'MA Chidambaram Stadium', 'Maharashtra Cricket Association Stadium', 'Mahinda Rajapaksa International Cricket Stadium', 'Malahide', 'Mangaung Oval', 'Manuka Oval', 'Maple Leaf North-West Ground', 'McLean Park', 'Melbourne Cricket Ground', 'Moses Mabhida Stadium', 'Narendra Modi Stadium', 'National Cricket Stadium', 'National Stadium', 'New Wanderers Stadium', 'Newlands', 'OUTsurance Oval', 'Old Trafford', 'Pallekele International Cricket Stadium', 'Perth Stadium', 'Providence Stadium', 'Punjab Cricket Association IS Bindra Stadium', "Queen's Park Oval", 'Queens Sports Club', 'R.Premadasa Stadium', 'Rajiv Gandhi International Cricket Stadium', 'Rawalpindi Cricket Stadium', 'Riverside Ground', 'Sabina Park', 'Saurashtra Cricket Association Stadium', 'Sawai Mansingh Stadium', 'Saxton Oval', 'Seddon Park', 'Senwes Park', 'Sharjah Cricket Stadium', 'Sheikh Abu Naser Stadium', 'Sheikh Zayed Stadium', 'Shere Bangla National Stadium', 'Sir Vivian Richards Stadium', 'Sky Stadium', 'Sophia Gardens', 'Sportpark Het Schootsveld', 'Sportpark Westvliet', "St George's Park", 'Stadium Australia', 'Subrata Roy Sahara Stadium', 'SuperSport Park', 'Sydney Cricket Ground', 'Sylhet International Cricket Stadium', 'The Rose Bowl', 'The Village', 'Tolerance Oval', 'Trent Bridge', 'Tribhuvan University International Cricket Ground', 'University Oval', 'VRA Ground', 'Vidarbha Cricket Association Stadium', 'WACA Ground', 'Wankhede Stadium', 'Warner Park', 'Westpac Stadium', 'White Hill Field', 'Windsor Park', 'Zahur Ahmed Chowdhury Stadium']


    # ================= UI =================
    st.title("📈 T20 Score Predictor")

    st.sidebar.header("Match Inputs")

    over = st.sidebar.number_input("Over (0–19)", min_value=0, max_value=19)
    ball = st.sidebar.number_input("Ball (1–6)", min_value=1, max_value=6)
    innings = st.sidebar.selectbox("Innings", [1])
    runs_from_ball = st.sidebar.number_input("Runs from Ball", min_value=0, max_value=6)

    current_score = st.sidebar.number_input("Current Score", min_value=0)

    batter_runs = st.sidebar.number_input("Batter Runs", min_value=0)
    batter_balls = st.sidebar.number_input("Batter Balls", min_value=0)

    non_striker_runs = st.sidebar.number_input("Non-Striker Runs", min_value=0)
    non_striker_balls = st.sidebar.number_input("Non-Striker Balls", min_value=0)

    innings_wickets = st.sidebar.number_input("Innings Wickets", min_value=0, max_value=10)

    last_five_runs = st.sidebar.number_input("Last 5 Overs Runs", min_value=0)
    last_five_wickets = st.sidebar.number_input("Last 5 Overs Wickets", min_value=0)

    wicket = st.sidebar.selectbox("Wicket", ["No", "Yes"])

    batting_team = st.sidebar.selectbox("Batting Team", teams)
    bowling_team = st.sidebar.selectbox("Bowling Team", teams)
    venue = st.sidebar.selectbox("Venue", venues)

    # ================= PREDICT =================
    if st.button("Predict"):

        if batting_team == bowling_team:
            st.error("Batting and Bowling teams must be different.")
        else:
            balls_remaining = 120 - (over * 6 + ball)

            input_dict = {
                "Over": over,
                "Ball": ball,
                "Innings": innings,
                "Runs From Ball": runs_from_ball,
                "Innings Runs": current_score,
                "Total Batter Runs": batter_runs,
                "Batter Balls Faced": batter_balls,
                "Total Non Striker Runs": non_striker_runs,
                "Non Striker Balls Faced": non_striker_balls,
                "Innings Wickets": innings_wickets,
                "last_five_runs": last_five_runs,
                "last_five_wickets": last_five_wickets,
                "Balls Remaining": balls_remaining,
                "Wicket": 1 if wicket == "Yes" else 0
            }

            # One-hot
            input_dict[venue] = 1
            input_dict[f"Bat_First_{batting_team}"] = 1
            input_dict[f"Bat_Second_{bowling_team}"] = 1

            input_df = pd.DataFrame([input_dict])
            input_df = input_df.reindex(columns=model_features, fill_value=0)

            # Debug view
            with st.expander("🔍 Input sent to model"):
                st.dataframe(input_df.T[input_df.iloc[0] != 0])

            # Prediction
            pred = int(model.predict(input_df)[0])

            st.success(f"🏏 Predicted Final Score: **{pred}**")

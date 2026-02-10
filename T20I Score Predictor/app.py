import streamlit as st
import pandas as pd
import joblib
import os
def run():
    # ================= PAGE CONFIG =================
    st.set_page_config(
        page_title="🏏 Cricket Score Predictor",
        layout="wide"
    )

    # ================= LOAD MODEL =================
    BASE_DIR = os.path.dirname(__file__)

    @st.cache_resource
    def load_assets():
        model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
        features = joblib.load(os.path.join(BASE_DIR, "model_features.joblib"))
        return model, features

    model, model_features = load_assets()

    # ================= CONSTANT LISTS =================
    teams = [
        "Afghanistan","Australia","Bangladesh","Canada","England","India",
        "Ireland","Italy","Namibia","Nepal","Netherlands","New Zealand",
        "Oman","Pakistan","Papua New Guinea","Scotland","South Africa",
        "Sri Lanka","Uganda","United Arab Emirates",
        "United States of America","West Indies","Zimbabwe"
    ]

    venues = [
        'AMI Stadium','Adelaide Oval',
        'Al Amerat Cricket Ground Oman Cricket (Ministry Turf 1)',
        'Al Amerat Cricket Ground Oman Cricket (Ministry Turf 2)',
        'Arnos Vale Ground','Arun Jaitley Stadium','Barabati Stadium',
        'Barsapara Cricket Stadium','Bay Oval','Bellerive Oval',
        'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium',
        'Boland Park','Brabourne Stadium','Bready Cricket Club',
        'Brian Lara Stadium','Brisbane Cricket Ground','Buffalo Park',
        'Bulawayo Athletic Club','Carrara Oval','Castle Avenue',
        'Central Broward Regional Park Stadium Turf Ground',
        'Civil Service Cricket Club','Clontarf Cricket Club Ground',
        'Coolidge Cricket Ground','County Ground',
        'Darren Sammy National Cricket Stadium','De Beers Diamond Oval',
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
        'Dubai International Cricket Stadium','Eden Gardens','Eden Park',
        'Edgbaston','Feroz Shah Kotla','GMHBA Stadium','Gaddafi Stadium',
        'Goldenacre','Grange Cricket Club Ground',
        'Greater Noida Sports Complex Ground','Green Park',
        'Greenfield International Stadium','Gymkhana Club Ground',
        'Hagley Oval','Harare Sports Club','Hazelaarweg','Headingley',
        'Himachal Pradesh Cricket Association Stadium',
        'Holkar Cricket Stadium','ICC Global Cricket Academy',
        'Indian Association Ground','JSCA International Stadium Complex',
        'Jade Stadium','John Davies Oval','Kennington Oval',
        'Kensington Oval','Khan Shaheb Osman Ali Stadium','Kingsmead',
        "Lord's",'M.Chinnaswamy Stadium','MA Chidambaram Stadium',
        'Maharashtra Cricket Association Stadium',
        'Mahinda Rajapaksa International Cricket Stadium',
        'Malahide','Mangaung Oval','Manuka Oval',
        'Maple Leaf North-West Ground','McLean Park',
        'Melbourne Cricket Ground','Moses Mabhida Stadium',
        'Narendra Modi Stadium','National Cricket Stadium',
        'National Stadium','New Wanderers Stadium','Newlands',
        'OUTsurance Oval','Old Trafford',
        'Pallekele International Cricket Stadium','Perth Stadium',
        'Providence Stadium','Punjab Cricket Association IS Bindra Stadium',
        "Queen's Park Oval",'Queens Sports Club','R.Premadasa Stadium',
        'Rajiv Gandhi International Cricket Stadium',
        'Rawalpindi Cricket Stadium','Riverside Ground','Sabina Park',
        'Saurashtra Cricket Association Stadium','Sawai Mansingh Stadium',
        'Saxton Oval','Seddon Park','Senwes Park','Sharjah Cricket Stadium',
        'Sheikh Abu Naser Stadium','Sheikh Zayed Stadium',
        'Shere Bangla National Stadium','Sir Vivian Richards Stadium',
        'Sky Stadium','Sophia Gardens','Sportpark Het Schootsveld',
        'Sportpark Westvliet',"St George's Park",'Stadium Australia',
        'Subrata Roy Sahara Stadium','SuperSport Park',
        'Sydney Cricket Ground','Sylhet International Cricket Stadium',
        'The Rose Bowl','The Village','Tolerance Oval',
        'Tony Ireland Stadium','Trent Bridge',
        'Tribhuvan University International Cricket Ground',
        'United Cricket Club Ground','University Oval','VRA Ground',
        'Vidarbha Cricket Association Stadium','Wankhede Stadium',
        'Warner Park','Western Australia Cricket Association Ground',
        'Westpac Stadium','White Hill Field','Windsor Park',
        'Zahur Ahmed Chowdhury Stadium'
    ]

    # ================= TITLE =================
    st.markdown(
        "<h1 style='text-align:center'>🏏 Cricket Score Predictor</h1>",
        unsafe_allow_html=True
    )

    # ================= LAYOUT =================
    left, right = st.columns([3, 1.4])

    # ================= LEFT PANEL =================
    with left:

        st.subheader("Match State")

        innings = st.radio("Innings", [1, 2], horizontal=True)
        wicket = st.radio("Wicket on last ball?", ["No", "Yes"], horizontal=True)

        over = st.slider("Over (0–19)", 0, 19, 1)
        ball = st.slider("Ball (1–6)", 1, 6, 1)
        runs_from_ball = st.slider("Runs From Ball", 0, 6, 0)
        innings_wickets = st.slider("Innings Wickets", 0, 9, 0)

        st.subheader("Batters")

        innings_runs = st.number_input("Innings Runs", min_value=0)
        striker_runs = st.number_input("Striker Runs", min_value=0)
        striker_balls = st.number_input("Striker Balls Faced", min_value=0)
        non_striker_runs = st.number_input("Non-Striker Runs", min_value=0)
        non_striker_balls = st.number_input("Non-Striker Balls Faced", min_value=0)

        st.subheader("Teams & Venue")

        batting_team = st.selectbox("Batting Team", teams)
        bowling_team = st.selectbox("Bowling Team", teams)
        venue = st.selectbox("Venue", venues)

    # ================= RIGHT PANEL =================
    with right:
        st.subheader("Prediction")

        if st.button("Predict Final Score", use_container_width=True):

            balls_bowled = (over * 6) + ball
            balls_remaining = max(0, 120 - balls_bowled)

            input_dict = {
                "Innings": innings,
                "Over": over + 1,
                "Ball": ball,
                "Runs From Ball": runs_from_ball,
                "Batter Runs": runs_from_ball,
                "Wicket": 1 if wicket == "Yes" else 0,
                "Innings Runs": innings_runs,
                "Innings Wickets": innings_wickets,
                "Balls Remaining": balls_remaining,
                "Total Batter Runs": striker_runs,
                "Batter Balls Faced": striker_balls,
                "Total Non Striker Runs": non_striker_runs,
                "Non Striker Balls Faced": non_striker_balls
            }

            input_dict[f"BatFirst_{batting_team}"] = 1
            input_dict[f"BatSecond_{bowling_team}"] = 1
            input_dict[venue] = 1

            input_df = pd.DataFrame([input_dict])
            input_df = input_df.reindex(columns=model_features, fill_value=0)

            prediction = int(model.predict(input_df)[0])

            st.success(f"🏏 Predicted Final Score: **{prediction}**")

            with st.expander("🔍 Input to Model"):
                st.dataframe(input_df)

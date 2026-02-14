import streamlit as st
import pandas as pd
import joblib
def run():
    # ================= PAGE CONFIG =================
    st.set_page_config(
        page_title="🏏 T20 Win Probability Predictor",
        layout="centered"
    )

    # ================= LOAD MODEL =================
    @st.cache_resource
    def load_model():
        import os

        BASE_DIR = os.path.dirname(__file__)

        model = joblib.load(os.path.join(BASE_DIR, "nnpredmodel.joblib"))
        features = joblib.load(os.path.join(BASE_DIR, "nnpredmodelfeatures.joblib"))
        return model, features

    model, model_features = load_model()

    # ================= OPTIONS =================
    teams = ['Afghanistan', 'Australia', 'Bangladesh', 'Canada', 'England', 'India', 'Ireland', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Oman', 'Pakistan', 'Scotland', 'South Africa', 'Sri Lanka', 'United Arab Emirates', 'United States of America', 'West Indies', 'Zimbabwe']

    venues = ['AMI Stadium', 'Adelaide Oval', 'Al Amerat Cricket Ground Oman Cricket (Ministry Turf 1)', 'Al Amerat Cricket Ground Oman Cricket (Ministry Turf 2)', 'Arnos Vale Ground', 'Arun Jaitley Stadium', 'Barabati Stadium', 'Barsapara Cricket Stadium', 'Bay Oval', 'Bellerive Oval', 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium', 'Boland Park', 'Brabourne Stadium', 'Bready Cricket Club', 'Brian Lara Stadium', 'Brisbane Cricket Ground', 'Buffalo Park', 'Bulawayo Athletic Club', 'Carrara Oval', 'Castle Avenue', 'Central Broward Regional Park Stadium Turf Ground', 'Civil Service Cricket Club', 'Clontarf Cricket Club Ground', 'Coolidge Cricket Ground', 'County Ground', 'Darren Sammy National Cricket Stadium', 'De Beers Diamond Oval', 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium', 'Dubai International Cricket Stadium', 'Eden Gardens', 'Eden Park', 'Edgbaston', 'Feroz Shah Kotla', 'GMHBA Stadium', 'Gaddafi Stadium', 'Grange Cricket Club Ground', 'Greater Noida Sports Complex Ground', 'Green Park', 'Greenfield International Stadium', 'Gymkhana Club Ground', 'Hagley Oval', 'Harare Sports Club', 'Hazelaarweg', 'Headingley', 'Himachal Pradesh Cricket Association Stadium', 'Holkar Cricket Stadium', 'ICC Global Cricket Academy', 'Indian Association Ground', 'JSCA International Stadium Complex', 'Jade Stadium', 'John Davies Oval', 'Kennington Oval', 'Kensington Oval', 'Khan Shaheb Osman Ali Stadium', 'Kingsmead', "Lord's", 'M.Chinnaswamy Stadium', 'MA Chidambaram Stadium', 'Maharashtra Cricket Association Stadium', 'Mahinda Rajapaksa International Cricket Stadium', 'Malahide', 'Mangaung Oval', 'Manuka Oval', 'Maple Leaf North-West Ground', 'McLean Park', 'Melbourne Cricket Ground', 'Moses Mabhida Stadium', 'Narendra Modi Stadium', 'National Cricket Stadium', 'National Stadium', 'New Wanderers Stadium', 'Newlands', 'OUTsurance Oval', 'Old Trafford', 'Pallekele International Cricket Stadium', 'Perth Stadium', 'Providence Stadium', 'Punjab Cricket Association IS Bindra Stadium', "Queen's Park Oval", 'Queens Sports Club', 'R.Premadasa Stadium', 'Rajiv Gandhi International Cricket Stadium', 'Rawalpindi Cricket Stadium', 'Riverside Ground', 'Sabina Park', 'Saurashtra Cricket Association Stadium', 'Sawai Mansingh Stadium', 'Saxton Oval', 'Seddon Park', 'Senwes Park', 'Sharjah Cricket Stadium', 'Sheikh Abu Naser Stadium', 'Sheikh Zayed Stadium', 'Shere Bangla National Stadium', 'Sir Vivian Richards Stadium', 'Sky Stadium', 'Sophia Gardens', 'Sportpark Het Schootsveld', 'Sportpark Westvliet', "St George's Park", 'Stadium Australia', 'Subrata Roy Sahara Stadium', 'SuperSport Park', 'Sydney Cricket Ground', 'Sylhet International Cricket Stadium', 'The Rose Bowl', 'The Village', 'Tolerance Oval', 'Trent Bridge', 'Tribhuvan University International Cricket Ground', 'University Oval', 'VRA Ground', 'Vidarbha Cricket Association Stadium', 'WACA Ground', 'Wankhede Stadium', 'Warner Park', 'Westpac Stadium', 'White Hill Field', 'Windsor Park', 'Zahur Ahmed Chowdhury Stadium']



    # ================= TITLE =================
    st.markdown(
        "<h1 style='text-align:center'>🏏 T20 Win Probability Predictor</h1>",
        unsafe_allow_html=True
    )

    st.divider()

    # ================= INPUT FORM =================
    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            over = st.number_input("Over", min_value=0, max_value=19, step=1)
            ball = st.number_input("Ball", min_value=1, max_value=6, step=1)
            runs_ball = st.number_input("Runs from Ball", min_value=0, max_value=6, step=1)
            innings_runs = st.number_input("Innings Runs", min_value=0)
            innings_wkts = st.number_input("Innings Wickets", min_value=0, max_value=9)
            target = st.number_input("Target Score", min_value=1)

        with col2:
            rlast5 = st.number_input("Last 5 over runs", min_value=0)
            wlast5 = st.number_input("Last 5 over wickets", min_value=0)
            batter_runs = st.number_input("Batter Runs", min_value=0)
            batter_balls = st.number_input("Batter Balls Faced", min_value=0)
            non_striker_runs = st.number_input("Non-Striker Runs", min_value=0)
            non_striker_balls = st.number_input("Non-Striker Balls Faced", min_value=0)

        st.divider()

        batting = st.selectbox("Batting Team", teams, index=0)
        bowling = st.selectbox("Bowling Team", teams, index=1)
        venue = st.selectbox("Venue", venues)

        submit = st.form_submit_button("Predict Win Probability")

    # ================= FEATURE BUILDER =================
    def build_input_df():
        if batting == bowling:
            st.error("Batting and bowling teams must be different")
            return None

        if innings_runs > target:
            st.error("Innings runs cannot exceed target")
            return None

        balls_used = over * 6 + ball
        balls_remaining = 120 - balls_used

        input_dict = {
            "Over": over,
            "Ball": ball,
            "Runs From Ball": runs_ball,
            "Balls Remaining": balls_remaining,
            "Innings Runs": innings_runs,
            "Innings Wickets": innings_wkts,
            "Target Score": target,
            "Total Batter Runs": batter_runs,
            "Total Non Striker Runs": non_striker_runs,
            "Batter Balls Faced": batter_balls,
            "Non Striker Balls Faced": non_striker_balls,
            "Runs to Get": target - innings_runs,
            "last_five_runs" : rlast5,
            "last_five_wickets" : wlast5
        }

        # One-hot encoding (MUST match training)
        input_dict[f"Bat_Second_{batting}"] = 1
        input_dict[f"Bat_First_{bowling}"] = 1
        input_dict[venue] = 1

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=model_features, fill_value=0)

        return input_df

    # ================= PREDICTION =================
    if submit:
        input_df = build_input_df()

        if input_df is not None:
            proba = model.predict_proba(input_df)[0]
            classes = model.classes_
            proba_map = dict(zip(classes, proba))

            batting_win = round(proba_map.get(1, 0) * 100, 2)
            bowling_win = round(proba_map.get(0, 0) * 100, 2)

            st.success("Prediction Successful")

            col1, col2 = st.columns(2)
            col1.metric(f"{batting} Win %", f"{batting_win}%")
            col2.metric(f"{bowling} Win %", f"{bowling_win}%")

            # Debug: show model input if needed
            with st.expander("🔍 Show model input"):
                st.dataframe(input_df)

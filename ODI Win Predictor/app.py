import streamlit as st
import pandas as pd
import joblib
def run():
    # ================= PAGE CONFIG =================
    st.set_page_config(
        page_title="🏏 ODI Win Predictor",
        layout="centered"
    )

    # ================= LOAD MODEL =================
    @st.cache_resource
    def load_model():
        import os

        BASE_DIR = os.path.dirname(__file__)

        model = joblib.load(os.path.join(BASE_DIR, "win_model.joblib"))
        model_features = joblib.load(os.path.join(BASE_DIR, "win_features.joblib"))
        return model, model_features

    model, model_features = load_model()

    # ================= OPTIONS =================
    teams = ['Afghanistan', 'Australia', 'Bangladesh', 'England', 'India', 'Ireland', 'Namibia', 'Nepal', 'New Zealand', 'Pakistan', 'South Africa', 'South Africa', 'Sri Lanka', 'West Indies', 'Zimbabwe']

    venues = ['AMI Stadium', 'Adelaide Oval', 'Andhra Cricket Association-Visakhapatnam District Cricket Association Stadium', "Antigua Recreation Ground, St John's", 'Arbab Niaz Stadium', 'Arnos Vale Ground, Kingstown, St Vincent', 'Arun Jaitley Stadium, Delhi', 'Bangabandhu National Stadium, Dhaka', 'Barabati Stadium, Cuttack', 'Barsapara Cricket Stadium, Guwahati', 'Basin Reserve', 'Bay Oval, Mount Maunganui', 'Beausejour Stadium, Gros Islet', 'Bellerive Oval, Hobart', 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow', 'Boland Bank Park, Paarl', 'Brabourne Stadium', 'Bready Cricket Club, Magheramason', 'Brian Lara Stadium, Tarouba, Trinidad', 'Brisbane Cricket Ground, Woolloongabba, Brisbane', 'Buffalo Park, East London', 'Bundaberg Rum Stadium, Cairns', 'Cambusdoon New Ground, Ayr', 'Captain Roop Singh Stadium, Gwalior', 'Carisbrook', 'Castle Avenue', "Cazaly's Stadium, Cairns", 'Chevrolet Park', 'Chittagong Divisional Stadium', 'City Oval, Pietermaritzburg', 'Civil Service Cricket Club, Stormont, Belfast', 'Clontarf Cricket Club Ground, Dublin', 'Cobham Oval (New)', 'County Ground, Chelmsford', 'Darren Sammy National Cricket Stadium, St Lucia', 'Davies Park, Queenstown', 'Diamond Oval, Kimberley', 'Docklands Stadium', 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam', 'Dubai International Cricket Stadium', 'Dubai Sports City Cricket Stadium', 'Eden Gardens, Kolkata', 'Eden Park, Auckland', 'Edgbaston, Birmingham', 'Feroz Shah Kotla', 'Gaddafi Stadium, Lahore', 'Galle International Stadium', 'Goodyear Park, Bloemfontein', 'Greater Noida Sports Complex Ground', 'Green Park', 'Greenfield International Stadium, Thiruvananthapuram', 'Gymkhana Club Ground', 'Hagley Oval, Christchurch', 'Harare Sports Club', 'Hazelaarweg, Rotterdam', 'Headingley', 'Himachal Pradesh Cricket Association Stadium', 'Holkar Cricket Stadium, Indore', 'Indian Petrochemicals Corporation Limited Sports Complex Ground', 'Iqbal Stadium, Faisalabad', 'JSCA International Stadium Complex, Ranchi', 'Jade Stadium', 'Jade Stadium, Christchurch', 'John Davies Oval', 'Keenan Stadium', 'Kennington Oval, London', 'Kensington Oval, Bridgetown, Barbados', 'Khan Shaheb Osman Ali Stadium', 'Kingsmead, Durban', 'Kinrara Academy Oval', 'Lal Bahadur Shastri Stadium, Hyderabad, Deccan', "Lord's, London", 'M Chinnaswamy Stadium', 'M.Chinnaswamy Stadium', 'MA Aziz Stadium, Chittagong', 'MA Chidambaram Stadium, Chepauk, Chennai', 'Madhavrao Scindia Cricket Ground', 'Maharani Usharaje Trust Cricket Ground', 'Maharashtra Cricket Association Stadium', 'Mahinda Rajapaksa International Cricket Stadium, Sooriyawewa, Hambantota', 'Malahide', 'Mangaung Oval, Bloemfontein', 'Manuka Oval', 'Marrara Cricket Ground, Darwin', 'McLean Park, Napier', 'Melbourne Cricket Ground', 'Multan Cricket Stadium', 'Nahar Singh Stadium', 'Nahar Singh Stadium, Faridabad', 'Narayanganj Osmani Stadium', 'Narendra Modi Stadium, Ahmedabad', 'National Cricket Stadium', 'National Cricket Stadium, Grenada', "National Cricket Stadium, St George's", 'National Stadium', 'National Stadium, Karachi', 'Nehru Stadium', 'Nehru Stadium, Fatorda', 'Nehru Stadium, Poona', 'New Wanderers Stadium, Johannesburg', 'Newlands, Cape Town', 'Niaz Stadium, Hyderabad', 'North West Cricket Stadium, Potchefstroom', 'OUTsurance Oval', 'Old Trafford', 'Old Trafford, Manchester', 'P Saravanamuttu Stadium', 'Pallekele International Cricket Stadium', 'Perth Stadium', 'Providence Stadium', 'Providence Stadium, Guyana', 'Punjab Cricket Association IS Bindra Stadium, Mohali', 'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh', 'Punjab Cricket Association Stadium, Mohali', "Queen's Park (New), St George's, Grenada", "Queen's Park Oval, Port of Spain", "Queen's Park Oval, Port of Spain, Trinidad", "Queen's Park Oval, Trinidad", 'Queens Sports Club, Bulawayo', 'Queenstown Events Centre', 'R.Premadasa Stadium, Khettarama', 'Rajiv Gandhi International Cricket Stadium, Dehradun', 'Rajiv Gandhi International Stadium, Uppal, Hyderabad', 'Rangiri Dambulla International Stadium', 'Rawalpindi Cricket Stadium', 'Reliance Stadium', 'Riverside Ground, Chester-le-Street', 'Riverway Stadium, Townsville', 'Sabina Park, Kingston, Jamaica', 'Sardar Patel (Gujarat) Stadium, Motera', 'Sardar Patel Stadium, Motera', 'Saurashtra Cricket Association Stadium', 'Sawai Mansingh Stadium', 'Saxton Oval', 'Sector 16 Stadium', 'Seddon Park, Hamilton', 'Senwes Park, Potchefstroom', 'Shaheed Chandu Stadium', 'Shaheed Veer Narayan Singh International Stadium, Raipur', 'Sharjah Cricket Stadium', 'Sheikh Abu Naser Stadium', 'Sheikh Zayed Stadium', 'Sheikhupura Stadium', 'Shere Bangla National Stadium, Mirpur', 'Sinhalese Sports Club', 'Sinhalese Sports Club Ground', 'Sir Vivian Richards Stadium', 'Sir Vivian Richards Stadium, North Sound', 'Sophia Gardens, Cardiff', 'Sportpark Het Schootsveld', 'Sportpark Maarschalkerweerd, Utrecht', "St George's Park", "St George's Park, Port Elizabeth", 'St Lawrence Ground', 'St Lawrence Ground, Canterbury', 'SuperSport Park, Centurion', 'Sydney Cricket Ground', 'Sylhet International Cricket Stadium', 'Takashinga Sports Club, Highfield, Harare', 'The Cooper Associates County Ground', 'The Rose Bowl', 'The Rose Bowl, Southampton', 'The Royal & Sun Alliance County Ground, Bristol', 'The Village, Malahide', 'The Village, Malahide, Dublin', 'The Wanderers Stadium', 'The Wanderers Stadium, Johannesburg', 'Tony Ireland Stadium, Townsville', 'Trent Bridge, Nottingham', 'Tribhuvan University International Cricket Ground, Kirtipur', 'United Cricket Club Ground, Windhoek', 'University Oval', 'VRA Ground, Amstelveen', 'Vidarbha C.A. Ground', 'Vidarbha Cricket Association Ground', 'Vidarbha Cricket Association Stadium, Jamtha', 'W.A.C.A. Ground', 'Wankhede Stadium, Mumbai', 'Warner Park, Basseterre', 'West End Park International Cricket Stadium, Doha', 'Western Australia Cricket Association Ground', 'Westpac Park, Hamilton', 'Westpac Stadium', 'Westpac Stadium, Wellington', 'Willowmoore Park, Benoni', 'Windsor Park, Roseau', 'Zohur Ahmed Chowdhury Stadium']


    # ================= UI =================
    st.title("🏏 ODI Win Probability Predictor")

    st.markdown("### Match Situation")

    col1, col2 = st.columns(2)

    with col1:
        target = st.number_input("Target Score", min_value=1, step=1)
        score = st.number_input("Current Score", min_value=0, step=1)
        wickets_fallen = st.number_input("Wickets Fallen", min_value=0, max_value=10, step=1)
        over = st.number_input("Over (e.g. 12.3)", min_value=0.0, max_value=50.0, step=0.1)

    with col2:
        batting = st.selectbox("Batting Team", teams)
        bowling = st.selectbox("Bowling Team", teams, index=1)
        venue = st.selectbox("Venue", venues)

    # ================= FEATURE BUILDER =================
    def build_input():
        if batting == bowling:
            st.error("Batting and Bowling teams must be different")
            return None

        if score > target:
            st.error("Current score cannot exceed target")
            return None

        balls_done = int(over // 1) * 6 + int(round((over % 1) * 10))
        balls_done = max(1, balls_done)

        balls_total = 300
        balls_left = balls_total - balls_done

        wickets_left = 10 - wickets_fallen
        runs_left = target - score

        crr = (score * 6) / balls_done
        rrr = (runs_left * 6) / max(1, balls_left)

        input_dict = {
            "target": target,
            "runs left": runs_left,
            "cum_innings_runs": score,
            "wickets_left": wickets_left,
            "ball": over,
            "crr": crr,
            "rrr": rrr
        }

        # One-hot encoding
        input_dict[f"batting_team_{batting}"] = 1
        input_dict[f"bowling_team_{bowling}"] = 1
        input_dict[venue] = 1

        df = pd.DataFrame([input_dict])
        df = df.reindex(columns=model_features, fill_value=0)

        return df

    # ================= PREDICTION =================
    st.markdown("---")

    if st.button("Predict Win Probability"):
        input_df = build_input()

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

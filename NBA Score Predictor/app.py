import streamlit as st
import pandas as pd
import joblib
def run():
    # ================= PAGE CONFIG =================
    st.set_page_config(
        page_title="🏀 NBA Score Predictor",
        layout="centered"
    )

    # ================= LOAD MODELS & DATA =================
    @st.cache_resource
    def load_models():
        import os

        BASE_DIR = os.path.dirname(__file__)

        home_model = joblib.load(os.path.join(BASE_DIR, "home_model.joblib"))
        away_model = joblib.load(os.path.join(BASE_DIR, "away_model.joblib"))
        model_features = joblib.load(os.path.join(BASE_DIR, "features.joblib"))
        team_stats = joblib.load(os.path.join(BASE_DIR, "team_stats.joblib"))
        return home_model, away_model, model_features, team_stats

    home_model, away_model, model_features, team_stats = load_models()

    teams = sorted(team_stats["team"].unique())

    # ================= FEATURE BUILDER =================
    def build_input_row(home_team, away_team):
        home_row = team_stats[team_stats["team"] == home_team]
        away_row = team_stats[team_stats["team"] == away_team]

        if home_row.empty or away_row.empty:
            return None

        input_dict = {}

        # Home features
        for col in home_row.columns:
            if col != "team":
                input_dict[f"home_{col}"] = home_row.iloc[0][col]

        # Away features
        for col in away_row.columns:
            if col != "team":
                input_dict[f"away_{col}"] = away_row.iloc[0][col]

        input_df = pd.DataFrame([input_dict])

        # 🔥 Align exactly with training features
        input_df = input_df.reindex(columns=model_features, fill_value=0)

        return input_df


    # ================= UI =================
    st.markdown(
        "<h1 style='text-align:center;'>🏀 NBA Score Predictor</h1>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        home_team = st.selectbox("Home Team", teams)

    with col2:
        away_team = st.selectbox("Away Team", teams, index=1)

    st.markdown("---")

    # ================= PREDICTION =================
    if st.button("Predict Final Score", use_container_width=True):

        if home_team == away_team:
            st.error("Home and Away teams must be different")
        else:
            input_df = build_input_row(home_team, away_team)

            if input_df is None:
                st.error("Team data not found")
            else:
                home_pred = round(home_model.predict(input_df)[0])
                away_pred = round(away_model.predict(input_df)[0])

                # No negative scores
                home_pred = max(0, int(home_pred))
                away_pred = max(0, int(away_pred))

                st.success(
                    f"**{home_team} {home_pred}  –  {away_pred} {away_team}**"
                )

                # Optional debug view
                with st.expander("🔍 Model Input (Debug)"):
                    st.dataframe(input_df.T)

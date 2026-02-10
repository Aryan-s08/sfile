import streamlit as st
import pandas as pd
import joblib
def run():
    # ---------------- LOAD MODELS ----------------
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
    formations = ['4-2-3-1', '4-3-3', '4-4-2', '3-4-3', '3-4-1-2', '3-1-4-2',
       '3-5-2', '5-3-2', '4-4-1-1', '4-5-1', '4-1-4-1', '5-4-1',
       '4-2-2-2', '4-3-1-2', '4-1-2-1-2◆', '3-5-1-1', '4-3-2-1',
       '4-1-3-2', '3-4-3◆', '3-2-4-1', '4-2-4-0', '3-3-3-1']

    # ---------------- PAGE CONFIG ----------------
    st.set_page_config(
        page_title="⚽ Football Score Predictor",
        layout="centered"
    )

    st.title("⚽ Football Score Predictor")
    st.caption("Predict final score based on teams & formations")

    # ---------------- INPUT UI ----------------
    col1, col2 = st.columns(2)

    with col1:
        home_team = st.selectbox("🏠 Home Team", teams)
        home_formation = st.selectbox("📋 Home Formation", formations)

    with col2:
        away_team = st.selectbox("✈️ Away Team", teams)
        away_formation = st.selectbox("📋 Away Formation", formations)

    # ---------------- FEATURE BUILDER ----------------
    def build_input(home_team, away_team, home_formation, away_formation):
        home_row = team_stats[team_stats["team"] == home_team]
        away_row = team_stats[team_stats["team"] == away_team]

        if home_row.empty or away_row.empty:
            return None

        input_dict = {}

        # ---- NUMERIC FEATURES (exclude xg/xga for manual handling) ----
        ignore_cols = ["team", "xg", "xga"]

        for col in home_row.columns:
            if col not in ignore_cols:
                input_dict[f"home_{col}"] = home_row.iloc[0][col]

        for col in away_row.columns:
            if col not in ignore_cols:
                input_dict[f"away_{col}"] = away_row.iloc[0][col]

        # ---- CORRECT xG / xGA CROSS-MAPPING ----
        input_dict["home_xg"]  = home_row.iloc[0]["xg"]
        input_dict["home_xga"] = away_row.iloc[0]["xga"]

        # ---- FORMATION ONE-HOT (MUST MATCH TRAINING PREFIX) ----
        input_dict[f"home_formation_{home_formation}"] = 1
        input_dict[f"away_formation_{away_formation}"] = 1

        df = pd.DataFrame([input_dict])

        # 🔥 ALIGN EXACTLY WITH TRAINING FEATURES
        df = df.reindex(columns=model_features, fill_value=0)
        with st.expander("🔍 Input dictionary"):
            st.json(input_dict)

        with st.expander("🔍 Final model input"):
            st.dataframe(df)

        return df

    # ---------------- PREDICTION ----------------
    if st.button("🔮 Predict Score", use_container_width=True):

        if home_team == away_team:
            st.error("Home and Away teams must be different")
        else:
            X = build_input(home_team, away_team, home_formation, away_formation)

            if X is None:
                st.error("Team data not found")
            else:
                home_score = round(home_model.predict(X)[0])
                away_score = round(away_model.predict(X)[0])

                home_score = max(0, home_score)
                away_score = max(0, away_score)

                st.success("### 🏁 Predicted Final Score")
                st.markdown(
                    f"""
                    <h2 style="text-align:center">
                    {home_team} <b>{home_score}</b> – <b>{away_score}</b> {away_team}
                    </h2>
                    """,
                    unsafe_allow_html=True
                )

                # Debug (optional)
                with st.expander("🔍 Show model input"):
                    st.dataframe(X)

import streamlit as st
import pandas as pd
import joblib
import os
def run():
    # ---------------- PAGE CONFIG ----------------
    st.set_page_config(
        page_title="🎾 Tennis Match Predictor",
        layout="centered"
    )

    st.title("🎾 Tennis Match Predictor")
    st.caption("Predict win probability using player statistics & match context")

    # ---------------- LOAD ASSETS ----------------
    @st.cache_resource
    def load_assets():
        BASE_DIR = os.path.dirname(__file__)
        model = joblib.load(os.path.join(BASE_DIR, "tmodel.joblib"))
        features = joblib.load(os.path.join(BASE_DIR, "tcolumns.joblib"))
        player_stats = joblib.load(os.path.join(BASE_DIR, "tstats.joblib"))
        return model, features, player_stats

    model, model_features, player_stats = load_assets()

    players = sorted(player_stats["player"].unique())

    # ---------------- INPUT UI ----------------
    col1, col2 = st.columns(2)

    with col1:
        player_1 = st.selectbox("🎾 Player 1", players)
        best_of = st.selectbox("🏆 Best Of", [3, 5])

    with col2:
        player_2 = st.selectbox("🎾 Player 2", players)
        surface = st.radio("🟫 Surface", ["Hard", "Clay", "Grass"], horizontal=True)

    # ---------------- FEATURE BUILDER ----------------
    def build_input(p1, p2, best_of, surface):
        p1_row = player_stats[player_stats["player"] == p1]
        p2_row = player_stats[player_stats["player"] == p2]

        if p1_row.empty or p2_row.empty:
            return None

        p1 = p1_row.iloc[0]
        p2 = p2_row.iloc[0]

        input_dict = {
            # -------- Match context --------
            "best_of": best_of,
            "Clay": int(surface == "Clay"),
            "Grass": int(surface == "Grass"),
            "Hard": int(surface == "Hard"),

            # -------- Player (p1) --------
            "player_L": int(p1["hand"] == "L"),
            "player_R": int(p1["hand"] == "R"),
            "player_ht": p1["ht"],
            "player_age": p1["age"],
            "player_rank": p1["rank"],
            "player_rank_points": p1["rank_points"],
            "player_ace": p1["ace"],
            "player_df": p1["df"],
            "player_svpt": p1["svpt"],
            "player_1stIn": p1["first_in"],
            "player_1stWon": p1["first_won"],
            "player_2ndWon": p1["second_won"],
            "player_SvGms": p1["SvGms"],
            "player_bpSaved": p1["bp_saved"],
            "player_bpFaced": p1["bp_faced"],

            # -------- Opponent (p2) --------
            "opp_L": int(p2["hand"] == "L"),
            "opp_R": int(p2["hand"] == "R"),
            "opp_ht": p2["ht"],
            "opp_age": p2["age"],
            "opp_rank": p2["rank"],
            "opp_rank_points": p2["rank_points"],
            "opp_ace": p2["ace"],
            "opp_df": p2["df"],
            "opp_svpt": p2["svpt"],
            "opp_1stIn": p2["first_in"],
            "opp_1stWon": p2["first_won"],
            "opp_2ndWon": p2["second_won"],
            "opp_SvGms": p2["SvGms"],
            "opp_bpSaved": p2["bp_saved"],
            "opp_bpFaced": p2["bp_faced"],
        }

        X = pd.DataFrame([input_dict])

        # 🔥 CRITICAL: align EXACTLY with training
        X = X.reindex(columns=model_features, fill_value=0)

        return X


    # ---------------- PREDICTION ----------------
    if st.button("🔮 Predict Winner", use_container_width=True):

        if player_1 == player_2:
            st.error("Please select two different players")
        else:
            X = build_input(player_1, player_2, best_of, surface)

            if X is None:
                st.error("Player stats not found")
            else:
                prob = model.predict_proba(X)[0]

                p1_win = prob[1] * 100
                p2_win = prob[0] * 100

                st.success("### 🏆 Win Probability")
                st.markdown(
                    f"""
                    <h3 style="text-align:center">
                    {player_1}: <b>{p1_win:.2f}%</b><br>
                    {player_2}: <b>{p2_win:.2f}%</b>
                    </h3>
                    """,
                    unsafe_allow_html=True
                )

                # ---- DEBUG ----
                with st.expander("🔍 Show model input"):
                    st.dataframe(X)

import streamlit as st
import pandas as pd
import joblib
import os
def run():
    # ---------------- PAGE CONFIG ----------------
    st.set_page_config(
        page_title="♟ Chess Match Predictor",
        layout="centered"
    )

    st.title("♟ Chess Match Predictor")
    st.caption("Predict result using Elo ratings")

    # ---------------- LOAD DATA & MODEL ----------------
    @st.cache_resource
    def load_assets():
        BASE_DIR = os.path.dirname(__file__)

        model = joblib.load(os.path.join(BASE_DIR, "cmodel.joblib"))
        features = joblib.load(os.path.join(BASE_DIR, "cfeatures.joblib"))
        player_stats = joblib.load(os.path.join(BASE_DIR, "cplayer_stats.joblib"))

        return model, features, player_stats

    model, model_features, player_stats = load_assets()

    players = sorted(player_stats["name"].unique())
    rounds = [ 1. ,  2. ,  3. ,  4. ,  5. ,  6. ,  7. ,  8. ,  9. , 10. , 11. ,
        12. , 13. , 14. ,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  2.1,
            2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  3.1,  3.2,  3.3,  3.4,  3.5,
            3.6,  3.7,  4.1,  4.2,  4.3,  4.4,  4.5,  4.6,  5.1,  5.2,  5.3,
            5.4,  5.5,  5.6,  5.7,  6.1,  6.2,  6.3,  6.4,  6.5,  6.6,  7.1,
            7.2,  7.3,  7.4,  7.5,  7.6,  7.7,  7.8,  4.7, 15. ,  1.8,  2.8,
            2.9,  3.8,  1.9,  6.7,  7.9,  8.2,  8.3,  8.4,  8.6,  8.8,  8.5,
            8.1,  8.7, 16. , 17. , 18. , 19. , 20. , 21. , 22. , 23. , 24. ,
        25. , 26. , 27. , 28. , 29. , 30. , 31. , 32. , 33. , 34. , 35. ,
        36. , 37. , 38. , 39. , 40. , 41. , 42. , 43. , 44. , 45. , 46. ,
        47. , 48. ]

    # ---------------- INPUT UI ----------------
    col1, col2 = st.columns(2)

    with col1:
        round_selected = st.selectbox("🏁 Round", rounds)
        white_player = st.selectbox("♙ White Player", players)

    with col2:
        black_player = st.selectbox("♟ Black Player", players)

    # ---------------- BUILD INPUT ----------------
    def build_input(round_name, white, black):
        if white == black:
            return None

        w_row = player_stats[player_stats["name"] == white]
        b_row = player_stats[player_stats["name"] == black]

        if w_row.empty or b_row.empty:
            return None

        input_dict = {
            "round": rounds.index(round_name) + 1,
            "white_elo": w_row.iloc[0]["elo"],
            "black_elo": b_row.iloc[0]["elo"],
            "elo_diff": w_row.iloc[0]["elo"] - b_row.iloc[0]["elo"]
        }

        df = pd.DataFrame([input_dict])
        df = df.reindex(columns=model_features, fill_value=0)
        return df

    # ---------------- PREDICTION ----------------
    if st.button("🔮 Predict Result", use_container_width=True):

        X = build_input(round_selected, white_player, black_player)

        if X is None:
            st.error("White and Black players must be different")
        else:
            probs = model.predict_proba(X)[0]
            class_map = dict(zip(model.classes_, probs))

            black_win = class_map[0] * 100
            white_win = class_map[1] * 100
            draw      = class_map[2] * 100

            st.success("### 🏆 Match Outcome Probabilities")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("♟ Black Win", f"{black_win:.2f}%")

            with col2:
                st.metric("♙ White Win", f"{white_win:.2f}%")

            with col3:
                st.metric("🤝 Draw", f"{draw:.2f}%")


            # -------- Debug --------
            with st.expander("🔍 Model Input"):
                st.dataframe(X)

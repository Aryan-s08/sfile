import streamlit as st
import importlib.util
import os

st.set_page_config(page_title="⛹🏾‍♀️ Sports Predictor 🏌🏾‍♂️", layout="centered")
st.title("  ⛹🏾‍♀️ Sports Predictor 🏌🏾‍♂️")

choice = st.selectbox(
    "Choose Predictor",
    [
        "📈T20I Score Predictor (After 5 Overs)",
        "📊T20I Score Predictor (Any Over)",
        "🏆ODI Win Predictor",
        "⛹🏽‍♂️NBA Score Predictor",
        "🥎Tennis Win Predictor",
        "🎯T20I Win Predictor"
    ]
)

st.divider()

def load_and_run(app_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, app_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run()

BASE_DIR = os.getcwd()

if choice == "📈T20I Score Predictor (After 5 Overs)":
    load_and_run(
        os.path.join(BASE_DIR, "T20I Score Predictor (Updated)", "app.py"),
        "t20_app"
    )

elif choice == "📊T20I Score Predictor (Any Over)":
    load_and_run(
        os.path.join(BASE_DIR, "T20I Score Predictor", "app.py"),
        "t20_app"
    )

elif choice == "🏆ODI Win Predictor":
    load_and_run(
        os.path.join(BASE_DIR, "ODI Win Predictor", "app.py"),
        "odi_app"
    )

elif choice == "🎯La Liga Score Predictor":
    load_and_run(
        os.path.join(BASE_DIR, "La Liga Score Predictor", "app.py"),
        "t20_app"
    )

elif choice == "⛹🏽‍♂️NBA Score Predictor":
    load_and_run(
        os.path.join(BASE_DIR, "NBA Score Predictor", "app.py"),
        "nba_app"
    )

elif choice == "🥎Tennis Win Predictor":
    load_and_run(
        os.path.join(BASE_DIR, "Tennis Win Predictor", "app.py"),
        "tennis_app"
    )

elif choice == "🎯T20I Win Predictor":
    load_and_run(
        os.path.join(BASE_DIR, "T20I Win Predictor", "app.py"),
        "t20_app"
    )
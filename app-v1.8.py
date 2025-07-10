import streamlit as st
import pandas as pd
import numpy as np
import json
import random
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="centered")
st.title("🧙‍♂️ Wizard Stephen — Simulateur basé sur tes événements")

# -------- UPLOAD JSON --------
uploaded_json = st.file_uploader("📂 Uploade ton fichier JSON depuis Scribor", type=["json"])

if uploaded_json:
    try:
        data = json.load(uploaded_json)
        st.success(f"{len(data)} événements chargés avec succès.")
    except Exception as e:
        st.error(f"Erreur lors de l'import du fichier JSON : {e}")
        st.stop()
    
    # -------- PARAMÈTRES --------
    st.sidebar.header("🎛️ Paramètres de simulation")
    n_simulations = st.sidebar.slider("Nombre de simulations", 10, 2000, 500)
    seuil_succès = st.sidebar.slider("Seuil score pour succès", 0.6, 1.0, 0.7)
    seuil_partiel = st.sidebar.slider("Seuil score pour partiel", 0.3, 0.59, 0.5)

    # Préparation des événements
    events = [(e["nom"], float(e["impact"]), float(e["base_prob"])) for e in data]

    def simulate_one(events, seuil_succès, seuil_partiel):
        history = []
        score = 0.0
        for nom, impact, base_prob in events:
            if random.random() < base_prob:
                history.append(nom)
                score += impact
        outcome = "échec"
        if score >= seuil_succès:
            outcome = "succès"
        elif score >= seuil_partiel:
            outcome = "partiel"
        return {"steps": history, "outcome": outcome, "score": round(score, 2)}

    df = pd.DataFrame([simulate_one(events, seuil_succès, seuil_partiel) for _ in range(n_simulations)])

    # --------- SYNTHÈSE ---------
    with st.expander("🧠 Synthèse stratégique", expanded=True):
        total_succ = len(df[df["outcome"] == "succès"])
        total_fail = len(df[df["outcome"] == "échec"])

        def extract_top_pattern(df, outcome_filter):
            steps = df[df["outcome"] == outcome_filter]["steps"]
            combos = Counter()
            for s in steps:
                for p in combinations(sorted(set(s)), 2):
                    combos[p] += 1
            return combos.most_common(1)[0] if combos else (None, 0)

        def extract_pattern_perdant_all(df):
            fails = df[df["outcome"] == "échec"]["steps"]
            combos_fail = Counter()
            for s in fails:
                for p in combinations(sorted(set(s)), 2):
                    combos_fail[p] += 1
            return combos_fail.most_common(5) if combos_fail else []

        p_win, n_win = extract_top_pattern(df, "succès")
        if p_win:
            pct_win = n_win / total_succ * 100 if total_succ > 0 else 0
            st.markdown(f"🔍 **Pattern gagnant détecté** : `{p_win[0]}` + `{p_win[1]}` ({n_win} scénarios, soit {pct_win:.1f} % des succès)")
            st.markdown("✅ Ce pattern est rare dans les échecs → il semble fortement corrélé au succès.")
            st.markdown("### 🛠️ Ligne d'action suggérée :")
            st.markdown(f"→ Favoriser l'apparition de `{p_win[0]}`, puis `{p_win[1]}` dès que possible.")
        else:
            st.info("Pas assez de données pour détecter un pattern gagnant clair.")

        st.markdown("---")

        perdants = extract_pattern_perdant_all(df)
        if perdants:
            st.markdown("❌ **Principaux patterns présents dans les échecs :**")
            for combo, count in perdants:
                pct = count / total_fail * 100 if total_fail > 0 else 0
                st.markdown(f"- `{combo[0]}` + `{combo[1]}` → présent dans {count} scénarios d’échec ({pct:.1f} % des échecs)")
            st.markdown("### ⚠️ Ligne d’alerte stratégique :")
            st.markdown("→ Surveille ces combinaisons dans tes scénarios : elles sont fortement présentes dans les échecs.")

            st.markdown("### 📉 Visualisation des patterns d’échec")
            labels = [f"{c[0]} + {c[1]}" for c, _ in perdants]
            values = [count for _, count in perdants]
            fig, ax = plt.subplots()
            sns.barplot(x=values, y=labels, ax=ax, palette="Reds_r")
            ax.set_title("Patterns fréquents dans les échecs")
            ax.set_xlabel("Nombre d’occurrences")
            ax.set_ylabel("Combinaisons d’événements")
            st.pyplot(fig)
        else:
            st.info("Aucun enchaînement d'événements fréquent détecté dans les échecs.")

else:
    st.info("Commence par uploader un fichier JSON généré avec Scribor.")

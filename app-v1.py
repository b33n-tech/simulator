import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

st.title("🔮 Doctor Strange Simulator — 1000 Futures")

st.markdown("Simule des centaines de scénarios pour identifier ceux qui mènent au succès. Trouve ta meilleure ligne d’action.")

# --------- INPUTS DE CONTEXTE ---------
st.sidebar.header("Contexte initial")
motivation = st.sidebar.slider("Motivation initiale", 0.0, 1.0, 0.7)
réseau = st.sidebar.selectbox("Niveau de réseau/soutien", ["faible", "modéré", "fort"])
budget = st.sidebar.selectbox("Budget disponible", ["faible", "moyen", "élevé"])
n_simulations = st.sidebar.slider("Nombre de scénarios à simuler", 10, 2000, 1000)

# --------- PARAMÈTRES INTERNES ---------
scenarios = []

# Score de base selon le contexte (simplifié)
context_score = motivation * 0.4
if réseau == "modéré": context_score += 0.2
elif réseau == "fort": context_score += 0.35
if budget == "moyen": context_score += 0.2
elif budget == "élevé": context_score += 0.35

# --------- MOTEUR DE SCÉNARIO ---------
possible_steps = [
    "formulation de l’idée",
    "recherche de soutien",
    "refus initial",
    "pivot stratégique",
    "soutien obtenu",
    "implémentation",
    "difficulté majeure",
    "retour positif",
    "impact visible",
]

def simulate_one():
    history = []
    success_score = context_score + np.random.normal(0, 0.1)
    
    # Génération des étapes
    for i in range(random.randint(5, 8)):
        step = random.choice(possible_steps)
        history.append(step)
        if "difficulté" in step:
            success_score -= 0.15
        if "soutien obtenu" in step:
            success_score += 0.2
        if "pivot" in step:
            success_score += 0.1

    outcome = "échec"
    bénéfices = []

    if success_score >= 0.65:
        outcome = "succès"
        bénéfices = ["impact local", "visibilité", "mobilisation"]
    elif success_score >= 0.5:
        outcome = "partiel"
        bénéfices = ["réseau élargi", "leçon apprise"]
    else:
        outcome = "échec"
        bénéfices = ["expérience"]

    return {
        "steps": history,
        "outcome": outcome,
        "bénéfices": bénéfices,
        "score": round(success_score, 2)
    }

# --------- LANCEMENT DES SIMULATIONS ---------
st.subheader("⏳ Simulation en cours...")

results = []
for _ in range(n_simulations):
    results.append(simulate_one())

# --------- ANALYSE DES SORTIES ---------
df = pd.DataFrame(results)
outcome_counts = df["outcome"].value_counts(normalize=True).round(2) * 100

st.markdown("### 📊 Résultats globaux")
st.write(outcome_counts.astype(str) + " %")

fig, ax = plt.subplots()
df["outcome"].value_counts().plot(kind='bar', ax=ax, color=['green', 'orange', 'red'])
ax.set_title("Distribution des issues finales")
st.pyplot(fig)

# --------- CAS GAGNANTS ---------
st.markdown("### 🟢 Scénarios Gagnants")
top_winners = df[df["outcome"] == "succès"].head(3)

for i, row in top_winners.iterrows():
    st.markdown(f"**Scénario #{i}** — Score: {row['score']}")
    st.markdown(" → **Étapes**: " + " → ".join(row['steps']))
    st.markdown(" → **Bénéfices**: " + ", ".join(row['bénéfices']))
    st.markdown("---")

st.markdown("Tu peux modifier le contexte à gauche pour explorer d’autres futurs 🧭")

from collections import Counter

def count_steps_by_outcome(df, outcome_label):
    steps = df[df["outcome"] == outcome_label]["steps"]
    all_steps = []
    for s in steps:
        all_steps.extend(s)
    return Counter(all_steps)

st.markdown("### 🧠 Analyse des facteurs clés")

success_steps = count_steps_by_outcome(df, "succès")
fail_steps = count_steps_by_outcome(df, "échec")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🟢 Étapes fréquentes dans les succès")
    for step, count in success_steps.most_common(5):
        st.markdown(f"- **{step}** : {count} fois")

with col2:
    st.markdown("#### 🔴 Étapes fréquentes dans les échecs")
    for step, count in fail_steps.most_common(5):
        st.markdown(f"- **{step}** : {count} fois")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter
from itertools import combinations

# --------- BIBLIOTHÈQUE DE SCÉNARIOS ---------
def get_scenario_components(scenario_type):
    if scenario_type == "Projet d’innovation locale":
        return {
            "contexte": {"motivation": 0.7, "reseau": "modéré", "budget": "moyen"},
            "événements": [
                ("formulation idée", 0.0),
                ("refus de subvention", -0.3),
                ("soutien bénévole", +0.3),
                ("pivot stratégique", +0.4),
                ("retard administratif", -0.2),
                ("soutien mairie", +0.5),
                ("épuisement mental", -0.5),
                ("mobilisation locale", +0.2),
                ("lancement test", +0.3)
            ]
        }
    elif scenario_type == "Recherche d’emploi en reconversion":
        return {
            "contexte": {"motivation": 0.8, "reseau": "faible", "budget": "faible"},
            "événements": [
                ("mise à jour CV", 0.0),
                ("refus d’entretien", -0.3),
                ("contact LinkedIn", +0.2),
                ("formation courte", +0.3),
                ("burnout", -0.4),
                ("simulation d’entretien", +0.2),
                ("job dating", +0.3),
                ("recentrage objectif", +0.2),
                ("proposition reçue", +0.5)
            ]
        }
    elif scenario_type == "Lancement de projet entrepreneurial tech":
        return {
            "contexte": {"motivation": 0.9, "reseau": "fort", "budget": "moyen"},
            "événements": [
                ("idée produit", 0.0),
                ("prototype rapide", +0.4),
                ("bug bloquant", -0.3),
                ("pitch public", +0.3),
                ("recherche associé", +0.2),
                ("feedback client", +0.3),
                ("absence de traction", -0.3),
                ("demo à un incubateur", +0.5),
                ("retrait associé", -0.4)
            ]
        }

# --------- SCORE DE CONTEXTE ---------
def contexte_score(context):
    score = context["motivation"] * 0.4
    if context["reseau"] == "modéré": score += 0.2
    elif context["reseau"] == "fort": score += 0.35
    if context["budget"] == "moyen": score += 0.2
    elif context["budget"] == "élevé": score += 0.35
    return score

# --------- SIMULATION ---------
def simulate_one(event_list, base_score):
    history = []
    score = base_score + np.random.normal(0, 0.1)
    n_steps = random.randint(6, 9)
    chosen_events = random.choices(event_list, k=n_steps)
    for evt, impact in chosen_events:
        history.append(evt)
        score += impact
    outcome = "échec"
    if score >= 0.7:
        outcome = "succès"
    elif score >= 0.5:
        outcome = "partiel"
    return {"steps": history, "outcome": outcome, "score": round(score, 2)}

# --------- PATTERNS ---------
def extract_top_pattern(df, outcome_filter):
    steps = df[df["outcome"] == outcome_filter]["steps"]
    combos = Counter()
    for s in steps:
        pairs = combinations(sorted(set(s)), 2)
        for p in pairs:
            combos[p] += 1
    return combos.most_common(1)[0] if combos else (None, 0)

def extract_pattern_perdant(df):
    fails = df[df["outcome"] == "échec"]["steps"]
    success = df[df["outcome"] == "succès"]["steps"]
    combos_fail = Counter()
    combos_success = Counter()
    for s in fails:
        for p in combinations(sorted(set(s)), 2):
            combos_fail[p] += 1
    for s in success:
        for p in combinations(sorted(set(s)), 2):
            combos_success[p] += 1
    for combo, fail_count in combos_fail.most_common():
        success_count = combos_success.get(combo, 0)
        if success_count < fail_count * 0.2:
            return combo, fail_count, success_count
    return None, 0, 0

# --------- NARRATEUR ---------
def generate_narrative(steps):
    if not steps: return "Scénario vide."
    first, last = steps[0], steps[-1]
    key_events = {
        "pivot stratégique": "Changement d’orientation",
        "refus de subvention": "Obstacle majeur initial",
        "soutien mairie": "Tournant institutionnel",
        "soutien bénévole": "Renfort humain clé",
        "burnout": "Effondrement énergétique",
        "formation courte": "Montée en compétences",
        "demo à un incubateur": "Visibilité stratégique",
        "absence de traction": "Réaction faible du marché",
        "proposition reçue": "Objectif atteint"
    }
    turning_points = [e for e in steps if e in key_events]
    narrative = f"\n📍 Départ : **{first}**."
    if turning_points:
        narrative += "\n🔀 Moments clés :"
        for e in turning_points:
            narrative += f"\n- {key_events[e]}"
    narrative += f"\n🎯 Fin : **{last}**."
    return narrative

# --------- FACTEURS CLÉS ---------
def compute_key_factors(df):
    success = df[df["outcome"] == "succès"]["steps"]
    fail = df[df["outcome"] == "échec"]["steps"]
    total_succ = len(success)
    total_fail = len(fail)
    all_events = set(e for s in df["steps"] for e in s)
    rows = []
    for event in all_events:
        count_succ = sum(event in s for s in success)
        count_fail = sum(event in s for s in fail)
        p_succ = (count_succ / total_succ * 100) if total_succ > 0 else 0
        p_fail = (count_fail / total_fail * 100) if total_fail > 0 else 0
        if p_succ > 60 and p_fail < 30:
            impact = "✅ Fort levier positif"
        elif p_fail > 50 and p_succ < 20:
            impact = "❌ Fort facteur d’échec"
        elif p_succ > p_fail:
            impact = "🟢 Plutôt favorable"
        elif p_fail > p_succ:
            impact = "🔴 Plutôt défavorable"
        else:
            impact = "🔶 Neutre / ambivalent"
        rows.append({"Événement": event, "% succès": round(p_succ, 1), "% échec": round(p_fail, 1), "Impact stratégique": impact})
    return pd.DataFrame(rows).sort_values(by="Impact stratégique")

# --------- UI ---------
st.set_page_config(layout="centered")
st.title("🔮 Simulateur de futurs — V5 Stratégique")

scenario_type = st.selectbox("Choisis un cas de figure", [
    "Projet d’innovation locale",
    "Recherche d’emploi en reconversion",
    "Lancement de projet entrepreneurial tech"
])

n_simulations = st.slider("Nombre de simulations", 10, 2000, 500)
components = get_scenario_components(scenario_type)
base = contexte_score(components["contexte"])
df = pd.DataFrame([simulate_one(components["événements"], base) for _ in range(n_simulations)])

st.subheader("📊 Répartition des issues")
outcome_counts = df["outcome"].value_counts(normalize=True).round(2) * 100
st.write(outcome_counts.astype(str) + " %")

fig, ax = plt.subplots()
df["outcome"].value_counts().plot(kind='bar', ax=ax, color=['green', 'orange', 'red'])
ax.set_title("Répartition des scénarios")
st.pyplot(fig)

# --------- SYNTHÈSE ---------
with st.expander("🧠 Synthèse stratégique", expanded=True):
    total_succ = len(df[df["outcome"] == "succès"])
    total_fail = len(df[df["outcome"] == "échec"])

    # Pattern gagnant
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

    # Pattern perdant
    p_lose, n_fail, n_succ = extract_pattern_perdant(df)
    if p_lose:
        pct_fail = n_fail / total_fail * 100 if total_fail > 0 else 0
        pct_succ = n_succ / total_succ * 100 if total_succ > 0 else 0
        st.markdown(f"❌ **Pattern perdant détecté** : `{p_lose[0]}` + `{p_lose[1]}` ({n_fail} scénarios, soit {pct_fail:.1f} % des échecs)")
        st.markdown(f"⚠️ Ce pattern est rare dans les succès ({n_succ} scénarios, soit {pct_succ:.1f} % des succès).")
        st.markdown("### ⚠️ Ligne d’alerte stratégique :")
        st.markdown(f"→ Surveille l’enchaînement `{p_lose[0]}` → `{p_lose[1]}` : il est fortement corrélé aux échecs.")
        st.markdown("S’il apparaît, anticipe un pivot, renforce des soutiens, ou requalifie l’objectif dès ce stade.")
    else:
        st.info("Aucun pattern perdant clair détecté.")

# --------- NARRATIFS ---------
st.markdown("### 🏆 Exemples de scénarios gagnants interprétés")
top = df[df["outcome"] == "succès"].head(3)
for i, row in top.iterrows():
    st.markdown(f"**Scénario #{i}** — Score: {row['score']}")
    st.markdown(generate_narrative(row['steps']))
    st.markdown("---")

# --------- FACTEURS CLÉS ---------
st.markdown("## 🧮 Facteurs clés")
st.markdown("Analyse croisée des événements fréquents dans les succès et les échecs.")
factors_df = compute_key_factors(df)
st.dataframe(factors_df, use_container_width=True)

# Ajout de diagrammes
fig2, ax2 = plt.subplots(figsize=(10, 5))
factors_df_sorted = factors_df.sort_values(by="% succès", ascending=False)
ax2.barh(factors_df_sorted["Événement"], factors_df_sorted["% succès"], color='green', label='% Succès')
ax2.barh(factors_df_sorted["Événement"], -factors_df_sorted["% échec"], color='red', label='% Échec')
ax2.set_title("% de présence des événements dans les scénarios de succès et d'échec")
ax2.legend()
st.pyplot(fig2)

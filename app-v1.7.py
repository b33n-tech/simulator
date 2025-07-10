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
                {"nom": "formulation idée", "impact": 0.0, "base_prob": 0.1},
                {"nom": "refus de subvention", "impact": -0.3, "base_prob": 0.1},
                {"nom": "soutien bénévole", "impact": +0.3, "base_prob": 0.15},
                {"nom": "pivot stratégique", "impact": +0.4, "base_prob": 0.1},
                {"nom": "retard administratif", "impact": -0.2, "base_prob": 0.1},
                {"nom": "soutien mairie", "impact": +0.5, "base_prob": 0.1},
                {"nom": "épuisement mental", "impact": -0.5, "base_prob": 0.05},
                {"nom": "mobilisation locale", "impact": +0.2, "base_prob": 0.15},
                {"nom": "lancement test", "impact": +0.3, "base_prob": 0.15},
            ],
        }
    elif scenario_type == "Recherche d’emploi en reconversion":
        return {
            "contexte": {"motivation": 0.8, "reseau": "faible", "budget": "faible"},
            "événements": [
                {"nom": "mise à jour CV", "impact": 0.0, "base_prob": 0.15},
                {"nom": "refus d’entretien", "impact": -0.3, "base_prob": 0.15},
                {"nom": "contact LinkedIn", "impact": +0.2, "base_prob": 0.15},
                {"nom": "formation courte", "impact": +0.3, "base_prob": 0.15},
                {"nom": "burnout", "impact": -0.4, "base_prob": 0.05},
                {"nom": "simulation d’entretien", "impact": +0.2, "base_prob": 0.15},
                {"nom": "job dating", "impact": +0.3, "base_prob": 0.15},
                {"nom": "recentrage objectif", "impact": +0.2, "base_prob": 0.1},
                {"nom": "proposition reçue", "impact": +0.5, "base_prob": 0.1},
            ],
        }
    elif scenario_type == "Lancement de projet entrepreneurial tech":
        return {
            "contexte": {"motivation": 0.9, "reseau": "fort", "budget": "moyen"},
            "événements": [
                {"nom": "idée produit", "impact": 0.0, "base_prob": 0.1},
                {"nom": "prototype rapide", "impact": +0.4, "base_prob": 0.15},
                {"nom": "bug bloquant", "impact": -0.3, "base_prob": 0.1},
                {"nom": "pitch public", "impact": +0.3, "base_prob": 0.15},
                {"nom": "recherche associé", "impact": +0.2, "base_prob": 0.15},
                {"nom": "feedback client", "impact": +0.3, "base_prob": 0.15},
                {"nom": "absence de traction", "impact": -0.3, "base_prob": 0.1},
                {"nom": "demo à un incubateur", "impact": +0.5, "base_prob": 0.1},
                {"nom": "retrait associé", "impact": -0.4, "base_prob": 0.05},
            ],
        }
    else:
        return {"contexte": {}, "événements": []}

def contexte_score(context):
    score = context["motivation"] * 0.4
    if context["reseau"] == "modéré":
        score += 0.2
    elif context["reseau"] == "fort":
        score += 0.35
    if context["budget"] == "moyen":
        score += 0.2
    elif context["budget"] == "élevé":
        score += 0.35
    return score

def simulate_one(event_list, base_score, weights):
    history = []
    score = base_score + np.random.normal(0, 0.1)
    n_steps = random.randint(6, 9)
    # Utilisation des pondérations modifiées
    probas = np.array([weights.get(e["nom"], e["base_prob"]) for e in event_list])
    probas = probas / probas.sum()
    chosen_events = np.random.choice(event_list, size=n_steps, replace=True, p=probas)
    for evt in chosen_events:
        history.append(evt["nom"])
        score += evt["impact"]
    if score >= 0.7:
        outcome = "succès"
    elif score >= 0.5:
        outcome = "partiel"
    else:
        outcome = "échec"
    return {"steps": history, "outcome": outcome, "score": round(score, 2)}

def extract_top_pattern(df, outcome_filter):
    steps = df[df["outcome"] == outcome_filter]["steps"]
    combos = Counter()
    for s in steps:
        pairs = combinations(sorted(set(s)), 2)
        for p in pairs:
            combos[p] += 1
    return combos.most_common(1)[0] if combos else (None, 0)

def extract_pattern_perdant_all(df):
    fails = df[df["outcome"] == "échec"]["steps"]
    combos_fail = Counter()
    for s in fails:
        for p in combinations(sorted(set(s)), 2):
            combos_fail[p] += 1
    return combos_fail.most_common(5) if combos_fail else []

# --- Streamlit UI ---
st.set_page_config(layout="centered")
st.title("🔮 Simulateur de futurs — V6 avec pondérations & immersion")

scenario_type = st.selectbox("Choisis un cas de figure", [
    "Projet d’innovation locale",
    "Recherche d’emploi en reconversion",
    "Lancement de projet entrepreneurial tech"
])

components = get_scenario_components(scenario_type)
base = contexte_score(components["contexte"])

st.markdown("## 🎛️ Ajuste les pondérations d'apparition des événements (entre 0 et 1)")

weights = {}
cols = st.columns(3)
for i, evt in enumerate(components["événements"]):
    col = cols[i % 3]
    w = col.slider(f"{evt['nom']} (impact: {evt['impact']})", 0.0, 1.0, float(evt["base_prob"]), 0.01)
    weights[evt["nom"]] = w

n_simulations = st.slider("Nombre de simulations", 10, 2000, 500)

# Génération des simulations
df = pd.DataFrame([simulate_one(components["événements"], base, weights) for _ in range(n_simulations)])

st.markdown("## 📊 Résumé des simulations")
st.write(df["outcome"].value_counts())

# Synthèse patterns gagnants / perdants
with st.expander("🧠 Synthèse stratégique", expanded=True):
    total_succ = len(df[df["outcome"] == "succès"])
    total_fail = len(df[df["outcome"] == "échec"])

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

# --- Exploration immersive d'un scénario ---
st.markdown("---")
st.markdown("## 🎭 Exploration immersive d'un scénario")

filter_outcome = st.selectbox("Filtrer par résultat", options=["Tous", "succès", "partiel", "échec"])

if filter_outcome == "Tous":
    filtered_df = df.copy()
else:
    filtered_df = df[df["outcome"] == filter_outcome]

idx = st.selectbox(f"Sélectionne un scénario parmi {len(filtered_df)}", filtered_df.index)

scenario = filtered_df.loc[idx]
steps = scenario["steps"]
score = scenario["score"]
outcome = scenario["outcome"]

st.markdown(f"### Résultat : **{outcome.upper()}** avec un score final de {score}")

# Affichage timeline et analyse
st.markdown("#### Timeline des événements")
cumulative_score = contexte_score(components["contexte"]) + np.random.normal(0,0.1)
scores_timeline = [cumulative_score]
for evt_name in steps:
    # chercher impact
    impact = next(e["impact"] for e in components["événements"] if e["nom"] == evt_name)
    cumulative_score += impact
    scores_timeline.append(cumulative_score)

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(range(len(scores_timeline)), scores_timeline, marker='o')
for i, evt_name in enumerate(steps, 1):
    impact = next(e["impact"] for e in components["événements"] if e["nom"] == evt_name)
    color = "green" if impact >= 0 else "red"
    ax.annotate(evt_name, (i, scores_timeline[i]), color=color, fontsize=9, rotation=45, ha="right", va="bottom")
ax.axhline(y=0.7, color='blue', linestyle='--', label='Seuil succès')
ax.axhline(y=0.5, color='orange', linestyle='--', label='Seuil partiel')
ax.set_xlabel("Étapes")
ax.set_ylabel("Score cumulatif")
ax.set_title("Évolution du score dans ce scénario")
ax.legend()
plt.tight_layout()
st.pyplot(fig)

# Mini-analyse pondérée des événements
st.markdown("#### Analyse détaillée du scénario")
counter_evt = Counter(steps)
total_evt = len(steps)
st.write(f"Nombre d’étapes : {total_evt}")

evt_weights = []
for evt_name, count_evt in counter_evt.items():
    impact = next(e["impact"] for e in components["événements"] if e["nom"] == evt_name)
    weighted_impact = impact * count_evt
    evt_weights.append((evt_name, count_evt, impact, weighted_impact))

evt_weights.sort(key=lambda x: abs(x[3]), reverse=True)
df_evt = pd.DataFrame(evt_weights, columns=["Événement", "Occurrences", "Impact unitaire", "Impact total (pondéré)"])
st.dataframe(df_evt)

# Interprétation synthétique
st.markdown("##### Synthèse interprétative")
strong_events = [e for e in evt_weights if abs(e[3]) > 0.3]
if strong_events:
    for evt_name, count_evt, impact, weighted_impact in strong_events:
        action = "a permis un gain" if weighted_impact > 0 else "a freiné la progression"
        st.markdown(f"> - L’événement **‘{evt_name}’** ({count_evt} fois) {action} de {abs(weighted_impact):.2f} points, important pour ce scénario.")
else:
    st.markdown("Pas d’événement dominant particulier dans ce scénario.")


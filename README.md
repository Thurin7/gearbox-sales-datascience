# 🔧 Gearbox Sales Analysis — Data Science Project

> Analyse et prédiction des ventes de boîtes de vitesses IVECO/ZF (pièces Reman vs New) sur les marchés France, Italie et Allemagne — 2023/2024

---

## 📌 Contexte

Ce projet simule une mission Data Science réelle dans un contexte **B2B / pièces détachées industrielles**.  
L'objectif est d'analyser les performances commerciales, d'identifier les drivers du chiffre d'affaires, et de construire un modèle prédictif pour anticiper les ventes.

---

## 🎯 Objectifs

- **Comprendre** la structure des ventes par région, type de produit et référence
- **Comparer** les performances REMAN (reconditionné) vs NEW (neuf)
- **Prédire** le chiffre d'affaires par transaction via des modèles ML
- **Segmenter** les références produits pour orienter la stratégie commerciale

---

## 📊 Données

| Indicateur | Valeur |
|------------|--------|
| Période | Janvier 2023 — Mars 2024 |
| Transactions | 243 lignes |
| Références produits | 54 références uniques |
| Marchés | France, Italie, Allemagne |
| CA total | 3 725 615 € |
| Part REMAN | 51.3% |
| Part NEW | 48.7% |

**Variables disponibles :**
- `Year`, `Month` — dimension temporelle
- `Part Number`, `Part Description` — référence produit
- `Product type` — Reman ou Genuine (New)
- `Market Region` — marché géographique
- `REMAN Quantity`, `REMAN Gross sales` — ventes pièces reconditionnées
- `NEW Quantity`, `NEW Gross sales` — ventes pièces neuves

---

## 🗂️ Structure du projet

```
gearbox-sales-datascience/
│
├── 📓 Gearbox_Sales_DataScience.ipynb   # Notebook principal
├── 📊 Gearbox_data.xlsx                 # Données sources
├── 🖥️  app.py                           # Dashboard Streamlit interactif
├── 📄 README.md                         # Ce fichier
└── 📦 requirements.txt                  # Dépendances Python
```

---

## 🔬 Méthodologie

### 1. 🔍 Nettoyage & Audit qualité
- Détection et correction d'une anomalie : région `"France "` (espace parasite) créant un doublon
- Vérification des valeurs manquantes, types, doublons

### 2. 📊 Analyse Exploratoire (EDA)
- CA par région, type de produit, référence
- Distribution des prix unitaires REMAN vs NEW
- **Premium index** : la pièce NEW coûte en moyenne **33% plus cher** que la REMAN
- Heatmap des corrélations

### 3. 📈 Analyse Temporelle
- Évolution mensuelle du CA avec moyenne mobile 3 mois
- Comparaison 2023 vs 2024 + projection annualisée
- Identification des pics de ventes (Juillet & Septembre 2023)

### 4. ⚙️ Feature Engineering
- Encodage `LabelEncoder` des variables catégorielles
- Variables cycliques **sin/cos** pour les mois (capture la périodicité)
- Ratios métier : `Premium_index`, `Ratio_REMAN`, `Prix_gap`, `Qty_ratio`

### 5. 🤖 Modélisation prédictive
Comparaison de 4 modèles sur MAE et R² :

| Modèle | Type |
|--------|------|
| Ridge | Régression linéaire régularisée (L2) |
| Random Forest | Ensemble d'arbres (bagging) |
| XGBoost | Gradient Boosting optimisé |
| LightGBM | Gradient Boosting rapide (GOSS + EFB) |

### 6. 🔵 Clustering K-Means
Segmentation des 54 références en **3 profils** :
- 🔴 **Top ventes** — CA élevé, prix premium → stock prioritaire
- 🟡 **Standard** — mix équilibré → gestion courante
- 🔵 **Faible rotation** — CA marginal → candidats au déréférencement

---

## 💡 Principaux Résultats

| Finding | Détail |
|---------|--------|
| 🥇 1er marché | France (CA total le plus élevé) |
| 📈 Croissance | 2024 en hausse vs 2023 sur tous les marchés |
| 💰 Premium NEW | +33% vs REMAN en moyenne |
| 🔑 Driver n°1 du CA | Quantité vendue (feature importance ML) |
| ⚠️ Levier identifié | REMAN sous-exploité en Allemagne |

---

## 🛠️ Stack technique

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-pink)

```
pandas · numpy · matplotlib · seaborn
scikit-learn · xgboost · lightgbm
streamlit · plotly
```

---

## 🚀 Lancer le projet

### 1. Cloner le repo
```bash
git clone https://github.com/TON_USERNAME/gearbox-sales-datascience.git
cd gearbox-sales-datascience
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer le notebook
```bash
jupyter notebook Gearbox_Sales_DataScience.ipynb
```

### 4. Lancer le dashboard Streamlit
```bash
streamlit run app.py
```

---

## 👤 Auteur

**[Ton Prénom Nom]**  
Étudiant en alternance — Data Science / Data Analyst  
[LinkedIn](https://linkedin.com/in/ton-profil) · [GitHub](https://github.com/ton-username)

---

*Projet réalisé dans le cadre d'une candidature en alternance Data Science — 2025*

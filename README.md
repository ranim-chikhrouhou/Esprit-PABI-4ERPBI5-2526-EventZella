# EventZilla BI — Marketplace Intelligence Solution

> **Couche ML + Streamlit** : notebooks, scripts et application sous **`ML/`** ; configuration Streamlit dans **`.streamlit/`**. Exclusions Git, synchronisation et commandes : **`LISEZMOI_gitMachine.md`**. Dépôt : [Esprit-PABI-4ERPBI5-2526-EventZella](https://github.com/ranim-chikhrouhou/Esprit-PABI-4ERPBI5-2526-EventZella.git).

## Équipe : ctrlAltWin (Classe 4Bi5)

- **Project Lead :** Ranim Chikhrouhou  
- **Institution :** Esprit (Spécialisation Business Intelligence)  
- **Année :** 2026  

---

## Présentation du projet

**EventZilla BI** est une solution décisionnelle de bout en bout pour une marketplace événementielle tunisienne. Le projet transforme des données brutes (SQL Server, Excel/CSV, etc.) en pilotage stratégique : croissance, rentabilité, satisfaction client — **Staging → Data Warehouse (schéma en étoile) → Power BI**, avec automatisation et **Row-Level Security (RLS)**.

---

## Structure du dépôt (Git)

| Dossier | Contenu |
|---------|---------|
| **`Reports/`** | Rapport Power BI (`.pbix`). |
| **`DataBase/`** | Sauvegardes / artefacts base (selon organisation du dépôt). |
| **`Data_Sources/`** | Jeux Excel sources. |
| **`ML/`** | **Machine Learning** : notebooks `00_A`–`06_B`, `05`, scripts `run_*.py`, `streamlit_app.py`, `DataExcell/`, `processed/` (JSON), `models_artifacts/` (métriques JSON). Voir `ML/README.md`. |
| **`.streamlit/`** | Thème et options de l’app Streamlit. |
| **`Liste_Des_Kpis_Updated_English_DAX.md`** | Documentation KPI (racine du dépôt, si présent). |

Sur une copie de travail étendue du projet, d’autres dossiers peuvent exister localement (`docs/`, `scripts/`, `deliverables/`, etc.) ; ils ne sont pas tous obligatoirement sur cette branche.

**Variables d’environnement (ML / SQL) :** voir `ML/ml_paths.py` (`EVENTZILLA_SQL_*`, `EVENTZILLA_ML_SQL_ONLY`, etc.). Ne pas committer de secrets.

---

## Architecture technique & pipeline (E-LT)

1. **Sources :** **Staging Area (SA)** sur **SQL Server** ; fichiers **Excel/CSV** (benchmarks, marketing).  
2. **Stockage :** **Data Warehouse (DW)** en **schéma en étoile** pour les mesures DAX.  
3. **Visualisation :** Power BI Desktop & Service.  
4. **Middleware :** **On-premises Data Gateway** entre le DW local et le cloud.  

---

## Automatisation & Executive Summary

- **Rafraîchissement :** planification bi-quotidienne (**8h00 / 14h00**) via la Gateway.  
- **Executive Summary :** page avec **Smart Narrative** pour des résumés textuels automatiques des KPI.  

---

## Gouvernance & sécurité (RLS)

Filtrage dynamique avec **`USERPRINCIPALNAME()`** :

- **Marketing :** CAC, ROI campagnes, funnel.  
- **Finance :** marges, commissions, rentabilité.  
- **Relation client :** NPS, plaintes, taux de résolution.  

---

## ML & Streamlit (raccourci)

```text
py -3.11 -m pip install -r ML/requirements.txt
streamlit run ML/streamlit_app.py
```

Ordre conseillé des notebooks : **00_A** → **01_E** / **02_C** / **03_D** / **04_F** → **05** ; **06_B** documente le critère « compréhension des modèles ».

---

## KPIs (aperçu)

Plus de **50 indicateurs** : performance commerciale (conversion, AOV), analyse financière (revenu, commissions), fidélité (rétention, NPS), etc.

---

*Projet académique — équipe **ctrlAltWin** — solution BI moderne, sécurisée et documentée.*

# EventZilla BI — Marketplace Intelligence Solution

> **Paquet `gitMachine` (ML + Streamlit)** : l’arborescence **machine learning** et le déploiement Streamlit sont sous **`ML/`** ; configuration Streamlit dans **`.streamlit/`**. Pour Git, exclusions et commandes `git push`, lire **`LISEZMOI_gitMachine.md`**. Dépôt public : [Esprit-PABI-4ERPBI5-2526-EventZella sur GitHub](https://github.com/ranim-chikhrouhou/Esprit-PABI-4ERPBI5-2526-EventZella.git).

## Équipe : ctrlAltWin (Classe 4Bi5)

- **Project Lead :** Ranim Chikhrouhou  
- **Institution :** Esprit (Spécialisation Business Intelligence)  
- **Année :** 2026  

---

## Présentation du projet

**EventZilla BI** est une solution décisionnelle de bout en bout pour une marketplace événementielle. Le projet transforme des données brutes (SQL Server, Excel/CSV, etc.) en un écosystème de pilotage : **Staging → Data Warehouse (schéma en étoile) → Power BI**, avec automatisation et sécurité (RLS).

---

## Arborescence du dépôt

À la **racine** du projet, seuls `README.md` et `.gitignore` restent volontairement (convention Git). Tout le reste est regroupé par domaine :

| Dossier | Contenu |
|---------|---------|
| **`scripts/`** | Scripts Python et PowerShell de génération (`build_*.py`, `convert_*.ps1`). |
| **`deliverables/`** | Sorties Markdown / HTML / PDF des tableaux de bord (fichiers générés ou exports). |
| **`docs/`** | `DAX_Measures.md`, `Liste_Des_Kpis_Updated_English_DAX.md`, `eventzilla/` (source `EventZilla_Dashboards_KPIs_Objectifs.md`), `references/` (PDF optionnels). |
| **`config/`** | `requirements.txt` pour les scripts Python. |
| **`Reports/`** | Rapport Power BI (`.pbix`) — voir `Reports/README.md`. |
| **`Database/`** | Emplacement alternatif / doc ; sauvegardes SQL principales : **`FilesMachine/DB/`** (`DW_Eventzilla`, `SA_eventzilla`). |
| **`FilesMachine/`** | **DB** (sauvegardes DW/SA), dossiers optionnels `data_original` / `datascrapped`, PDF référence — voir `FilesMachine/README.md`. |
| **`Data_Sources/`** | Fichiers sources Excel — voir `Data_Sources/README.md`. |
| **`ScriptsDiagrams/`** | Schémas DWH (DBML, Mermaid, PlantUML). |
| **`Trees/`** | Arbres de décision + `ArbreDecsiontxt.txt`. |
| **`EDAs/`** | Notebooks d’analyse exploratoire. |
| **`ML/`** | **Machine Learning** : notebooks, scripts `run_*.py`, et **fichiers `.csv` source** (tout sous `ML/`, hors `processed/` et `models_artifacts/`). Voir `ML/README.md`. |

**Variables d’environnement (ML / SQL) :** `EVENTZILLA_SQL_URI` — connexion à la base **`DW_Eventzilla`** restaurée depuis `FilesMachine/DB/DW_Eventzilla` (voir `ML/ml_paths.py`).

---

## Architecture technique & pipeline (E-LT)

1. **Sources :** **Staging Area (SA)** sur **SQL Server** ; fichiers **Excel/CSV** (benchmarks, marketing) dans `Data_Sources/`.  
2. **Stockage :** **Data Warehouse (DW)** en **schéma en étoile** pour les mesures DAX.  
3. **Visualisation :** Power BI Desktop & Service.  
4. **Middleware :** **On-premises Data Gateway** entre le DW local et le cloud.  

---

## Automatisation & Executive Summary

- **Rafraîchissement :** planification bi-quotidienne (**8h00 / 14h00**) via la Gateway.  
- **Executive Summary :** page dédiée avec **Smart Narrative** pour des résumés textuels automatiques des KPI.  

---

## Gouvernance & sécurité (RLS)

**Row-Level Security** avec filtrage dynamique **`USERPRINCIPALNAME()`** :

- **Rôle Marketing :** CAC, ROI campagnes, funnel commercial.  
- **Rôle Finance :** marges, commissions, rentabilité.  
- **Rôle relation client :** NPS, réclamations, taux de résolution.  

Validation via **« Voir en tant que »** dans Power BI Service.  

---

## Documentation DAX & génération des livrables

- Point d’entrée : **`docs/DAX_Measures.md`**  
- Régénération (depuis la **racine** du dépôt) :

```powershell
cd "chemin\vers\PI BI NEW"
pip install -r config/requirements.txt
python scripts/build_dashboards_table2_with_formulas.py
python scripts/build_dashboards_table2_dax_et_visuels.py
```

PDF du document objectifs (Pandoc) : `.\scripts\convert_EventZilla_doc_to_pdf.ps1`

---

## KPIs implémentés (aperçu)

Plus de **50 indicateurs** : performance commerciale (conversion, AOV), analyse financière (revenu, commissions), fidélité (rétention, NPS), etc. — détail dans `docs/eventzilla/EventZilla_Dashboards_KPIs_Objectifs.md` et `deliverables/`.  

---

*Projet académique — équipe **ctrlAltWin** — solution BI moderne, sécurisée et documentée pour Git.*

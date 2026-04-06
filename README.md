# 🚀 EventZilla BI - Marketplace Intelligence Solution

## 👥 Équipe : ctrlAltWin (Classe 4Bi5)
**Project Lead :** Ranim Chikhrouhou
**Institution :** Esprit (Spécialisation Business Intelligence)
**Année :** 2026

---

## 📌 Présentation du Projet
**EventZilla BI** est une solution décisionnelle de bout en bout conçue pour une marketplace événementielle tunisienne. Ce projet transforme des données brutes provenant de sources hétérogènes en un écosystème de pilotage stratégique automatisé, permettant de monitorer la croissance, la rentabilité et la satisfaction client.

## 🏗️ Architecture Technique & Pipeline (E-LT)
Pour garantir une robustesse de **Niveau A**, la solution repose sur un pipeline de données industriel :

1. **Sources de Données :** * **Staging Area (SA) :** SQL Server pour l'ingestion initiale.
   * **Fichiers Externes :** Excel/CSV pour les benchmarks de marché et données marketing.
2. **Stockage :** **Data Warehouse (DW)** modélisé en **schéma en étoile** (Star Schema) pour optimiser les performances des mesures DAX.
3. **Visualisation :** Power BI Desktop & Service.
4. **Middleware :** **On-premises Data Gateway** assurant la liaison sécurisée entre le DW local et le cloud.

## 🔄 Automatisation & Executive Synthesis
Ce projet remplit les critères d'excellence en matière d'automatisation :
* **Data Refresh Automation :** Planification bi-quotidienne (8h00 / 14h00) via la Gateway, garantissant des données toujours fraîches pour les décideurs.
* **Executive Synthesis :** Inclusion d'une page **"Executive Summary"** utilisant la **Narration Dynamique** (Smart Narrative). Ce système génère automatiquement des résumés textuels intelligents des performances (ex: CA, NPS, Taux de conversion) sans intervention manuelle.

## 🔐 Gouvernance & Sécurité (RLS)
La sécurité des données est gérée par **Row-Level Security (RLS)**, garantissant la confidentialité selon le profil utilisateur :
* [cite_start]**Rôle Marketing :** Visibilité sur le CAC (Coût d'Acquisition) et le ROI des campagnes[cite: 172, 182].
* [cite_start]**Rôle Finance :** Accès aux marges, commissions et analyses de rentabilité[cite: 22, 179].
* [cite_start]**Rôle Client :** Monitoring du NPS et du taux de résolution des plaintes[cite: 44, 68].
* **Logique :** Utilisation de `USERPRINCIPALNAME()` pour un filtrage dynamique et sécurisé.

## 📂 Structure du Dépôt (Git Deployment)
Le dépôt est organisé pour une reproductibilité et une maintenance logicielle complète :
* 📦 `/Reports` : Fichier source `.pbix` finalisé.
* 📦 `/Database` : Backups `.bak` de la **SA** et du **DW** pour la restauration système.
* 📦 `/Data_Sources` : Datasets Excel sources.
* [cite_start]📄 `DAX_Measures.md` : Documentation technique exhaustive des formules (KPIs) [cite: 1-182].
* 📄 `README.md` : Documentation générale du projet.

## 📊 Indicateurs Clés (KPIs) Implémentés
Le système monitor plus de 50 indicateurs, notamment :
* [cite_start]**Performance Commerciale :** Taux de conversion, AOV (Average Order Value)[cite: 11, 53].
* [cite_start]**Analyse Financière :** Revenu Total, Commissions nettes[cite: 29, 181].
* [cite_start]**Loyauté :** Taux de rétention des bénéficiaires et NPS[cite: 14, 68].

---
*Ce projet démontre la capacité de l'équipe **ctrlAltWin** à livrer une solution BI moderne, sécurisée et entièrement automatisée.*
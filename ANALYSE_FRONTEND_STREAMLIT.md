# 📊 ANALYSE COMPLÈTE - Frontend Streamlit EventZilla

## 🎯 VUE D'ENSEMBLE

**Fichier principal:** `ML/streamlit_app.py` (3,800+ lignes)  
**Framework:** Streamlit  
**Architecture:** Single-page application avec navigation par sidebar  
**Authentification:** SQL Server (table AppUsers)  
**Rôles:** 3 rôles avec pages spécifiques

---

## 📄 STRUCTURE DES PAGES

### Pages disponibles (6 au total):

| Page | Constante | Fonction | Description |
|------|-----------|----------|-------------|
| **Home** | `PAGE_HOME` | `page_home()` | Dashboard KPI + navigation |
| **Booking Status** | `PAGE_CLASSIF` | `page_classification()` | Prédiction statut réservation |
| **Price Estimation** | `PAGE_REGR` | `page_regression()` | Estimation prix final |
| **Customer Segments** | `PAGE_CLUSTER` | `page_clustering()` | Segmentation clients |
| **Trends & Forecast** | `PAGE_TS` | `page_timeseries()` | Séries temporelles |
| **Summary** | `PAGE_RECAP` | `page_recap()` | Récapitulatif synthétique |

### Ordre de navigation:
```python
PAGE_ORDER = (
    PAGE_HOME,      # Home
    PAGE_CLASSIF,   # Booking Status
    PAGE_REGR,      # Price Estimation
    PAGE_CLUSTER,   # Customer Segments
    PAGE_TS,        # Trends & Forecast
    PAGE_RECAP,     # Summary
)
```

---

## 👥 CONTRÔLE D'ACCÈS PAR RÔLE

### Rôle: Marketing Manager (Ranim)
```python
"marketing_manager": (PAGE_HOME, PAGE_CLUSTER, PAGE_CLASSIF)
```
- ✅ Home
- ✅ Customer Segments (segmentation clients)
- ✅ Booking Status (prédiction annulation)
- ❌ Price Estimation
- ❌ Trends & Forecast
- ❌ Summary

### Rôle: Financial Manager (Naïma)
```python
"financial_manager": (PAGE_HOME, PAGE_REGR, PAGE_TS)
```
- ✅ Home
- ✅ Price Estimation (régression prix)
- ✅ Trends & Forecast (séries temporelles)
- ❌ Customer Segments
- ❌ Booking Status
- ❌ Summary

### Rôle: CRM Manager (Anas)
```python
"crm_manager": (PAGE_HOME, PAGE_CLASSIF, PAGE_CLUSTER)
```
- ✅ Home
- ✅ Booking Status (prédiction annulation)
- ✅ Customer Segments (segmentation clients)
- ❌ Price Estimation
- ❌ Trends & Forecast
- ❌ Summary

---

## 🎨 SYSTÈME DE COULEURS

### Palette de marque:
```python
BRAND = {
    "deep":   "#0f172a",  # Bleu très foncé
    "main":   "#1e293b",  # Bleu foncé
    "accent": "#6366f1",  # Indigo
    "sky":    "#0ea5e9",  # Cyan
    "soft":   "#f1f5f9",  # Gris clair
}
```

### Couleurs par page:
```python
PAGE_ACCENT = {
    "synth":   ("#1e40af", "#3b82f6", "#eff6ff"),  # Bleu (Home/Summary)
    "classif": ("#047857", "#10b981", "#ecfdf5"),  # Vert (Classification)
    "regr":    ("#6d28d9", "#8b5cf6", "#f5f3ff"),  # Violet (Régression)
    "cluster": ("#c2410c", "#ea580c", "#fff7ed"),  # Orange (Clustering)
    "ts":      ("#b45309", "#f59e0b", "#fffbeb"),  # Ambre (Time Series)
}
```

### Couleurs de navigation:
```python
NAV_COLORS = {
    PAGE_HOME:    "#6366f1",  # Indigo
    PAGE_CLASSIF: "#10b981",  # Vert
    PAGE_REGR:    "#8b5cf6",  # Violet
    PAGE_CLUSTER: "#ea580c",  # Orange
    PAGE_TS:      "#f59e0b",  # Ambre
}
```

---

## 🏗️ COMPOSANTS PRINCIPAUX

### 1. Authentification
```python
def _render_login_screen()
def is_authenticated(session_state)
def get_role(session_state)
```

**Fonctionnalités:**
- Login via SQL Server (table AppUsers)
- Validation email/password
- Stockage session Streamlit
- Redirection automatique après login

### 2. Navigation
```python
def sidebar_brand_and_nav() -> str
def goto_page(label: str)
def _page_nav_footer(current: str)
```

**Fonctionnalités:**
- Sidebar avec logo EventZilla
- Filtrage des pages par rôle
- Boutons de navigation colorés
- Footer avec Précédent/Suivant

### 3. Hero Sections
```python
def hero_variant(variant, title, subtitle, badges)
```

**Variantes:**
- `synth` - Bleu (Home/Summary)
- `classif` - Vert (Classification)
- `regr` - Violet (Régression)
- `cluster` - Orange (Clustering)
- `ts` - Ambre (Time Series)

### 4. Cartes KPI
```python
def _kpi_card_html(value, label, color)
def _ml_model_card_html(badge, title, models, color)
```

**Styles:**
- Gradient backgrounds
- Couleurs dynamiques
- Animations hover
- Responsive design

---

## 📊 PAGE HOME (Dashboard)

### Sections:

#### 1. Business Analytics (10 KPIs)
**Ligne 1:**
- Total Réservations
- Revenue (TND)
- Valeur Commande Moy.
- Taux Annulation
- Segments Clients

**Ligne 2:**
- Balance Score (Classification)
- R² Régression
- Quality Score (Clustering)
- Prediction Error (Time Series)
- Horizon Prévision

#### 2. Modèles ML déployés (4 cartes)
- Status Prediction (RF + LR)
- Régression (Ridge + RF)
- Customer Grouping (K-Means + HC)
- Séries temporelles (Holt + ARIMA)

#### 3. Navigation rapide (5 boutons)
- Status Prediction
- Régression
- Customer Grouping
- Séries temporelles
- Voir le récapitulatif

---

## 🔮 PAGE CLASSIFICATION (Booking Status)

### Fonctionnalités:

#### 1. Formulaire de prédiction
**12 champs:**
- id_date, id_event, id_servicecategory
- id_benchmark, id_provider
- final_price, service_price, benchmark_avg_price
- event_budget
- cal_month, cal_year, quarter

#### 2. Résultat de prédiction
- Statut prédit (cancelled/confirmed/pending)
- Probabilités par classe
- Graphique en barres
- Interprétation métier

#### 3. Métriques du modèle
- Accuracy, Precision, Recall, F1-Score
- Matrice de confusion
- Courbe ROC (si disponible)

---

## 💰 PAGE REGRESSION (Price Estimation)

### Fonctionnalités:

#### 1. Formulaire de prédiction
**12 champs:**
- id_date, id_event, id_servicecategory
- id_benchmark, id_provider
- service_price, benchmark_avg_price
- event_budget
- cal_month, cal_year, quarter
- commission_margin

#### 2. Résultat de prédiction
- Montant prédit (TND)
- Intervalle de confiance
- Comparaison avec benchmark
- Graphique de distribution

#### 3. Métriques du modèle
- R², RMSE, MAE
- Graphique résidus
- Feature importance

---

## 👥 PAGE CLUSTERING (Customer Segments)

### Fonctionnalités:

#### 1. Sélection du type
- Bénéficiaires (clients)
- Prestataires (providers)

#### 2. Formulaire RFM
**6 métriques:**
- nb_reservations_loyalty
- ca_total_loyalty
- panier_moyen_loyalty
- recency_days_loyalty
- avg_nb_visitors_loyalty
- volume_reservations_site_loyalty

#### 3. Résultat de segmentation
- Segment ID
- Label métier (VIP/Fidèle/Occasionnel/À risque)
- Caractéristiques du segment
- Recommandations

#### 4. Visualisations
- Distribution des segments
- Profils RFM
- Graphiques 2D/3D

---

## 📈 PAGE TIME SERIES (Trends & Forecast)

### Fonctionnalités:

#### 1. Sélection de la série
- Nombre de réservations
- Chiffre d'affaires
- Panier moyen
- Autres KPIs

#### 2. Paramètres de prévision
- Horizon (mois)
- Taille du holdout
- Modèle (Holt/ARIMA)

#### 3. Visualisations
- Graphique historique
- Train/Test/Forecast
- Intervalles de confiance
- Tendances

#### 4. Métriques
- RMSE, MAE, MAPE
- Comparaison Holt vs ARIMA
- Tableau des prévisions

---

## 📋 PAGE SUMMARY (Récapitulatif)

### Sections:

#### 1. Vue d'ensemble
- Résumé des 4 familles de modèles
- Métriques clés
- Statut des modèles

#### 2. Cartes par famille
- Classification
- Régression
- Clustering
- Time Series

#### 3. Recommandations
- Actions suggérées
- Prochaines étapes

---

## 🔧 FONCTIONS UTILITAIRES

### Chargement de données:
```python
def load_json(path)
def load_model(path)
def extract_classification_metrics(mc)
def extract_regression_metrics(mr)
```

### Visualisations:
```python
def _plotly_layout(**kwargs)
def _plotly_x_datetime(ts)
def _kpi_card_html(value, label, color)
def _ml_model_card_html(badge, title, models, color)
```

### UI Components:
```python
def section_header(title, subtitle)
def hero_variant(variant, title, subtitle, badges)
def _inject_page_accent(deep, main, soft)
```

---

## 📁 DÉPENDANCES EXTERNES

### Fichiers de configuration:
- `ML/ml_paths.py` - Chemins des modèles
- `ML/schema_eventzilla.py` - Requêtes SQL
- `.env` - Variables d'environnement

### Fichiers de modèles:
- `ML/models_artifacts/*.joblib` - Modèles ML
- `ML/models_artifacts/metrics_*.json` - Métriques

### Assets:
- `ML/assets/logo_eventzilla.png` - Logo
- `ML/assets/logo_eventzilla.svg` - Logo SVG

---

## 🎨 STYLES CSS PERSONNALISÉS

### Thèmes injectés:
- Boutons avec couleurs dynamiques
- Cartes KPI avec gradients
- Métriques avec accents colorés
- Expanders stylisés
- Formulaires avec bordures colorées

### Exemple d'injection:
```python
def _inject_page_accent(deep: str, main: str, soft: str):
    st.markdown(f"""
    <style>
        .stButton>button {{
            background: linear-gradient(135deg, {deep} 0%, {main} 100%);
            color: white;
            border: none;
            border-radius: 8px;
        }}
        .stMetric {{
            background: {soft};
            border-left: 4px solid {main};
        }}
    </style>
    """, unsafe_allow_html=True)
```

---

## 🔐 SÉCURITÉ

### Contrôles implémentés:
1. **Authentification obligatoire** - Toutes les pages
2. **Filtrage par rôle** - Pages spécifiques par utilisateur
3. **Validation des inputs** - Formulaires
4. **Session management** - Streamlit session_state
5. **SQL injection protection** - Requêtes paramétrées

### Vérifications:
```python
# Vérification authentification
if not is_authenticated(st.session_state):
    _render_login_screen()
    st.stop()

# Vérification rôle
role = get_role(st.session_state)
allowed_pages = ROLE_PAGES.get(role, PAGE_ORDER)
if page not in allowed_pages:
    st.warning("⛔ Page non accessible")
    st.stop()
```

---

## 📊 MÉTRIQUES ET MODÈLES

### Fichiers JSON chargés:
1. `metrics_classification.json` - Métriques RF/LR
2. `metrics_regression.json` - Métriques Ridge/RF
3. `metrics_clustering.json` - Métriques K-Means
4. `metrics_timeseries.json` - Métriques Holt/ARIMA

### Modèles .joblib chargés:
1. `classification_status_champion_pipeline.joblib`
2. `ridge_regression_primary.joblib`
3. `kmeans_loyalty_beneficiary.joblib`
4. `kmeans_loyalty_provider.joblib`
5. Scalers et encoders associés

---

## 🚀 POINTS D'AMÉLIORATION POSSIBLES

### UX/UI:
- [ ] Ajouter des tooltips explicatifs
- [ ] Améliorer la responsive mobile
- [ ] Ajouter des animations de chargement
- [ ] Créer un mode sombre/clair

### Fonctionnalités:
- [ ] Export des résultats en PDF
- [ ] Historique des prédictions
- [ ] Comparaison de scénarios
- [ ] Notifications en temps réel

### Performance:
- [ ] Cache des modèles ML
- [ ] Lazy loading des graphiques
- [ ] Optimisation des requêtes SQL
- [ ] Compression des assets

### Sécurité:
- [ ] 2FA (authentification à deux facteurs)
- [ ] Logs d'audit
- [ ] Rate limiting
- [ ] HTTPS obligatoire

---

## 📝 RÉSUMÉ TECHNIQUE

| Aspect | Détails |
|--------|---------|
| **Lignes de code** | ~3,800 lignes |
| **Pages** | 6 pages principales |
| **Rôles** | 3 rôles utilisateurs |
| **Modèles ML** | 4 familles (12+ modèles) |
| **Graphiques** | Plotly interactive |
| **Base de données** | SQL Server |
| **Authentification** | SQL Server (AppUsers) |
| **Framework** | Streamlit 1.x |
| **Python** | 3.11+ |

---

## 🎯 ARCHITECTURE VISUELLE

```
┌─────────────────────────────────────────────────────────┐
│                  STREAMLIT APP                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐      ┌──────────────┐               │
│  │   Sidebar    │      │  Main Area   │               │
│  │              │      │              │               │
│  │  - Logo      │      │  - Hero      │               │
│  │  - Nav       │      │  - Content   │               │
│  │  - Logout    │      │  - Footer    │               │
│  └──────────────┘      └──────────────┘               │
│         │                      │                       │
│         ▼                      ▼                       │
│  ┌──────────────┐      ┌──────────────┐               │
│  │  ROLE_PAGES  │      │  Page Func   │               │
│  │  (filter)    │      │  - home()    │               │
│  │              │      │  - classif() │               │
│  └──────────────┘      │  - regr()    │               │
│                        │  - cluster() │               │
│                        │  - ts()      │               │
│                        │  - recap()   │               │
│                        └──────────────┘               │
│                               │                        │
│                               ▼                        │
│                        ┌──────────────┐               │
│                        │  ML Models   │               │
│                        │  + Metrics   │               │
│                        └──────────────┘               │
└─────────────────────────────────────────────────────────┘
```

---

**Prêt pour les modifications! Dites-moi ce que vous voulez changer.** 🚀

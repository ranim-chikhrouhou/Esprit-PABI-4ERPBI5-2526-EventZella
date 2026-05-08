# Arbres de décision — Version française (5 DESCRIPTIF, 2 EXPLICATIF, 1 PRÉDICTIF)

Structure : **Objectif global** → **5 DESCRIPTIF** / **2 EXPLICATIF** / **1 PRÉDICTIF** → Objectifs opérationnels (verbes) → KPI.  
Copier un bloc dans https://mermaid.live (de `graph TB` à la dernière ligne `class`).  
Couleurs : Global = bleu ciel, Descriptif = rose, Explicatif = violet, Prédictif = vert.

---

## BLOC 1 — Responsable Marketing

graph TB
  G[Objectif Global - Optimiser la performance commerciale]
  G --> D1
  G --> D2
  G --> D3
  G --> D4
  G --> D5
  G --> E1
  G --> E2
  G --> PRED
  D1[DESCRIPTIF 1 - Décrire l état actuel de la performance commerciale]
  D1 --> D1a[Visualiser le nombre total de réservations]
  D1a --> D1a1[KPI Nombre total de réservations]
  D1 --> D1b[Visualiser le taux de conversion]
  D1b --> D1b1[KPI Taux de conversion]
  D1 --> D1c[Visualiser le nombre de visiteurs]
  D1c --> D1c1[KPI Nombre de visiteurs]
  D2[DESCRIPTIF 2 - Décrire la diversité des catégories réservées]
  D2 --> D2a[Visualiser la diversité des catégories réservées]
  D2a --> D2a1[KPI Diversité des catégories réservées]
  D3[DESCRIPTIF 3 - Décrire les top N catégories à ajouter]
  D3 --> D3a[Identifier les top N catégories à ajouter]
  D3a --> D3a1[KPI Top N catégories à ajouter]
  D4[DESCRIPTIF 4 - Décrire la répartition géographique]
  D4 --> D4a[Visualiser la répartition géographique]
  D4a --> D4a1[KPI Répartition des salles par gouvernorat]
  D5[DESCRIPTIF 5 - Décrire la répartition des catégories de services]
  D5 --> D5a[Analyser la diversité, les top N et la répartition]
  D5a --> D5a1[KPI Diversité Top N Répartition gouvernorat]
  E1[EXPLICATIF 1 - Expliquer la conversion, les campagnes, la valeur client et la rétention]
  E1 --> E1a[Analyser la conversion selon taux d acceptation, annulation et visiteurs]
  E1a --> E1a1[KPI Taux de conversion Taux d acceptation Nombre de visiteurs]
  E1 --> E1b[Analyser les campagnes marketing : CAC, réservations, conversion, visiteurs]
  E1b --> E1b1[KPI CAC Nombre total de réservations Taux de conversion Nombre de visiteurs]
  E1 --> E1c[Comprendre la relation entre LTV CAC rétention et valeur client]
  E1c --> E1c1[KPI LTV CAC Taux de rétention bénéficiaires Nombre total de réservations]
  E1 --> E1d[Analyser la rétention selon annulation et catégories nouvelles vs couvertes]
  E1d --> E1d1[KPI Taux de rétention bénéficiaires Taux d annulation Catégories nouvelles vs couvertes]
  E2[EXPLICATIF 2 - Expliquer le positionnement tarifaire, la croissance, les jours fériés et la couverture géographique]
  E2 --> E2a[Analyser les réservations sous alignées au dessus du marché]
  E2a --> E2a1[KPI Part réservations sous marché Part réservations alignées marché Part réservations au dessus marché Taux de conversion]
  E2 --> E2b[Analyser les opportunités de croissance par top N catégories et diversité]
  E2b --> E2b1[KPI Top N catégories à ajouter Catégories nouvelles vs couvertes Diversité des catégories réservées]
  E2 --> E2c[Analyser le comportement des réservations les jours fériés]
  E2c --> E2c1[KPI Taux de réservation les jours fériés Nombre total de réservations Taux de conversion]
  E2 --> E2d[Analyser la couverture géographique et son impact sur les réservations]
  E2d --> E2d1[KPI Répartition des salles par gouvernorat Nombre de salles suggérables Nombre total de réservations Taux de conversion]
  PRED[PRÉDICTIF - Prédire l évolution future de la performance commerciale]
  PRED --> P1[Prédire le nombre de réservations futures]
  P1 --> P1k[KPI Nombre total de réservations projection]
  PRED --> P2[Prédire le taux de conversion futur]
  P2 --> P2k[KPI Taux de conversion projection]
  PRED --> P3[Prédire l évolution du taux de rétention]
  P3 --> P3k[KPI Taux de rétention bénéficiaires projection]
  PRED --> P4[Prédire l évolution du CAC et du LTV]
  P4 --> P4k[KPI CAC projection LTV projection]
  PRED --> P5[Prédire le taux de réservation les jours fériés]
  P5 --> P5k[KPI Taux de réservation les jours fériés projection]
  classDef global fill:#87CEEB,stroke:#4682B4
  classDef descriptive fill:#FFB6C1,stroke:#C2185B
  classDef explanatory fill:#E1BEE7,stroke:#7B1FA2
  classDef predictive fill:#C8E6C9,stroke:#2E7D32
  class G global
  class D1,D2,D3,D4,D5,D1a,D1b,D1c,D1a1,D1b1,D1c1,D2a,D2a1,D3a,D3a1,D4a,D4a1,D5a,D5a1 descriptive
  class E1,E2,E1a,E1b,E1c,E1d,E1a1,E1b1,E1c1,E1d1,E2a,E2b,E2c,E2d,E2a1,E2b1,E2c1,E2d1 explanatory
  class PRED,P1,P2,P3,P4,P5,P1k,P2k,P3k,P4k,P5k predictive

---

## BLOC 2 — Responsable Financier

graph TB
  G[Objectif Global - Optimiser la rentabilité]
  G --> D1
  G --> D2
  G --> D3
  G --> D4
  G --> D5
  G --> E1
  G --> E2
  G --> PRED
  D1[DESCRIPTIF 1 - Décrire l état actuel des revenus et de la performance financière]
  D1 --> D1a[Visualiser le chiffre d affaires total]
  D1a --> D1a1[KPI Chiffre d affaires total]
  D1 --> D1b[Visualiser le panier moyen]
  D1b --> D1b1[KPI Panier moyen]
  D2[DESCRIPTIF 2 - Décrire les top N catégories à ajouter]
  D2 --> D2a[Identifier les top N catégories à ajouter]
  D2a --> D2a1[KPI Top N catégories à ajouter]
  D3[DESCRIPTIF 3 - Décrire la structure des revenus et des commissions]
  D3 --> D3a[Visualiser le taux de commission sur réservation]
  D3a --> D3a1[KPI Taux de commission sur réservation]
  D3 --> D3b[Comparer les revenus par période]
  D3b --> D3b1[KPI Chiffre d affaires total Panier moyen]
  D4[DESCRIPTIF 4 - Décrire l impact des jours fériés sur le chiffre d affaires]
  D4 --> D4a[Visualiser l impact des jours fériés sur le CA]
  D4a --> D4a1[KPI Impact jours fériés sur CA]
  D5[DESCRIPTIF 5 - Décrire les indicateurs de revenus et de commissions]
  D5 --> D5a[Analyser les revenus par période et les commissions]
  D5a --> D5a1[KPI Chiffre d affaires Panier moyen Commissions]
  E1[EXPLICATIF 1 - Expliquer la rentabilité, le CAC et la valeur client]
  E1 --> E1a[Analyser la rentabilité via LTV CAC panier et catégories]
  E1a --> E1a1[KPI LTV CAC Panier Chiffre d affaires Catégories nouvelles vs couvertes]
  E1 --> E1b[Analyser le CAC par rapport au chiffre d affaires au taux de conversion et au panier]
  E1b --> E1b1[KPI CAC Chiffre d affaires Taux de conversion Panier moyen]
  E1 --> E1c[Analyser l impact du LTV sur le chiffre d affaires le panier et les catégories]
  E1c --> E1c1[KPI LTV Chiffre d affaires Panier moyen Catégories nouvelles vs couvertes]
  E2[EXPLICATIF 2 - Expliquer le positionnement tarifaire les jours fériés les commissions et la croissance]
  E2 --> E2a[Analyser les réservations sous alignées au dessus du marché vs CA et panier]
  E2a --> E2a1[KPI Part réservations sous marché Part alignées marché Part au dessus marché CA Panier]
  E2 --> E2b[Analyser l impact des jours fériés sur le CA et le panier]
  E2b --> E2b1[KPI Impact jours fériés sur CA Panier Revenus]
  E2 --> E2c[Analyser le taux de commission vs revenus panier et catégories]
  E2c --> E2c1[KPI Taux de commission Chiffre d affaires Panier Catégories nouvelles vs couvertes]
  E2 --> E2d[Analyser la croissance via top N catégories et rentabilité]
  E2d --> E2d1[KPI Top N catégories à ajouter Catégories couvertes LTV CAC]
  PRED[PRÉDICTIF - Prédire l évolution future des revenus et de la rentabilité]
  PRED --> P1[Prédire l évolution du chiffre d affaires]
  P1 --> P1k[KPI Chiffre d affaires total projection]
  PRED --> P2[Prédire l évolution du panier moyen]
  P2 --> P2k[KPI Panier moyen projection]
  PRED --> P3[Prédire l évolution du LTV et du CAC]
  P3 --> P3k[KPI LTV CAC projection]
  PRED --> P4[Prédire l impact des jours fériés futurs sur le CA]
  P4 --> P4k[KPI Impact jours fériés sur CA projection]
  classDef global fill:#87CEEB,stroke:#4682B4
  classDef descriptive fill:#FFB6C1,stroke:#C2185B
  classDef explanatory fill:#E1BEE7,stroke:#7B1FA2
  classDef predictive fill:#C8E6C9,stroke:#2E7D32
  class G global
  class D1,D2,D3,D4,D5,D1a,D1b,D1a1,D1b1,D2a,D2a1,D3a,D3b,D3a1,D3b1,D4a,D4a1,D5a,D5a1 descriptive
  class E1,E2,E1a,E1b,E1c,E1a1,E1b1,E1c1,E2a,E2b,E2c,E2d,E2a1,E2b1,E2c1,E2d1 explanatory
  class PRED,P1,P2,P3,P4,P1k,P2k,P3k,P4k predictive

---

## BLOC 3 — Responsable Relation Client

graph TB
  G[Objectif Global - Améliorer la relation client]
  G --> D1
  G --> D2
  G --> D3
  G --> D4
  G --> D5
  G --> E1
  G --> E2
  G --> PRED
  D1[DESCRIPTIF 1 - Décrire l état actuel de la satisfaction et de la relation client]
  D1 --> D1a[Visualiser le nombre de réclamations]
  D1a --> D1a1[KPI Nombre de réclamations]
  D1 --> D1b[Visualiser le taux d annulation]
  D1b --> D1b1[KPI Taux d annulation]
  D2[DESCRIPTIF 2 - Décrire la note moyenne et le taux de réclamations]
  D2 --> D2a[Comparer la note moyenne prestataires]
  D2a --> D2a1[KPI Note moyenne prestataires]
  D2 --> D2b[Visualiser le taux de réclamations pour 100 réservations]
  D2b --> D2b1[KPI Taux de réclamations pour 100 réservations]
  D3[DESCRIPTIF 3 - Décrire la qualité du service et la réactivité]
  D3 --> D3a[Visualiser le taux de résolution des réclamations]
  D3a --> D3a1[KPI Taux de résolution réclamations]
  D3 --> D3b[Comparer le NPS]
  D3b --> D3b1[KPI NPS]
  D4[DESCRIPTIF 4 - Décrire la part des salles joignables]
  D4 --> D4a[Visualiser la part des salles joignables]
  D4a --> D4a1[KPI Part des salles joignables]
  D5[DESCRIPTIF 5 - Décrire les indicateurs de satisfaction et de service]
  D5 --> D5a[Analyser les réclamations la résolution le NPS et l accessibilité]
  D5a --> D5a1[KPI Réclamations Résolution NPS Salles joignables]
  E1[EXPLICATIF 1 - Expliquer la satisfaction les réclamations la rétention et la qualité de service]
  E1 --> E1a[Analyser la satisfaction via note prestataires annulation et réclamations pour 100]
  E1a --> E1a1[KPI Note moyenne prestataires Réclamations Annulation Réclamations 100 Note]
  E1 --> E1b[Analyser les réclamations vs annulation note et résolution]
  E1b --> E1b1[KPI Réclamations Annulation Note Résolution]
  E1 --> E1c[Analyser la rétention vs réclamations pour 100 et annulation]
  E1c --> E1c1[KPI Réclamations 100 Rétention Annulation Réclamations]
  E1 --> E1d[Analyser la qualité de service via résolution réclamations et salles joignables]
  E1d --> E1d1[KPI Résolution Réclamations Réclamations 100 Salles joignables]
  E2[EXPLICATIF 2 - Expliquer le positionnement tarifaire le NPS et l accessibilité]
  E2 --> E2a[Analyser les réservations sous alignées au dessus marché vs réclamations et annulation]
  E2a --> E2a1[KPI Part réservations sous marché Part alignées au dessus Réclamations Annulation]
  E2 --> E2b[Analyser le NPS vs réclamations rétention et note]
  E2b --> E2b1[KPI Réclamations Rétention Note NPS]
  E2 --> E2c[Analyser l accessibilité via salles joignables et indicateurs de réclamations]
  E2c --> E2c1[KPI Salles joignables Réclamations 100 Réclamations Annulation]
  PRED[PRÉDICTIF - Prédire l évolution future de la satisfaction et de la relation client]
  PRED --> P1[Prédire l évolution du taux de réclamations pour 100 réservations]
  P1 --> P1k[KPI Taux de réclamations pour 100 réservations projection]
  PRED --> P2[Prédire l évolution du nombre de réclamations]
  P2 --> P2k[KPI Nombre de réclamations projection]
  PRED --> P3[Prédire l évolution du taux de rétention]
  P3 --> P3k[KPI Taux de rétention bénéficiaires projection]
  PRED --> P4[Prédire l évolution du NPS]
  P4 --> P4k[KPI NPS projection]
  classDef global fill:#87CEEB,stroke:#4682B4
  classDef descriptive fill:#FFB6C1,stroke:#C2185B
  classDef explanatory fill:#E1BEE7,stroke:#7B1FA2
  classDef predictive fill:#C8E6C9,stroke:#2E7D32
  class G global
  class D1,D2,D3,D4,D5,D1a,D1b,D1a1,D1b1,D2a,D2b,D2a1,D2b1,D3a,D3b,D3a1,D3b1,D4a,D4a1,D5a,D5a1 descriptive
  class E1,E2,E1a,E1b,E1c,E1d,E1a1,E1b1,E1c1,E1d1,E2a,E2b,E2c,E2a1,E2b1,E2c1 explanatory
  class PRED,P1,P2,P3,P4,P1k,P2k,P3k,P4k predictive

# EventZilla — Synthèse objectifs, KPIs et dashboards Power BI

**Projet :** BI intégrée EventZilla.  
**Sources :** `Trees/ArbreDecision_Mermaid_FR.md` (niveaux d’objectifs et KPIs par feuille d’arbre), `FilesPdf/KPIs_FINAL.pdf` (formules).  
**Note :** Dans `KPIs_FINAL.pdf`, la formule du **taux de commission** est tronquée ; la ligne du tableau reprend le texte disponible et un complément indicatif entre crochets.

**Structure du document**

1. **Tableau 1** — Objectifs (global → GIMSI → opérationnel), KPI associé et formule.  
2. **Tableau 2** — Au plus **3 dashboards par décideur** ; **une ligne = un KPI** (ou une information clé) avec le **visuel Power BI** correspondant sur la même ligne.

---

## Tableau 1 — Chaîne objectifs → KPIs → formules

**Colonnes :** (1) Objectif global — (2) Deuxième niveau (famille DESCRIPTIF / EXPLICATIF / PRÉDICTIF + libellé) — (3) Objectif opérationnel — (4) KPI — (5) Formule (*KPIs_FINAL.pdf*, notation agrégée type SQL / BI).

*Une ligne = un KPI ; lorsqu’un objectif opérationnel porte plusieurs KPIs, les lignes (1)–(3) sont répétées.*

| Objectif global | 2ᵉ niveau — objectif (famille GIMSI) | Objectif opérationnel | KPI | Formule |
|-----------------|--------------------------------------|------------------------|-----|---------|
| Optimiser la performance commerciale | DESCRIPTIF 1 — Décrire l’état actuel de la performance commerciale | Visualiser le nombre total de réservations | Nombre total de réservations | `COUNT(id_reservation)` |
| Optimiser la performance commerciale | DESCRIPTIF 1 — Décrire l’état actuel de la performance commerciale | Visualiser le taux de conversion | Taux de conversion | `(COUNT(reservations_confirmées) / SUM(visitors)) × 100` |
| Optimiser la performance commerciale | DESCRIPTIF 1 — Décrire l’état actuel de la performance commerciale | Visualiser le nombre de visiteurs | Nombre de visiteurs | `SUM(visitors)` |
| Optimiser la performance commerciale | DESCRIPTIF 2 — Décrire la diversité des catégories réservées | Visualiser la diversité des catégories réservées | Diversité des catégories réservées | `(COUNT(DISTINCT category_id) / COUNT(total_categories)) × 100` |
| Optimiser la performance commerciale | DESCRIPTIF 3 — Décrire les top N catégories à ajouter | Identifier les top N catégories à ajouter | Top N catégories à ajouter | `TOP N(event_count_observed)` hors catégories couvertes |
| Optimiser la performance commerciale | DESCRIPTIF 4 — Décrire la répartition géographique | Visualiser la répartition géographique | Répartition des salles par gouvernorat | `COUNT(venue_name) GROUP BY gouvernorat` |
| Optimiser la performance commerciale | DESCRIPTIF 5 — Décrire la répartition des catégories de services | Analyser la diversité, les top N et la répartition | Diversité des catégories réservées | `(COUNT(DISTINCT category_id) / COUNT(total_categories)) × 100` |
| Optimiser la performance commerciale | DESCRIPTIF 5 — Décrire la répartition des catégories de services | Analyser la diversité, les top N et la répartition | Top N catégories à ajouter | `TOP N(event_count_observed)` hors catégories couvertes |
| Optimiser la performance commerciale | DESCRIPTIF 5 — Décrire la répartition des catégories de services | Analyser la diversité, les top N et la répartition | Répartition des salles par gouvernorat | `COUNT(venue_name) GROUP BY gouvernorat` |
| Optimiser la performance commerciale | EXPLICATIF 1 — Expliquer la conversion, les campagnes, la valeur client et la rétention | Analyser la conversion selon taux d’acceptation, annulation et visiteurs | Taux de conversion | `(COUNT(reservations_confirmées) / SUM(visitors)) × 100` |
| Optimiser la performance commerciale | EXPLICATIF 1 — Expliquer la conversion, les campagnes, la valeur client et la rétention | Analyser la conversion selon taux d’acceptation, annulation et visiteurs | Taux d’acceptation | `(COUNT(reservations_acceptées) / COUNT(total_reservations)) × 100` |
| Optimiser la performance commerciale | EXPLICATIF 1 — Expliquer la conversion, les campagnes, la valeur client et la rétention | Analyser la conversion selon taux d’acceptation, annulation et visiteurs | Nombre de visiteurs | `SUM(visitors)` |
| Optimiser la performance commerciale | EXPLICATIF 1 — Expliquer la conversion, les campagnes, la valeur client et la rétention | Analyser les campagnes marketing : CAC, réservations, conversion, visiteurs | CAC (coût d’acquisition client) | `SUM(marketing_spend) / SUM(new_beneficiaries)` |
| Optimiser la performance commerciale | EXPLICATIF 1 — Expliquer la conversion, les campagnes, la valeur client et la rétention | Analyser les campagnes marketing : CAC, réservations, conversion, visiteurs | Nombre total de réservations | `COUNT(id_reservation)` |
| Optimiser la performance commerciale | EXPLICATIF 1 — Expliquer la conversion, les campagnes, la valeur client et la rétention | Analyser les campagnes marketing : CAC, réservations, conversion, visiteurs | Taux de conversion | `(COUNT(reservations_confirmées) / SUM(visitors)) × 100` |
| Optimiser la performance commerciale | EXPLICATIF 1 — Expliquer la conversion, les campagnes, la valeur client et la rétention | Analyser les campagnes marketing : CAC, réservations, conversion, visiteurs | Nombre de visiteurs | `SUM(visitors)` |
| Optimiser la performance commerciale | EXPLICATIF 1 — Expliquer la conversion, les campagnes, la valeur client et la rétention | Comprendre la relation entre LTV, CAC, rétention et valeur client | LTV (lifetime value) | `Panier moyen × Fréquence × Durée rétention` |
| Optimiser la performance commerciale | EXPLICATIF 1 — Expliquer la conversion, les campagnes, la valeur client et la rétention | Comprendre la relation entre LTV, CAC, rétention et valeur client | CAC (coût d’acquisition client) | `SUM(marketing_spend) / SUM(new_beneficiaries)` |
| Optimiser la performance commerciale | EXPLICATIF 1 — Expliquer la conversion, les campagnes, la valeur client et la rétention | Comprendre la relation entre LTV, CAC, rétention et valeur client | Taux de rétention bénéficiaires | `(COUNT(beneficiaries_recurrents) / COUNT(total_beneficiaries)) × 100` |
| Optimiser la performance commerciale | EXPLICATIF 1 — Expliquer la conversion, les campagnes, la valeur client et la rétention | Comprendre la relation entre LTV, CAC, rétention et valeur client | Nombre total de réservations | `COUNT(id_reservation)` |
| Optimiser la performance commerciale | EXPLICATIF 1 — Expliquer la conversion, les campagnes, la valeur client et la rétention | Analyser la rétention selon annulation et catégories nouvelles vs couvertes | Taux de rétention bénéficiaires | `(COUNT(beneficiaries_recurrents) / COUNT(total_beneficiaries)) × 100` |
| Optimiser la performance commerciale | EXPLICATIF 1 — Expliquer la conversion, les campagnes, la valeur client et la rétention | Analyser la rétention selon annulation et catégories nouvelles vs couvertes | Taux d’annulation | `(COUNT(reservations_annulées) / COUNT(total_reservations)) × 100` |
| Optimiser la performance commerciale | EXPLICATIF 1 — Expliquer la conversion, les campagnes, la valeur client et la rétention | Analyser la rétention selon annulation et catégories nouvelles vs couvertes | Catégories nouvelles vs couvertes | `SET_DIFF(catégories marché, catégories EventZella)` |
| Optimiser la performance commerciale | EXPLICATIF 2 — Expliquer le positionnement tarifaire, la croissance, les jours fériés et la couverture géographique | Analyser les réservations sous, alignées, au-dessus du marché | Part réservations sous marché | `(COUNT(final_price < 0.85*benchmark) / COUNT(total)) × 100` |
| Optimiser la performance commerciale | EXPLICATIF 2 — Expliquer le positionnement tarifaire, la croissance, les jours fériés et la couverture géographique | Analyser les réservations sous, alignées, au-dessus du marché | Part réservations alignées marché | `(COUNT(0.85*benchmark ≤ final_price ≤ 1.15*benchmark) / COUNT(total)) × 100` |
| Optimiser la performance commerciale | EXPLICATIF 2 — Expliquer le positionnement tarifaire, la croissance, les jours fériés et la couverture géographique | Analyser les réservations sous, alignées, au-dessus du marché | Part réservations au-dessus marché | `(COUNT(final_price > 1.15*benchmark) / COUNT(total)) × 100` |
| Optimiser la performance commerciale | EXPLICATIF 2 — Expliquer le positionnement tarifaire, la croissance, les jours fériés et la couverture géographique | Analyser les réservations sous, alignées, au-dessus du marché | Taux de conversion | `(COUNT(reservations_confirmées) / SUM(visitors)) × 100` |
| Optimiser la performance commerciale | EXPLICATIF 2 — Expliquer le positionnement tarifaire, la croissance, les jours fériés et la couverture géographique | Analyser les opportunités de croissance par top N catégories et diversité | Top N catégories à ajouter | `TOP N(event_count_observed)` hors catégories couvertes |
| Optimiser la performance commerciale | EXPLICATIF 2 — Expliquer le positionnement tarifaire, la croissance, les jours fériés et la couverture géographique | Analyser les opportunités de croissance par top N catégories et diversité | Catégories nouvelles vs couvertes | `SET_DIFF(catégories marché, catégories EventZella)` |
| Optimiser la performance commerciale | EXPLICATIF 2 — Expliquer le positionnement tarifaire, la croissance, les jours fériés et la couverture géographique | Analyser les opportunités de croissance par top N catégories et diversité | Diversité des catégories réservées | `(COUNT(DISTINCT category_id) / COUNT(total_categories)) × 100` |
| Optimiser la performance commerciale | EXPLICATIF 2 — Expliquer le positionnement tarifaire, la croissance, les jours fériés et la couverture géographique | Analyser le comportement des réservations les jours fériés | Taux de réservation les jours fériés | `(réservations avec reservation_date ∈ jours fériés nationaux) / nombre total réservations × 100` |
| Optimiser la performance commerciale | EXPLICATIF 2 — Expliquer le positionnement tarifaire, la croissance, les jours fériés et la couverture géographique | Analyser le comportement des réservations les jours fériés | Nombre total de réservations | `COUNT(id_reservation)` |
| Optimiser la performance commerciale | EXPLICATIF 2 — Expliquer le positionnement tarifaire, la croissance, les jours fériés et la couverture géographique | Analyser le comportement des réservations les jours fériés | Taux de conversion | `(COUNT(reservations_confirmées) / SUM(visitors)) × 100` |
| Optimiser la performance commerciale | EXPLICATIF 2 — Expliquer le positionnement tarifaire, la croissance, les jours fériés et la couverture géographique | Analyser la couverture géographique et son impact sur les réservations | Répartition des salles par gouvernorat | `COUNT(venue_name) GROUP BY gouvernorat` |
| Optimiser la performance commerciale | EXPLICATIF 2 — Expliquer le positionnement tarifaire, la croissance, les jours fériés et la couverture géographique | Analyser la couverture géographique et son impact sur les réservations | Nombre de salles suggérables | `COUNT(venue_name)` |
| Optimiser la performance commerciale | EXPLICATIF 2 — Expliquer le positionnement tarifaire, la croissance, les jours fériés et la couverture géographique | Analyser la couverture géographique et son impact sur les réservations | Nombre total de réservations | `COUNT(id_reservation)` |
| Optimiser la performance commerciale | EXPLICATIF 2 — Expliquer le positionnement tarifaire, la croissance, les jours fériés et la couverture géographique | Analyser la couverture géographique et son impact sur les réservations | Taux de conversion | `(COUNT(reservations_confirmées) / SUM(visitors)) × 100` |
| Optimiser la performance commerciale | PRÉDICTIF — Prédire l’évolution future de la performance commerciale | Prédire le nombre de réservations futures | Nombre total de réservations (projection) | Prévision sur série historique de `COUNT(id_reservation)` — méthode à paramétrer en BI (*hors détail KPIs_FINAL.pdf*) |
| Optimiser la performance commerciale | PRÉDICTIF — Prédire l’évolution future de la performance commerciale | Prédire le taux de conversion futur | Taux de conversion (projection) | Prévision sur série historique du taux de conversion (*hors KPIs_FINAL.pdf*) |
| Optimiser la performance commerciale | PRÉDICTIF — Prédire l’évolution future de la performance commerciale | Prédire l’évolution du taux de rétention | Taux de rétention bénéficiaires (projection) | Prévision sur série du taux de rétention (*hors KPIs_FINAL.pdf*) |
| Optimiser la performance commerciale | PRÉDICTIF — Prédire l’évolution future de la performance commerciale | Prédire l’évolution du CAC et du LTV | CAC (projection) | Prévision sur série du CAC (*hors KPIs_FINAL.pdf*) |
| Optimiser la performance commerciale | PRÉDICTIF — Prédire l’évolution future de la performance commerciale | Prédire l’évolution du CAC et du LTV | LTV (projection) | Prévision sur série du LTV (*hors KPIs_FINAL.pdf*) |
| Optimiser la performance commerciale | PRÉDICTIF — Prédire l’évolution future de la performance commerciale | Prédire le taux de réservation les jours fériés | Taux de réservation les jours fériés (projection) | Prévision sur série du taux jours fériés (*hors KPIs_FINAL.pdf*) |
| Optimiser la rentabilité | DESCRIPTIF 1 — Décrire l’état actuel des revenus et de la performance financière | Visualiser le chiffre d’affaires total | Chiffre d’affaires total | `SUM(final_price)` |
| Optimiser la rentabilité | DESCRIPTIF 1 — Décrire l’état actuel des revenus et de la performance financière | Visualiser le panier moyen | Panier moyen | `AVG(final_price)` |
| Optimiser la rentabilité | DESCRIPTIF 2 — Décrire les top N catégories à ajouter | Identifier les top N catégories à ajouter | Top N catégories à ajouter | `TOP N(event_count_observed)` hors catégories couvertes |
| Optimiser la rentabilité | DESCRIPTIF 3 — Décrire la structure des revenus et des commissions | Visualiser le taux de commission sur réservation | Taux de commission sur réservation | PDF : « prix final de réservation – prix prestataire » — *incomplet* ; [ex. `(final_price - service_price) / final_price` selon CDC] |
| Optimiser la rentabilité | DESCRIPTIF 3 — Décrire la structure des revenus et des commissions | Comparer les revenus par période | Chiffre d’affaires total | `SUM(final_price)` |
| Optimiser la rentabilité | DESCRIPTIF 3 — Décrire la structure des revenus et des commissions | Comparer les revenus par période | Panier moyen | `AVG(final_price)` |
| Optimiser la rentabilité | DESCRIPTIF 4 — Décrire l’impact des jours fériés sur le ch.D’affaires | Visualiser l’impact des jours fériés sur le CA | Impact jours fériés sur CA | `SUM(final_price WHERE holiday) - SUM(final_price WHERE non_holiday)` |
| Optimiser la rentabilité | DESCRIPTIF 5 — Décrire les indicateurs de revenus et de commissions | Analyser les revenus par période et les commissions | Chiffre d’affaires total | `SUM(final_price)` |
| Optimiser la rentabilité | DESCRIPTIF 5 — Décrire les indicateurs de revenus et de commissions | Analyser les revenus par période et les commissions | Panier moyen | `AVG(final_price)` |
| Optimiser la rentabilité | DESCRIPTIF 5 — Décrire les indicateurs de revenus et de commissions | Analyser les revenus par période et les commissions | Commissions | Dérivé du taux de commission sur réservation × base (CA ou marge) — à cadrer avec la formule commission complétée |
| Optimiser la rentabilité | EXPLICATIF 1 — Expliquer la rentabilité, le CAC et la valeur client | Analyser la rentabilité via LTV, CAC, panier et catégories | LTV (lifetime value) | `Panier moyen × Fréquence × Durée rétention` |
| Optimiser la rentabilité | EXPLICATIF 1 — Expliquer la rentabilité, le CAC et la valeur client | Analyser la rentabilité via LTV, CAC, panier et catégories | CAC (coût d’acquisition client) | `SUM(marketing_spend) / SUM(new_beneficiaries)` |
| Optimiser la rentabilité | EXPLICATIF 1 — Expliquer la rentabilité, le CAC et la valeur client | Analyser la rentabilité via LTV, CAC, panier et catégories | Panier moyen | `AVG(final_price)` |
| Optimiser la rentabilité | EXPLICATIF 1 — Expliquer la rentabilité, le CAC et la valeur client | Analyser la rentabilité via LTV, CAC, panier et catégories | Chiffre d’affaires total | `SUM(final_price)` |
| Optimiser la rentabilité | EXPLICATIF 1 — Expliquer la rentabilité, le CAC et la valeur client | Analyser la rentabilité via LTV, CAC, panier et catégories | Catégories nouvelles vs couvertes | `SET_DIFF(catégories marché, catégories EventZella)` |
| Optimiser la rentabilité | EXPLICATIF 1 — Expliquer la rentabilité, le CAC et la valeur client | Analyser le CAC par rapport au CA, au taux de conversion et au panier | CAC (coût d’acquisition client) | `SUM(marketing_spend) / SUM(new_beneficiaries)` |
| Optimiser la rentabilité | EXPLICATIF 1 — Expliquer la rentabilité, le CAC et la valeur client | Analyser le CAC par rapport au CA, au taux de conversion et au panier | Chiffre d’affaires total | `SUM(final_price)` |
| Optimiser la rentabilité | EXPLICATIF 1 — Expliquer la rentabilité, le CAC et la valeur client | Analyser le CAC par rapport au CA, au taux de conversion et au panier | Taux de conversion | `(COUNT(reservations_confirmées) / SUM(visitors)) × 100` |
| Optimiser la rentabilité | EXPLICATIF 1 — Expliquer la rentabilité, le CAC et la valeur client | Analyser le CAC par rapport au CA, au taux de conversion et au panier | Panier moyen | `AVG(final_price)` |
| Optimiser la rentabilité | EXPLICATIF 1 — Expliquer la rentabilité, le CAC et la valeur client | Analyser l’impact du LTV sur le CA, le panier et les catégories | LTV (lifetime value) | `Panier moyen × Fréquence × Durée rétention` |
| Optimiser la rentabilité | EXPLICATIF 1 — Expliquer la rentabilité, le CAC et la valeur client | Analyser l’impact du LTV sur le CA, le panier et les catégories | Chiffre d’affaires total | `SUM(final_price)` |
| Optimiser la rentabilité | EXPLICATIF 1 — Expliquer la rentabilité, le CAC et la valeur client | Analyser l’impact du LTV sur le CA, le panier et les catégories | Panier moyen | `AVG(final_price)` |
| Optimiser la rentabilité | EXPLICATIF 1 — Expliquer la rentabilité, le CAC et la valeur client | Analyser l’impact du LTV sur le CA, le panier et les catégories | Catégories nouvelles vs couvertes | `SET_DIFF(catégories marché, catégories EventZella)` |
| Optimiser la rentabilité | EXPLICATIF 2 — Expliquer le positionnement tarifaire, les jours fériés, les commissions et la croissance | Analyser les réservations sous / alignées / au-dessus du marché vs CA et panier | Part réservations sous marché | `(COUNT(final_price < 0.85*benchmark) / COUNT(total)) × 100` |
| Optimiser la rentabilité | EXPLICATIF 2 — Expliquer le positionnement tarifaire, les jours fériés, les commissions et la croissance | Analyser les réservations sous / alignées / au-dessus du marché vs CA et panier | Part réservations alignées marché | `(COUNT(0.85*benchmark ≤ final_price ≤ 1.15*benchmark) / COUNT(total)) × 100` |
| Optimiser la rentabilité | EXPLICATIF 2 — Expliquer le positionnement tarifaire, les jours fériés, les commissions et la croissance | Analyser les réservations sous / alignées / au-dessus du marché vs CA et panier | Part réservations au-dessus marché | `(COUNT(final_price > 1.15*benchmark) / COUNT(total)) × 100` |
| Optimiser la rentabilité | EXPLICATIF 2 — Expliquer le positionnement tarifaire, les jours fériés, les commissions et la croissance | Analyser les réservations sous / alignées / au-dessus du marché vs CA et panier | Chiffre d’affaires total | `SUM(final_price)` |
| Optimiser la rentabilité | EXPLICATIF 2 — Expliquer le positionnement tarifaire, les jours fériés, les commissions et la croissance | Analyser les réservations sous / alignées / au-dessus du marché vs CA et panier | Panier moyen | `AVG(final_price)` |
| Optimiser la rentabilité | EXPLICATIF 2 — Expliquer le positionnement tarifaire, les jours fériés, les commissions et la croissance | Analyser l’impact des jours fériés sur le CA et le panier | Impact jours fériés sur CA | `SUM(final_price WHERE holiday) - SUM(final_price WHERE non_holiday)` |
| Optimiser la rentabilité | EXPLICATIF 2 — Expliquer le positionnement tarifaire, les jours fériés, les commissions et la croissance | Analyser l’impact des jours fériés sur le CA et le panier | Panier moyen | `AVG(final_price)` |
| Optimiser la rentabilité | EXPLICATIF 2 — Expliquer le positionnement tarifaire, les jours fériés, les commissions et la croissance | Analyser l’impact des jours fériés sur le CA et le panier | Revenus (CA) | `SUM(final_price)` |
| Optimiser la rentabilité | EXPLICATIF 2 — Expliquer le positionnement tarifaire, les jours fériés, les commissions et la croissance | Analyser le taux de commission vs revenus, panier et catégories | Taux de commission sur réservation | Voir ligne commission (PDF incomplet) |
| Optimiser la rentabilité | EXPLICATIF 2 — Expliquer le positionnement tarifaire, les jours fériés, les commissions et la croissance | Analyser le taux de commission vs revenus, panier et catégories | Chiffre d’affaires total | `SUM(final_price)` |
| Optimiser la rentabilité | EXPLICATIF 2 — Expliquer le positionnement tarifaire, les jours fériés, les commissions et la croissance | Analyser le taux de commission vs revenus, panier et catégories | Panier moyen | `AVG(final_price)` |
| Optimiser la rentabilité | EXPLICATIF 2 — Expliquer le positionnement tarifaire, les jours fériés, les commissions et la croissance | Analyser le taux de commission vs revenus, panier et catégories | Catégories nouvelles vs couvertes | `SET_DIFF(catégories marché, catégories EventZella)` |
| Optimiser la rentabilité | EXPLICATIF 2 — Expliquer le positionnement tarifaire, les jours fériés, les commissions et la croissance | Analyser la croissance via top N catégories et rentabilité | Top N catégories à ajouter | `TOP N(event_count_observed)` hors catégories couvertes |
| Optimiser la rentabilité | EXPLICATIF 2 — Expliquer le positionnement tarifaire, les jours fériés, les commissions et la croissance | Analyser la croissance via top N catégories et rentabilité | Catégories couvertes | Complement / dénombrement des catégories déjà sur la plateforme — à lier au référentiel catégories (*précision CDC*) |
| Optimiser la rentabilité | EXPLICATIF 2 — Expliquer le positionnement tarifaire, les jours fériés, les commissions et la croissance | Analyser la croissance via top N catégories et rentabilité | LTV (lifetime value) | `Panier moyen × Fréquence × Durée rétention` |
| Optimiser la rentabilité | EXPLICATIF 2 — Expliquer le positionnement tarifaire, les jours fériés, les commissions et la croissance | Analyser la croissance via top N catégories et rentabilité | CAC (coût d’acquisition client) | `SUM(marketing_spend) / SUM(new_beneficiaries)` |
| Optimiser la rentabilité | PRÉDICTIF — Prédire l’évolution future des revenus et de la rentabilité | Prédire l’évolution du chiffre d’affaires | Chiffre d’affaires total (projection) | Prévision sur série `SUM(final_price)` (*hors KPIs_FINAL.pdf*) |
| Optimiser la rentabilité | PRÉDICTIF — Prédire l’évolution future des revenus et de la rentabilité | Prédire l’évolution du panier moyen | Panier moyen (projection) | Prévision sur série `AVG(final_price)` (*hors KPIs_FINAL.pdf*) |
| Optimiser la rentabilité | PRÉDICTIF — Prédire l’évolution future des revenus et de la rentabilité | Prédire l’évolution du LTV et du CAC | LTV (projection) | Prévision série LTV (*hors KPIs_FINAL.pdf*) |
| Optimiser la rentabilité | PRÉDICTIF — Prédire l’évolution future des revenus et de la rentabilité | Prédire l’évolution du LTV et du CAC | CAC (projection) | Prévision série CAC (*hors KPIs_FINAL.pdf*) |
| Optimiser la rentabilité | PRÉDICTIF — Prédire l’évolution future des revenus et de la rentabilité | Prédire l’impact des jours fériés futurs sur le CA | Impact jours fériés sur CA (projection) | Prévision sur indicateur d’écart férié/non férié (*hors KPIs_FINAL.pdf*) |
| Améliorer la relation client | DESCRIPTIF 1 — Décrire l’état actuel de la satisfaction et de la relation client | Visualiser le nombre de réclamations | Nombre de réclamations | `COUNT(id_complaint)` |
| Améliorer la relation client | DESCRIPTIF 1 — Décrire l’état actuel de la satisfaction et de la relation client | Visualiser le taux d’annulation | Taux d’annulation | `(COUNT(reservations_annulées) / COUNT(total_reservations)) × 100` |
| Améliorer la relation client | DESCRIPTIF 2 — Décrire la note moyenne et le taux de réclamations | Comparer la note moyenne prestataires | Note moyenne prestataires | `AVG(rating)` |
| Améliorer la relation client | DESCRIPTIF 2 — Décrire la note moyenne et le taux de réclamations | Visualiser le taux de réclamations pour 100 réservations | Taux de réclamations pour 100 réservations | `(nb_reclamations / nb_reservations) × 100` |
| Améliorer la relation client | DESCRIPTIF 3 — Décrire la qualité du service et la réactivité | Visualiser le taux de résolution des réclamations | Taux de résolution réclamations | `(COUNT(closed_complaints) / COUNT(total_complaints)) × 100` |
| Améliorer la relation client | DESCRIPTIF 3 — Décrire la qualité du service et la réactivité | Comparer le NPS | NPS (Net Promoter Score) | `% promoteurs - % détracteurs` ; `rating ≥ 4` promoteur, `rating ≤ 2` détracteur |
| Améliorer la relation client | DESCRIPTIF 4 — Décrire la part des salles joignables | Visualiser la part des salles joignables | Part des salles joignables | `COUNT(contact NOT NULL) / COUNT(total venues) × 100` |
| Améliorer la relation client | DESCRIPTIF 5 — Décrire les indicateurs de satisfaction et de service | Analyser les réclamations, la résolution, le NPS et l’accessibilité | Nombre de réclamations | `COUNT(id_complaint)` |
| Améliorer la relation client | DESCRIPTIF 5 — Décrire les indicateurs de satisfaction et de service | Analyser les réclamations, la résolution, le NPS et l’accessibilité | Taux de résolution réclamations | `(COUNT(closed_complaints) / COUNT(total_complaints)) × 100` |
| Améliorer la relation client | DESCRIPTIF 5 — Décrire les indicateurs de satisfaction et de service | Analyser les réclamations, la résolution, le NPS et l’accessibilité | NPS | `% promoteurs - % détracteurs` |
| Améliorer la relation client | DESCRIPTIF 5 — Décrire les indicateurs de satisfaction et de service | Analyser les réclamations, la résolution, le NPS et l’accessibilité | Part des salles joignables | `COUNT(contact NOT NULL) / COUNT(total venues) × 100` |
| Améliorer la relation client | EXPLICATIF 1 — Expliquer la satisfaction, les réclamations, la rétention et la qualité de service | Analyser la satisfaction via note prestataires, annulation et réclamations pour 100 | Note moyenne prestataires | `AVG(rating)` |
| Améliorer la relation client | EXPLICATIF 1 — Expliquer la satisfaction, les réclamations, la rétention et la qualité de service | Analyser la satisfaction via note prestataires, annulation et réclamations pour 100 | Nombre de réclamations | `COUNT(id_complaint)` |
| Améliorer la relation client | EXPLICATIF 1 — Expliquer la satisfaction, les réclamations, la rétention et la qualité de service | Analyser la satisfaction via note prestataires, annulation et réclamations pour 100 | Taux d’annulation | `(COUNT(reservations_annulées) / COUNT(total_reservations)) × 100` |
| Améliorer la relation client | EXPLICATIF 1 — Expliquer la satisfaction, les réclamations, la rétention et la qualité de service | Analyser la satisfaction via note prestataires, annulation et réclamations pour 100 | Taux de réclamations pour 100 réservations | `(nb_reclamations / nb_reservations) × 100` |
| Améliorer la relation client | EXPLICATIF 1 — Expliquer la satisfaction, les réclamations, la rétention et la qualité de service | Analyser les réclamations vs annulation, note et résolution | Nombre de réclamations | `COUNT(id_complaint)` |
| Améliorer la relation client | EXPLICATIF 1 — Expliquer la satisfaction, les réclamations, la rétention et la qualité de service | Analyser les réclamations vs annulation, note et résolution | Taux d’annulation | `(COUNT(reservations_annulées) / COUNT(total_reservations)) × 100` |
| Améliorer la relation client | EXPLICATIF 1 — Expliquer la satisfaction, les réclamations, la rétention et la qualité de service | Analyser les réclamations vs annulation, note et résolution | Note moyenne prestataires | `AVG(rating)` |
| Améliorer la relation client | EXPLICATIF 1 — Expliquer la satisfaction, les réclamations, la rétention et la qualité de service | Analyser les réclamations vs annulation, note et résolution | Taux de résolution réclamations | `(COUNT(closed_complaints) / COUNT(total_complaints)) × 100` |
| Améliorer la relation client | EXPLICATIF 1 — Expliquer la satisfaction, les réclamations, la rétention et la qualité de service | Analyser la rétention vs réclamations pour 100 et annulation | Taux de réclamations pour 100 réservations | `(nb_reclamations / nb_reservations) × 100` |
| Améliorer la relation client | EXPLICATIF 1 — Expliquer la satisfaction, les réclamations, la rétention et la qualité de service | Analyser la rétention vs réclamations pour 100 et annulation | Taux de rétention bénéficiaires | `(COUNT(beneficiaries_recurrents) / COUNT(total_beneficiaries)) × 100` |
| Améliorer la relation client | EXPLICATIF 1 — Expliquer la satisfaction, les réclamations, la rétention et la qualité de service | Analyser la rétention vs réclamations pour 100 et annulation | Taux d’annulation | `(COUNT(reservations_annulées) / COUNT(total_reservations)) × 100` |
| Améliorer la relation client | EXPLICATIF 1 — Expliquer la satisfaction, les réclamations, la rétention et la qualité de service | Analyser la rétention vs réclamations pour 100 et annulation | Nombre de réclamations | `COUNT(id_complaint)` |
| Améliorer la relation client | EXPLICATIF 1 — Expliquer la satisfaction, les réclamations, la rétention et la qualité de service | Analyser la qualité de service via résolution, réclamations et salles joignables | Taux de résolution réclamations | `(COUNT(closed_complaints) / COUNT(total_complaints)) × 100` |
| Améliorer la relation client | EXPLICATIF 1 — Expliquer la satisfaction, les réclamations, la rétention et la qualité de service | Analyser la qualité de service via résolution, réclamations et salles joignables | Nombre de réclamations | `COUNT(id_complaint)` |
| Améliorer la relation client | EXPLICATIF 1 — Expliquer la satisfaction, les réclamations, la rétention et la qualité de service | Analyser la qualité de service via résolution, réclamations et salles joignables | Taux de réclamations pour 100 réservations | `(nb_reclamations / nb_reservations) × 100` |
| Améliorer la relation client | EXPLICATIF 1 — Expliquer la satisfaction, les réclamations, la rétention et la qualité de service | Analyser la qualité de service via résolution, réclamations et salles joignables | Part des salles joignables | `COUNT(contact NOT NULL) / COUNT(total venues) × 100` |
| Améliorer la relation client | EXPLICATIF 2 — Expliquer le positionnement tarifaire, le NPS et l’accessibilité | Analyser les réservations sous / alignées / au-dessus du marché vs réclamations et annulation | Part réservations sous marché | `(COUNT(final_price < 0.85*benchmark) / COUNT(total)) × 100` |
| Améliorer la relation client | EXPLICATIF 2 — Expliquer le positionnement tarifaire, le NPS et l’accessibilité | Analyser les réservations sous / alignées / au-dessus du marché vs réclamations et annulation | Part réservations alignées marché | `(COUNT(0.85*benchmark ≤ final_price ≤ 1.15*benchmark) / COUNT(total)) × 100` |
| Améliorer la relation client | EXPLICATIF 2 — Expliquer le positionnement tarifaire, le NPS et l’accessibilité | Analyser les réservations sous / alignées / au-dessus du marché vs réclamations et annulation | Part réservations au-dessus marché | `(COUNT(final_price > 1.15*benchmark) / COUNT(total)) × 100` |
| Améliorer la relation client | EXPLICATIF 2 — Expliquer le positionnement tarifaire, le NPS et l’accessibilité | Analyser les réservations sous / alignées / au-dessus du marché vs réclamations et annulation | Nombre de réclamations | `COUNT(id_complaint)` |
| Améliorer la relation client | EXPLICATIF 2 — Expliquer le positionnement tarifaire, le NPS et l’accessibilité | Analyser les réservations sous / alignées / au-dessus du marché vs réclamations et annulation | Taux d’annulation | `(COUNT(reservations_annulées) / COUNT(total_reservations)) × 100` |
| Améliorer la relation client | EXPLICATIF 2 — Expliquer le positionnement tarifaire, le NPS et l’accessibilité | Analyser le NPS vs réclamations, rétention et note | Nombre de réclamations | `COUNT(id_complaint)` |
| Améliorer la relation client | EXPLICATIF 2 — Expliquer le positionnement tarifaire, le NPS et l’accessibilité | Analyser le NPS vs réclamations, rétention et note | Taux de rétention bénéficiaires | `(COUNT(beneficiaries_recurrents) / COUNT(total_beneficiaries)) × 100` |
| Améliorer la relation client | EXPLICATIF 2 — Expliquer le positionnement tarifaire, le NPS et l’accessibilité | Analyser le NPS vs réclamations, rétention et note | Note moyenne prestataires | `AVG(rating)` |
| Améliorer la relation client | EXPLICATIF 2 — Expliquer le positionnement tarifaire, le NPS et l’accessibilité | Analyser le NPS vs réclamations, rétention et note | NPS | `% promoteurs - % détracteurs` |
| Améliorer la relation client | EXPLICATIF 2 — Expliquer le positionnement tarifaire, le NPS et l’accessibilité | Analyser l’accessibilité via salles joignables et indicateurs de réclamations | Part des salles joignables | `COUNT(contact NOT NULL) / COUNT(total venues) × 100` |
| Améliorer la relation client | EXPLICATIF 2 — Expliquer le positionnement tarifaire, le NPS et l’accessibilité | Analyser l’accessibilité via salles joignables et indicateurs de réclamations | Taux de réclamations pour 100 réservations | `(nb_reclamations / nb_reservations) × 100` |
| Améliorer la relation client | EXPLICATIF 2 — Expliquer le positionnement tarifaire, le NPS et l’accessibilité | Analyser l’accessibilité via salles joignables et indicateurs de réclamations | Nombre de réclamations | `COUNT(id_complaint)` |
| Améliorer la relation client | EXPLICATIF 2 — Expliquer le positionnement tarifaire, le NPS et l’accessibilité | Analyser l’accessibilité via salles joignables et indicateurs de réclamations | Taux d’annulation | `(COUNT(reservations_annulées) / COUNT(total_reservations)) × 100` |
| Améliorer la relation client | PRÉDICTIF — Prédire l’évolution future de la satisfaction et de la relation client | Prédire l’évolution du taux de réclamations pour 100 réservations | Taux de réclamations pour 100 réservations (projection) | Prévision série du ratio réclamations/réservations × 100 (*hors KPIs_FINAL.pdf*) |
| Améliorer la relation client | PRÉDICTIF — Prédire l’évolution future de la satisfaction et de la relation client | Prédire l’évolution du nombre de réclamations | Nombre de réclamations (projection) | Prévision sur `COUNT(id_complaint)` (*hors KPIs_FINAL.pdf*) |
| Améliorer la relation client | PRÉDICTIF — Prédire l’évolution future de la satisfaction et de la relation client | Prédire l’évolution du taux de rétention | Taux de rétention bénéficiaires (projection) | Prévision sur taux de rétention (*hors KPIs_FINAL.pdf*) |
| Améliorer la relation client | PRÉDICTIF — Prédire l’évolution future de la satisfaction et de la relation client | Prédire l’évolution du NPS | NPS (projection) | Prévision sur NPS (*hors KPIs_FINAL.pdf*) |

---

## Tableau 2 — Dashboards (3 par décideur) : KPI et visuel sur la même ligne

**Organisation :** chaque décideur a **au plus 3 rapports**. Les anciens regroupements ont été **fusionnés** (ex. activité + prix marché + fériés pour le marketing).  
**Colonnes :** Décideur — n° du dashboard — Titre du dashboard — **KPI ou information affichée** — **Visuel Power BI** associé à ce KPI.

| Décideur | Dash. | Titre du dashboard | KPI ou information | Visuel Power BI |
|----------|:-----:|-------------------|-------------------|-----------------|
| Responsable Marketing | 1 | **Importance de l’activité commerciale, du prix marché et du calendrier** | Nombre total de réservations | Carte multiplicateurs + **graphique en courbes** (tendance) |
| Responsable Marketing | 1 | **Importance de l’activité commerciale, du prix marché et du calendrier** | Taux de conversion | Carte KPI + **courbes** ou **colonnes** par période |
| Responsable Marketing | 1 | **Importance de l’activité commerciale, du prix marché et du calendrier** | Taux d’acceptation | Carte KPI + **histogramme** |
| Responsable Marketing | 1 | **Importance de l’activité commerciale, du prix marché et du calendrier** | Taux d’annulation | Carte KPI + **courbes** |
| Responsable Marketing | 1 | **Importance de l’activité commerciale, du prix marché et du calendrier** | Nombre de visiteurs | Carte KPI + **courbes** |
| Responsable Marketing | 1 | **Importance de l’activité commerciale, du prix marché et du calendrier** | Vue d’ensemble visiteurs → réservations | **Entonnoir** |
| Responsable Marketing | 1 | **Importance de l’activité commerciale, du prix marché et du calendrier** | Part des réservations sous le marché | **Graphique en anneau** ou **barres** (une série) |
| Responsable Marketing | 1 | **Importance de l’activité commerciale, du prix marché et du calendrier** | Part des réservations alignées au marché | **Graphique en anneau** ou **barres** |
| Responsable Marketing | 1 | **Importance de l’activité commerciale, du prix marché et du calendrier** | Part des réservations au-dessus du marché | **Graphique en anneau** ou **barres** |
| Responsable Marketing | 1 | **Importance de l’activité commerciale, du prix marché et du calendrier** | Taux de réservation les jours fériés | **Colonnes groupées** (férié / non férié) + **carte** KPI |
| Responsable Marketing | 2 | **Importance de la valeur client et qualité de l’offre géographique** | CAC | Carte KPI + **nuage de points** (ex. vs conversion) |
| Responsable Marketing | 2 | **Importance de la valeur client et qualité de l’offre géographique** | LTV | Carte KPI + **courbes** |
| Responsable Marketing | 2 | **Importance de la valeur client et qualité de l’offre géographique** | Taux de rétention des bénéficiaires | Carte KPI + **courbes** |
| Responsable Marketing | 2 | **Importance de la valeur client et qualité de l’offre géographique** | Nombre total de réservations (contexte) | **Courbes** |
| Responsable Marketing | 2 | **Importance de la valeur client et qualité de l’offre géographique** | Taux de conversion (contexte) | **Graphique combiné** lignes + colonnes |
| Responsable Marketing | 2 | **Importance de la valeur client et qualité de l’offre géographique** | Catégories nouvelles vs couvertes | **Table** ou **matrice** + **barres** |
| Responsable Marketing | 2 | **Importance de la valeur client et qualité de l’offre géographique** | Diversité des catégories réservées | Carte KPI + **treemap** ou **anneau** |
| Responsable Marketing | 2 | **Importance de la valeur client et qualité de l’offre géographique** | Top N catégories à ajouter | **Barres horizontales** (classement) |
| Responsable Marketing | 2 | **Importance de la valeur client et qualité de l’offre géographique** | Répartition des salles par gouvernorat | **Carte remplie** ou **barres** par région |
| Responsable Marketing | 2 | **Importance de la valeur client et qualité de l’offre géographique** | Nombre de salles suggérables | Carte KPI + **barres** par zone |
| Responsable Marketing | 3 | **Anticipation de l’activité commerciale** | Nombre total de réservations (projection) | **Ligne** historique + **ligne de prévision** |
| Responsable Marketing | 3 | **Anticipation de l’activité commerciale** | Taux de conversion (projection) | **Ligne** + prévision |
| Responsable Marketing | 3 | **Anticipation de l’activité commerciale** | Taux de rétention bénéficiaires (projection) | **Ligne** + prévision |
| Responsable Marketing | 3 | **Anticipation de l’activité commerciale** | CAC (projection) | **Ligne** + prévision |
| Responsable Marketing | 3 | **Anticipation de l’activité commerciale** | LTV (projection) | **Ligne** + prévision |
| Responsable Marketing | 3 | **Anticipation de l’activité commerciale** | Taux de réservation jours fériés (projection) | **Ligne** + prévision |
| Responsable Financier | 1 | **Importance du chiffre d’affaires et des commissions** | Chiffre d’affaires total | Carte KPI + **aires** ou **colonnes** (évolution) |
| Responsable Financier | 1 | **Importance du chiffre d’affaires et des commissions** | Panier moyen | Carte KPI + **courbes** |
| Responsable Financier | 1 | **Importance du chiffre d’affaires et des commissions** | Taux de commission sur réservation | Carte KPI + **colonnes** |
| Responsable Financier | 1 | **Importance du chiffre d’affaires et des commissions** | Montant des commissions (agrégat) | Carte KPI + **courbes** |
| Responsable Financier | 1 | **Importance du chiffre d’affaires et des commissions** | Impact des jours fériés sur le CA | **Colonnes groupées** (férié / non férié) ou **indicateur** d’écart |
| Responsable Financier | 2 | **Qualité de la rentabilité, du catalogue et du marché** | LTV | Carte KPI + **dispersion** LTV vs segment |
| Responsable Financier | 2 | **Qualité de la rentabilité, du catalogue et du marché** | CAC | Carte KPI + **nuage de points** |
| Responsable Financier | 2 | **Qualité de la rentabilité, du catalogue et du marché** | Panier moyen | **Courbes** + carte KPI |
| Responsable Financier | 2 | **Qualité de la rentabilité, du catalogue et du marché** | Chiffre d’affaires total (contexte rentabilité) | **Courbes** |
| Responsable Financier | 2 | **Qualité de la rentabilité, du catalogue et du marché** | Taux de conversion | Carte + **barres** |
| Responsable Financier | 2 | **Qualité de la rentabilité, du catalogue et du marché** | Catégories nouvelles vs couvertes | **Matrice** ou **barres** |
| Responsable Financier | 2 | **Qualité de la rentabilité, du catalogue et du marché** | Parts réservations sous / alignées / au-dessus marché | **Barres empilées 100 %** ou **anneaux multiples** |
| Responsable Financier | 2 | **Qualité de la rentabilité, du catalogue et du marché** | Impact jours fériés sur le CA | **Ligne + colonnes** combinés |
| Responsable Financier | 2 | **Qualité de la rentabilité, du catalogue et du marché** | Top N catégories à ajouter | **Barres horizontales** |
| Responsable Financier | 2 | **Qualité de la rentabilité, du catalogue et du marché** | Catégories déjà couvertes | **Table** ou **anneau** (volume par catégorie) |
| Responsable Financier | 3 | **Anticipation du chiffre d’affaires et de la rentabilité** | Chiffre d’affaires total (projection) | **Ligne** historique + prévision |
| Responsable Financier | 3 | **Anticipation du chiffre d’affaires et de la rentabilité** | Panier moyen (projection) | **Ligne** + prévision |
| Responsable Financier | 3 | **Anticipation du chiffre d’affaires et de la rentabilité** | LTV (projection) | **Ligne** + prévision |
| Responsable Financier | 3 | **Anticipation du chiffre d’affaires et de la rentabilité** | CAC (projection) | **Ligne** + prévision |
| Responsable Financier | 3 | **Anticipation du chiffre d’affaires et de la rentabilité** | Impact jours fériés sur le CA (projection) | **Ligne** + prévision |
| Responsable Relation Client | 1 | **Qualité de l’expérience et satisfaction sur le suivi** | Nombre de réclamations | Carte KPI + **courbes** |
| Responsable Relation Client | 1 | **Qualité de l’expérience et satisfaction sur le suivi** | Taux d’annulation | Carte KPI + **courbes** |
| Responsable Relation Client | 1 | **Qualité de l’expérience et satisfaction sur le suivi** | Note moyenne des prestataires | Carte KPI + **jauge** ou histogramme de notes |
| Responsable Relation Client | 1 | **Qualité de l’expérience et satisfaction sur le suivi** | Taux de réclamations pour 100 réservations | Carte KPI + **courbes** |
| Responsable Relation Client | 1 | **Qualité de l’expérience et satisfaction sur le suivi** | Répartition des motifs de réclamation (si données) | **Barres** ou **Pareto** |
| Responsable Relation Client | 1 | **Qualité de l’expérience et satisfaction sur le suivi** | Taux de résolution des réclamations | Carte KPI + **barres** par statut |
| Responsable Relation Client | 1 | **Qualité de l’expérience et satisfaction sur le suivi** | NPS | **Jauge** ou **carte avec indicateur** |
| Responsable Relation Client | 1 | **Qualité de l’expérience et satisfaction sur le suivi** | Distribution des scores de recommandation | **Histogramme** |
| Responsable Relation Client | 2 | **Qualité de l’accès aux salles et du niveau de prix** | Part des salles joignables | Carte KPI + **carte** géographique ou **barres** par région |
| Responsable Relation Client | 2 | **Qualité de l’accès aux salles et du niveau de prix** | Part réservations sous marché | **Anneau** ou **barre** |
| Responsable Relation Client | 2 | **Qualité de l’accès aux salles et du niveau de prix** | Part réservations alignées marché | **Anneau** ou **barre** |
| Responsable Relation Client | 2 | **Qualité de l’accès aux salles et du niveau de prix** | Part réservations au-dessus marché | **Anneau** ou **barre** |
| Responsable Relation Client | 2 | **Qualité de l’accès aux salles et du niveau de prix** | Lien prix vs plaintes (analyse) | **Matrice** ou **nuage de points** (prix / réclamations) |
| Responsable Relation Client | 3 | **Anticipation de la satisfaction et de la fidélité** | Taux de réclamations pour 100 réservations (projection) | **Ligne** + prévision |
| Responsable Relation Client | 3 | **Anticipation de la satisfaction et de la fidélité** | Nombre de réclamations (projection) | **Ligne** + prévision |
| Responsable Relation Client | 3 | **Anticipation de la satisfaction et de la fidélité** | Taux de rétention bénéficiaires (projection) | **Ligne** + prévision |
| Responsable Relation Client | 3 | **Anticipation de la satisfaction et de la fidélité** | NPS (projection) | **Ligne** + prévision + **carte** objectif/cible |

*Les libellés de visuels correspondent aux types d’objets standards dans Power BI (carte, graphique en courbes, barres, entonnoir, matrice, nuage de points, carte remplie, jauge, etc.).*

---

## Export PDF

Utiliser `scripts/convert_EventZilla_doc_to_pdf.ps1` (à lancer depuis la racine du dépôt) ou l’aperçu Markdown → **Imprimer** → PDF.

---

*Les formules non couvertes par `KPIs_FINAL.pdf` (projections, « catégories couvertes » en détail, commission complète) sont à figer avec le CDC.*

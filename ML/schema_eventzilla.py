# -*- coding: utf-8 -*-
"""
Schéma Data Warehouse EventZilla — aligné sur la modélisation du projet :
- `ScriptsDiagrams/EventZilla_DWH_Model.dbml` et `EventZilla_DWH_Par_Fact_Mermaid.md`
- Noms **faits / dimensions** tels que dans Power BI / `docs/Liste_Des_Kpis_Updated_English_DAX.md` :
  `Fact_PerformanceCommerciale`, `Fact_RentabiliteFinanciere`, `Fact_SatisfactionClient`, etc.

Sauvegardes SQL Server (dossier `FilesMachine/DB/`) :
- **DW_Eventzilla** — entrepôt (utiliser cette base pour le ML)
- **SA_eventzilla** — staging area

Si SSMS affiche des noms de tables différents, modifiez uniquement les constantes ci-dessous.

**Fait rentabilité sans FK vers ``DimReservation``** (pas de ``id_reservation`` sur le fait) : cas courant.
La requête large tente alors, dans l’ordre : (1) jointure directe ``f.id_event = r.<col>`` si
``DimReservation`` expose une colonne événement (ex. ``id_event_SK``) ; (2) pont
``f → DimEvent → DimReservation`` ; sinon le jeu sort **sans** ``status``. Pour forcer la colonne
côté réservation : variable d’environnement ``EVENTZILLA_DIM_RESERVATION_EVENT_COL`` (nom exact SSMS).

Pour le jeu financier large, préférer ``build_sql_ml_financial_wide(engine)`` : la liste des colonnes
est dérivée de ``INFORMATION_SCHEMA`` afin d’éviter les erreurs *colonne non valide* lorsque le DW
diffère légèrement du modèle DBML (ex. pas de ``fact_finance_id``, pas de ``quarter`` sur ``DimDate``).
"""
from __future__ import annotations

import os

# Schéma SQL des tables (dbo par défaut)
_SQL_SCHEMA = os.environ.get("EVENTZILLA_SQL_SCHEMA", "dbo")

# Clés côté **fait** (FK) et côté **dimension** (PK / surrogate), alignées sur les ALTER TABLE SSMS EventZilla :
#   Fact.*.id_date       → DimDate(id_date_SK)
#   Fact.*.id_reservation → DimReservation(id_reservation_SK)
#   Fact.*.id_event      → DimEvent(id_event_SK)
_FACT_RESERVATION_FK = os.environ.get("EVENTZILLA_FACT_RESERVATION_FK", "id_reservation")
_DIM_RESERVATION_PK = os.environ.get("EVENTZILLA_DIM_RESERVATION_PK", "id_reservation_SK")
_DIM_DATE_PK = os.environ.get("EVENTZILLA_DIM_DATE_PK", "id_date_SK")
_DIM_EVENT_PK = os.environ.get("EVENTZILLA_DIM_EVENT_PK", "id_event_SK")

_JOIN_FACT_DIM_RES = (
    f"f.[{_FACT_RESERVATION_FK}] = r.[{_DIM_RESERVATION_PK}]"
)
_JOIN_FACT_DIM_DATE = f"f.[id_date] = d.[{_DIM_DATE_PK}]"

# --- Faits (Power BI / projet) ---
FACT_RENTABILITE = "Fact_RentabiliteFinanciere"  # DBML: Fact_FinancialProfitability
FACT_PERFORMANCE = "Fact_PerformanceCommerciale"  # DBML: Fact_CommercialPerformance
FACT_SATISFACTION = "Fact_SatisfactionClient"  # DBML: Fact_CustomerSatisfaction

# --- Dimensions ---
DIM_DATE = "DimDate"
DIM_RESERVATION = "DimReservation"
DIM_BENEFICIARY = "DimBeneficiary"
DIM_EVENT = "DimEvent"
DIM_SERVICE_CATEGORY = "DimServiceCategory"
DIM_BENCHMARK = "DimBenchmarkPrice"
DIM_PROVIDER = "DimProvider"
DIM_VISITORS = "DimVisitors"
DIM_FEEDBACK = "DimFeedback"
DIM_COMPLAINT = "DimComplaint"
DIM_MARKETING_SPEND = "DimMarketingSpend"
DIM_TENDANCE_MARCHE = "DimTendanceMarche"
DIM_VENUE = "DimVenue"

# Colonnes « identifiant » souvent à exclure des features clustering (distances K-Means).
# (Le nom exact de la PK du fait performance varie selon les scripts de déploiement ;
# ne pas la mettre dans le SELECT si elle n’existe pas — cf. SQL_ML_PERFORMANCE_WIDE.)
CLUSTERING_NUMERIC_DROP: tuple[str, ...] = (
    "fact_marketing_id",
    "fact_finance_id",
    "fact_performance_id",
    "fact_commercial_id",
    "id_date",
    "id_reservation",
)

# Requêtes pour ML (dbo.) — à exécuter sur la base **DW** restaurée depuis `DW_Eventzilla`

# Requête statique de secours : **fait + date** seulement (sans DimReservation) pour les DW où
# le fait n’a pas ``id_reservation``. Préférer ``build_sql_ml_financial_wide(engine)``.
SQL_ML_FINANCIAL_WIDE: str = f"""
SELECT TOP 200000
  f.id_date,
  f.id_event,
  f.id_servicecategory,
  f.id_benchmark,
  f.id_provider,
  f.final_price,
  f.service_price,
  f.benchmark_avg_price,
  f.event_budget,
  d.full_date,
  d.[month] AS cal_month,
  d.[year] AS cal_year,
  d.is_holiday
FROM [{_SQL_SCHEMA}].[{FACT_RENTABILITE}] f
INNER JOIN [{_SQL_SCHEMA}].[{DIM_DATE}] d ON {_JOIN_FACT_DIM_DATE}
""".strip()

SQL_ML_PERFORMANCE_WIDE: str = f"""
SELECT TOP 200000
  f.id_date,
  f.id_event,
  f.id_reservation,
  f.id_beneficiary,
  f.id_servicecategory,
  f.id_provider,
  f.id_visitors,
  f.nb_visitors,
  f.nb_reservations_site,
  f.final_price,
  f.event_budget,
  f.service_price,
  r.status AS reservation_status,
  d.full_date,
  d.[month] AS cal_month,
  d.[year] AS cal_year,
  d.is_holiday
FROM [{_SQL_SCHEMA}].[{FACT_PERFORMANCE}] f
INNER JOIN [{_SQL_SCHEMA}].[{DIM_RESERVATION}] r ON {_JOIN_FACT_DIM_RES}
INNER JOIN [{_SQL_SCHEMA}].[{DIM_DATE}] d ON {_JOIN_FACT_DIM_DATE}
""".strip()

# Agrégations **par bénéficiaire** (RFM / fidélité) — une ligne = un client actif sur la période DW.
SQL_ML_CLUSTERING_LOYALTY_BENEFICIARY: str = f"""
SELECT TOP 200000
  f.id_beneficiary,
  CAST(COUNT(*) AS FLOAT) AS nb_reservations_loyalty,
  CAST(SUM(CAST(f.final_price AS FLOAT)) AS FLOAT) AS ca_total_loyalty,
  CAST(AVG(CAST(f.final_price AS FLOAT)) AS FLOAT) AS panier_moyen_loyalty,
  CAST(DATEDIFF(day, MAX(d.full_date), CAST(GETDATE() AS DATE)) AS FLOAT) AS recency_days_loyalty,
  CAST(AVG(CAST(ISNULL(f.nb_visitors, 0) AS FLOAT)) AS FLOAT) AS avg_nb_visitors_loyalty,
  CAST(SUM(CAST(ISNULL(f.nb_reservations_site, 0) AS FLOAT)) AS FLOAT) AS volume_reservations_site_loyalty
FROM [{_SQL_SCHEMA}].[{FACT_PERFORMANCE}] f
INNER JOIN [{_SQL_SCHEMA}].[{DIM_RESERVATION}] r ON {_JOIN_FACT_DIM_RES}
INNER JOIN [{_SQL_SCHEMA}].[{DIM_DATE}] d ON {_JOIN_FACT_DIM_DATE}
WHERE f.id_beneficiary IS NOT NULL
GROUP BY f.id_beneficiary
HAVING COUNT(*) >= 1
""".strip()

# Agrégations **par prestataire** (fidélité / volume / CA) — une ligne = un fournisseur.
SQL_ML_CLUSTERING_LOYALTY_PROVIDER: str = f"""
SELECT TOP 200000
  f.id_provider,
  CAST(COUNT(*) AS FLOAT) AS nb_reservations_loyalty,
  CAST(SUM(CAST(f.final_price AS FLOAT)) AS FLOAT) AS ca_total_loyalty,
  CAST(AVG(CAST(f.final_price AS FLOAT)) AS FLOAT) AS panier_moyen_loyalty,
  CAST(DATEDIFF(day, MAX(d.full_date), CAST(GETDATE() AS DATE)) AS FLOAT) AS recency_days_loyalty,
  CAST(AVG(CAST(ISNULL(f.nb_visitors, 0) AS FLOAT)) AS FLOAT) AS avg_nb_visitors_loyalty,
  CAST(SUM(CAST(ISNULL(f.nb_reservations_site, 0) AS FLOAT)) AS FLOAT) AS volume_reservations_site_loyalty
FROM [{_SQL_SCHEMA}].[{FACT_PERFORMANCE}] f
INNER JOIN [{_SQL_SCHEMA}].[{DIM_RESERVATION}] r ON {_JOIN_FACT_DIM_RES}
INNER JOIN [{_SQL_SCHEMA}].[{DIM_DATE}] d ON {_JOIN_FACT_DIM_DATE}
WHERE f.id_provider IS NOT NULL
GROUP BY f.id_provider
HAVING COUNT(*) >= 1
""".strip()

SQL_ML_TIME_SERIES_RESERVATIONS: str = f"""
SELECT
  d.[year] AS cal_year,
  d.[month] AS cal_month,
  COUNT(*) AS nb_fact_rows,
  SUM(f.final_price) AS revenue_sum,
  AVG(CAST(f.final_price AS FLOAT)) AS avg_final_price
FROM [{_SQL_SCHEMA}].[{FACT_RENTABILITE}] f
INNER JOIN [{_SQL_SCHEMA}].[{DIM_DATE}] d ON {_JOIN_FACT_DIM_DATE}
GROUP BY d.[year], d.[month]
ORDER BY d.[year], d.[month]
""".strip()

SQL_LIST_TABLES: str = """
SELECT TABLE_SCHEMA, TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE'
ORDER BY TABLE_SCHEMA, TABLE_NAME
""".strip()


def _fetch_dbo_column_names(engine, table_name: str) -> list[str]:
    """Colonnes réelles (ordre ORDINAL) pour adapter les SELECT au DW."""
    from sqlalchemy import text

    q = text(
        """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :sch AND TABLE_NAME = :tn
        ORDER BY ORDINAL_POSITION
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(q, {"sch": _SQL_SCHEMA, "tn": table_name}).fetchall()
    return [str(r[0]) for r in rows]


def _lower_map(names: list[str]) -> dict[str, str]:
    return {n.lower(): n for n in names}


def _first_existing(cmap: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c.lower() in cmap:
            return cmap[c.lower()]
    return None


def _resolve_financial_reservation_join(
    f_m: dict[str, str],
    r_m: dict[str, str],
    fact_column_names: list[str],
) -> tuple[str, str]:
    """
    Noms réels des colonnes ``f.[…]`` et ``r.[…]`` pour la jointure sur la réservation.
    Certains DW n’ont pas ``id_reservation`` sur le fait rentabilité : on essaie des alias courants
    ou une paire explicite via l’environnement.
    """
    fc_env = os.environ.get("EVENTZILLA_FINANCIAL_FACT_RES_COL", "").strip()
    rc_env = os.environ.get("EVENTZILLA_FINANCIAL_DIM_RES_COL", "").strip()
    if fc_env and rc_env:
        if fc_env.lower() not in f_m or rc_env.lower() not in r_m:
            raise ValueError(
                f"EVENTZILLA_FINANCIAL_FACT_RES_COL / DIM_RES_COL invalides "
                f"({fc_env!r}, {rc_env!r}) — colonnes absentes."
            )
        return f_m[fc_env.lower()], r_m[rc_env.lower()]

    if _DIM_RESERVATION_PK.lower() not in r_m:
        raise ValueError(
            f"Colonne {_DIM_RESERVATION_PK} absente sur {DIM_RESERVATION} (clé réservation)."
        )
    fk_r = r_m[_DIM_RESERVATION_PK.lower()]

    rent_one = os.environ.get("EVENTZILLA_FACT_RENTABILITE_RESERVATION_FK", "").strip().lower()
    if rent_one and rent_one in f_m:
        return f_m[rent_one], fk_r

    fact_try = (
        _FACT_RESERVATION_FK.lower(),
        "id_reservation",
        "id_reservation_sk",
        "reservation_id",
        "fk_reservation",
        "fk_id_reservation",
        "idresa",
        "reservationid",
    )
    for c in fact_try:
        if c in f_m:
            return f_m[c], fk_r

    commons = sorted(k for k in f_m if k in r_m and k != "id_date")
    for k in commons:
        if "reservation" in k:
            return f_m[k], r_m[k]

    preview = ", ".join(fact_column_names[:35])
    if len(fact_column_names) > 35:
        preview += ", …"
    raise ValueError(
        "Impossible de joindre automatiquement le fait rentabilité à DimReservation "
        f"(pas de colonne FK réservation reconnue). Colonnes du fait : {preview}. "
        "Définissez EVENTZILLA_FINANCIAL_FACT_RES_COL et EVENTZILLA_FINANCIAL_DIM_RES_COL "
        "(noms exacts côté fait et côté dimension), ou EVENTZILLA_FACT_RENTABILITE_RESERVATION_FK."
    )


def _financial_join_bridge_event_sql(sch: str, engine, f_m: dict[str, str], r_m: dict[str, str]) -> str | None:
    """Jointure f → DimEvent → DimReservation lorsque le fait n’a pas id_reservation mais a id_event."""
    e_cols = _fetch_dbo_column_names(engine, DIM_EVENT)
    e_m = _lower_map(e_cols)
    ev_pk = _DIM_EVENT_PK.lower()
    if "id_event" not in f_m or ev_pk not in e_m:
        return None
    e_res = _first_existing(
        e_m,
        (
            "id_reservation",
            "reservation_id",
            "id_reservation_sk",
            "fk_reservation",
            "fk_id_reservation",
            "id_reservation_fk",
        ),
    )
    if not e_res or _DIM_RESERVATION_PK.lower() not in r_m:
        return None
    r_pk = r_m[_DIM_RESERVATION_PK.lower()]
    fe = f_m["id_event"]
    ee = e_m[ev_pk]
    ev = DIM_EVENT
    rr = DIM_RESERVATION
    return (
        f"INNER JOIN [{sch}].[{ev}] e ON f.[{fe}] = e.[{ee}]\n"
        f"INNER JOIN [{sch}].[{rr}] r ON e.[{e_res}] = r.[{r_pk}]"
    )


def _reservation_dim_event_column(r_m: dict[str, str]) -> str | None:
    """
    Colonne sur ``DimReservation`` pour joindre ``f.id_event`` (fait **sans** FK réservation).
    Priorité : ``EVENTZILLA_DIM_RESERVATION_EVENT_COL``, puis noms usuels.
    """
    env = os.environ.get("EVENTZILLA_DIM_RESERVATION_EVENT_COL", "").strip()
    if env:
        ll = env.lower()
        if ll in r_m:
            return r_m[ll]
    return _first_existing(
        r_m,
        (
            "id_event_sk",
            "id_event",
            "event_id",
            "fk_event",
            "id_event_fk",
            "sk_event",
        ),
    )


def _financial_join_reservation_via_event_on_dim(sch: str, f_m: dict[str, str], r_m: dict[str, str]) -> str | None:
    """
    Jointure ``f.id_event = r.<col>`` lorsque le fait rentabilité **n’a pas** de FK vers
    ``DimReservation`` mais porte ``id_event`` et la dimension réservation référence l’événement.

    Peut dupliquer des lignes du fait si plusieurs réservations partagent le même événement ;
    acceptable pour la classification sur ``status``.
    """
    if "id_event" not in f_m:
        return None
    rev = _reservation_dim_event_column(r_m)
    if not rev:
        return None
    fe = f_m["id_event"]
    rr = DIM_RESERVATION
    return f"INNER JOIN [{sch}].[{rr}] r ON f.[{fe}] = r.[{rev}]"


def build_sql_ml_financial_wide(engine) -> str:
    """
    Construit ``SELECT TOP 200000`` sur le fait rentabilité + DimReservation + DimDate
    en ne listant que les colonnes présentes dans ``INFORMATION_SCHEMA`` (évite 42S22).

    Jointure ``DimReservation`` : d’abord FK réservation sur le fait si présente ; sinon
    ``f.id_event = r.<événement>`` (fait sans FK réservation) ; sinon pont via ``DimEvent``.

    À utiliser avec ``read_dw_sql(build_sql_ml_financial_wide(engine), engine)``.
    """
    f_cols = _fetch_dbo_column_names(engine, FACT_RENTABILITE)
    r_cols = _fetch_dbo_column_names(engine, DIM_RESERVATION)
    d_cols = _fetch_dbo_column_names(engine, DIM_DATE)
    if not f_cols:
        raise ValueError(
            f"Aucune colonne trouvée pour [{_SQL_SCHEMA}].[{FACT_RENTABILITE}] — vérifier le nom de table."
        )
    f_m = _lower_map(f_cols)
    r_m = _lower_map(r_cols)
    d_m = _lower_map(d_cols)

    if "id_date" not in f_m:
        raise ValueError("Colonne id_date requise sur le fait rentabilité.")
    ddpk = _DIM_DATE_PK.lower()
    if ddpk not in d_m:
        raise ValueError(
            f"Colonne {_DIM_DATE_PK} absente sur {DIM_DATE} (jointure f.id_date → d.{_DIM_DATE_PK}, cf. FK SSMS)."
        )

    sch = _SQL_SCHEMA
    ff = FACT_RENTABILITE
    rr = DIM_RESERVATION
    dd = DIM_DATE
    use_reservation = True
    try:
        fk_f, fk_r = _resolve_financial_reservation_join(f_m, r_m, f_cols)
        join_clause = f"INNER JOIN [{sch}].[{rr}] r ON f.[{fk_f}] = r.[{fk_r}]"
    except ValueError:
        # Cas le plus fréquent sans id_reservation sur le fait : id_event + colonne événement sur DimReservation
        bridge_ev = _financial_join_reservation_via_event_on_dim(sch, f_m, r_m)
        if bridge_ev is not None:
            join_clause = bridge_ev
        else:
            bridge = _financial_join_bridge_event_sql(sch, engine, f_m, r_m)
            if bridge is not None:
                join_clause = bridge
            else:
                # DW sans chemin reconnu vers DimReservation : jeu date + mesures seulement.
                use_reservation = False
                join_clause = ""

    id_df = f_m["id_date"]
    id_dd = d_m[ddpk]

    want_fact = (
        "fact_finance_id",
        "id_date",
        "id_reservation",
        "id_event",
        "id_servicecategory",
        "id_benchmark",
        "id_provider",
        "final_price",
        "service_price",
        "benchmark_avg_price",
        "event_budget",
    )
    select_parts: list[str] = []
    for w in want_fact:
        if w.lower() in f_m:
            c = f_m[w.lower()]
            select_parts.append(f"f.[{c}]")

    if use_reservation:
        st = _first_existing(r_m, ("status", "reservation_status", "statut"))
        if st:
            select_parts.append(f"r.[{st}] AS reservation_status")
        rd = _first_existing(r_m, ("reservation_date", "booking_date", "created_date", "date_reservation"))
        if rd:
            select_parts.append(f"r.[{rd}] AS reservation_date")

    dim_specs = (
        ("full_date", "full_date"),
        ("month", "cal_month"),
        ("year", "cal_year"),
        ("quarter", "quarter"),
        ("is_holiday", "is_holiday"),
        ("is_weekend", "is_weekend"),
    )
    for logical, alias in dim_specs:
        if logical.lower() not in d_m:
            continue
        c = d_m[logical.lower()]
        if alias == logical:
            select_parts.append(f"d.[{c}]")
        else:
            select_parts.append(f"d.[{c}] AS {alias}")

    body = ",\n  ".join(select_parts)
    join_lines = f"{join_clause}\n" if join_clause.strip() else ""
    return f"""
SELECT TOP 200000
  {body}
FROM [{sch}].[{ff}] f
{join_lines}INNER JOIN [{sch}].[{dd}] d ON f.[{id_df}] = d.[{id_dd}]
""".strip()


def build_sql_ml_performance_status_bridge(engine) -> str:
    """
    Pont **Fact_Performance** + ``DimReservation`` : ``id_date``, ``id_event``, ``reservation_status``.

    Le fait rentabilité n’a souvent **pas** de FK vers ``DimReservation`` ; le fait performance, lui,
    joint en général ``r.status`` via ``id_reservation``. On réaligne ensuite le jeu large financier
    sur ``(id_date, id_event)``.
    """
    p_cols = _fetch_dbo_column_names(engine, FACT_PERFORMANCE)
    r_cols = _fetch_dbo_column_names(engine, DIM_RESERVATION)
    if not p_cols or not r_cols:
        raise ValueError("Colonnes introuvables pour Fact_Performance ou DimReservation.")
    p_m = _lower_map(p_cols)
    r_m = _lower_map(r_cols)
    if "id_date" not in p_m:
        raise ValueError("Fact_Performance sans id_date.")
    if "id_event" not in p_m:
        raise ValueError(
            "Fact_Performance sans id_event — impossible d'aligner avec le fait rentabilité sur (id_date, id_event)."
        )
    fk_f, fk_r = _resolve_financial_reservation_join(p_m, r_m, p_cols)
    st = _first_existing(r_m, ("status", "reservation_status", "statut"))
    if not st:
        raise ValueError("DimReservation sans colonne status / statut.")
    id_df = p_m["id_date"]
    id_ev = p_m["id_event"]
    sch = _SQL_SCHEMA
    ff = FACT_PERFORMANCE
    rr = DIM_RESERVATION
    return f"""
SELECT TOP 200000
  f.[{id_df}] AS id_date,
  f.[{id_ev}] AS id_event,
  r.[{st}] AS reservation_status
FROM [{sch}].[{ff}] f
INNER JOIN [{sch}].[{rr}] r ON f.[{fk_f}] = r.[{fk_r}]
""".strip()


def ml_financial_wide_sql_tables_lineage() -> str:
    """
    Résumé des **tables** impliquées dans ``build_sql_ml_financial_wide`` / ``SQL_ML_FINANCIAL_WIDE``
    (pour documentation / notebooks — noms alignés SSMS ``dbo``).
    """
    sch = _SQL_SCHEMA
    return (
        f"Jeu large ML `df_ml` / `dw_financial_wide` provient typiquement de :\n"
        f"  • Fait rentabilité : [{sch}].[{FACT_RENTABILITE}] (alias SQL **f**)\n"
        f"  • Calendrier      : [{sch}].[{DIM_DATE}] (alias **d**) — jointure "
        f"f.id_date = d.{_DIM_DATE_PK}\n"
        f"  • Réservations    : [{sch}].[{DIM_RESERVATION}] (alias **r**) "
        f"lorsque la jointure FK ou événement / pont est résolue\n"
        f"(sinon le SELECT peut se limiter à fait + date)."
    )


def infer_column_dw_source(col: str) -> str:
    """
    Pour une **colonne du DataFrame** issu de la requête large (avant ou après one-hot),
    indique la table DW la plus probable (documentation pédagogique).

    Les colonnes dummy ``get_dummies`` reprennent souvent le préfixe de la colonne source
    (ex. ``full_date_2024-01-15`` → modalité de ``full_date`` sur DimDate).
    """
    sch = _SQL_SCHEMA
    s = str(col).strip()
    sl = s.lower()

    if sl.startswith("full_date_"):
        return f"[{sch}].[{DIM_DATE}] (modalité 0/1 issue de la colonne full_date)"
    if sl.startswith("reservation_status_") or sl.startswith("status_"):
        return f"[{sch}].[{DIM_RESERVATION}] (modalité 0/1 issue du statut réservation)"
    if sl in (
        "full_date",
        "cal_month",
        "cal_year",
        "quarter",
        "is_holiday",
        "is_weekend",
        "month",
        "year",
    ):
        return f"[{sch}].[{DIM_DATE}]"
    if sl in ("reservation_status", "reservation_date", "status", "booking_date", "date_reservation"):
        return f"[{sch}].[{DIM_RESERVATION}]"

    fact_like = (
        "final_price",
        "service_price",
        "benchmark_avg_price",
        "event_budget",
        "fact_finance_id",
        "id_date",
        "id_reservation",
        "id_event",
        "id_servicecategory",
        "id_benchmark",
        "id_provider",
    )
    if sl in fact_like or (sl.startswith("id_") and not sl.startswith("full_date")):
        return f"[{sch}].[{FACT_RENTABILITE}]"

    if "_" in s and not sl.startswith("full_date"):
        head = s.split("_", 1)[0].lower()
        if head in ("reservation", "status", "statut"):
            return f"[{sch}].[{DIM_RESERVATION}] (dummy lié à {s!r})"
        if head in ("full",) and "date" in sl:
            return f"[{sch}].[{DIM_DATE}] (dummy lié à {s!r})"

    return (
        f"[{sch}].[{FACT_RENTABILITE}] ou dimension — vérifier le SELECT pour la colonne {s!r}"
    )


__all__ = [
    "CLUSTERING_NUMERIC_DROP",
    "DIM_BENCHMARK",
    "DIM_BENEFICIARY",
    "DIM_COMPLAINT",
    "DIM_DATE",
    "DIM_EVENT",
    "DIM_FEEDBACK",
    "DIM_MARKETING_SPEND",
    "DIM_PROVIDER",
    "DIM_RESERVATION",
    "DIM_SERVICE_CATEGORY",
    "DIM_TENDANCE_MARCHE",
    "DIM_VENUE",
    "DIM_VISITORS",
    "FACT_PERFORMANCE",
    "FACT_RENTABILITE",
    "FACT_SATISFACTION",
    "SQL_LIST_TABLES",
    "SQL_ML_CLUSTERING_LOYALTY_BENEFICIARY",
    "SQL_ML_CLUSTERING_LOYALTY_PROVIDER",
    "SQL_ML_FINANCIAL_WIDE",
    "SQL_ML_PERFORMANCE_WIDE",
    "SQL_ML_TIME_SERIES_RESERVATIONS",
    "build_sql_ml_financial_wide",
    "build_sql_ml_performance_status_bridge",
    "infer_column_dw_source",
    "ml_financial_wide_sql_tables_lineage",
]

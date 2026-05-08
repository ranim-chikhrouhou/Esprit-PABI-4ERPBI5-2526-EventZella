# -*- coding: utf-8 -*-
"""
Génère EventZilla_Dashboards_Table2_DAX_Visuels_Detailles.{md,html,pdf}
à partir de EventZilla_Dashboards_Table2_Avec_Formules_Complet.md.

- DAX : aligné sur **Liste Des Kpis.pdf** (mesures officielles Power BI).
- Lignes **prédictives** (dash. 3 / Anticipation / « (projection) ») : fond coloré (ML ultérieur).
Réf. modèle : FilesPdf/MODELISATIONNFINALL.pdf (3 faits + dimensions).
"""
from __future__ import annotations

import html as html_lib
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DELIVERABLES = ROOT / "deliverables"
SRC_MD = DELIVERABLES / "EventZilla_Dashboards_Table2_Avec_Formules_Complet.md"


def _liste_kpis_pdf() -> Path:
    for candidate in (
        ROOT / "docs" / "references" / "Liste Des Kpis.pdf",
        ROOT / "Liste Des Kpis.pdf",
    ):
        if candidate.is_file():
            return candidate
    return ROOT / "docs" / "references" / "Liste Des Kpis.pdf"


LISTE_KPIS_PDF = _liste_kpis_pdf()
OUT_MD = DELIVERABLES / "EventZilla_Dashboards_Table2_DAX_Visuels_Detailles.md"
OUT_HTML = DELIVERABLES / "EventZilla_Dashboards_Table2_DAX_Visuels_Detailles.html"
OUT_PDF = DELIVERABLES / "EventZilla_Dashboards_Table2_DAX_Visuels_Detailles.pdf"

MODEL_NOTE = (
    "Référence modèle : **`FilesPdf/MODELISATIONNFINALL.pdf`** + **`Liste Des Kpis.pdf`** (DAX). "
    "Faits : `Fact_PerformanceCommerciale`, `Fact_RentabiliteFinanciere`, `Fact_SatisfactionClient`. "
    "Dimensions : `DimDate`, `DimReservation`, `DimEvent`, `DimBeneficiary`, `DimProvider`, `DimServiceCategory`, "
    "`DimBenchmarkPrice`, `DimVisitors`, `DimFeedback`, `DimComplaint`, `DimMarketingSpend`, `DimTendanceMarche`, `DimVenue` "
    "(selon votre import — noms à ajuster si différents dans Power BI)."
)


def norm_key(s: str) -> str:
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s.strip()).lower()
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = re.sub(r"\s+", " ", s)
    return s


def strip_md_bold(s: str) -> str:
    return re.sub(r"\*\*([^*]+)\*\*", r"\1", s).strip()


def is_predictive_row(cells: list[str]) -> bool:
    if len(cells) < 4:
        return False
    dash = cells[1].strip()
    if dash == "3":
        return True
    kpi = strip_md_bold(cells[3]).lower()
    titre = strip_md_bold(cells[2]).lower()
    if "(projection)" in kpi:
        return True
    if "anticipation" in titre:
        return True
    return False


def parse_src_table(path: Path) -> tuple[list[str], list[list[str]]]:
    text = path.read_text(encoding="utf-8")
    rows: list[list[str]] = []
    for line in text.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        if re.match(r"^\|\s*-+", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def inline_html(s: str) -> str:
    out: list[str] = []
    rest = s
    while rest:
        m = re.search(r"`([^`]+)`", rest)
        if not m:
            out.append(html_lib.escape(rest))
            break
        out.append(html_lib.escape(rest[: m.start()]))
        out.append("<code>" + html_lib.escape(m.group(1)) + "</code>")
        rest = rest[m.end() :]
    return "".join(out)


def try_edge_pdf(html_path: Path, pdf_path: Path) -> bool:
    for edge in (
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ):
        if not edge.is_file():
            continue
        url = "file:///" + str(html_path.resolve()).replace("\\", "/")
        try:
            subprocess.run(
                [str(edge), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", url],
                check=True,
                capture_output=True,
                timeout=120,
            )
            return pdf_path.is_file()
        except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
            continue
    return False


KPI_ALIASES_LOOKUP: dict[str, str] = {
    norm_key(a): norm_key(b)
    for a, b in {
        "taux de conversion (contexte)": "taux de conversion",
        "nombre total de réservations (contexte)": "nombre total de réservations",
        "chiffre d'affaires total (contexte rentabilité)": "chiffre d'affaires total",
        "part des réservations sous le marché": "part sous marché",
        "part des réservations alignées au marché": "part alignée marché",
        "part des réservations au-dessus du marché": "part au-dessus marché",
        "montant des commissions (agrégat)": "montant commissions (tnd)",
        "taux de commission sur réservation": "taux commission %",
        "taux de rétention des bénéficiaires": "taux rétention bénéficiaires %",
        "note moyenne des prestataires": "note moyenne prestataires",
        "taux de résolution des réclamations": "taux résolution réclamations",
        "taux de réclamations pour 100 réservations": "taux réclamations / 100 résa",
        "taux de réservation les jours fériés": "taux réservation jours fériés nombre réservations",
        "part des salles joignables": "part des salles joignables",
        "catégories déjà couvertes": "nb catégories catalogue",
        "catégories nouvelles vs couvertes": "nb catégories potentielles",
        "nombre de visiteurs": "nombre de visiteurs",
        "taux d'annulation": "taux annulation",
    }.items()
}


# DAX issus de Liste Des Kpis.pdf (reproductibles tels quels ; ajuster guillemets si besoin)
def _dax_blocks() -> dict[str, str]:
    return {
        "part des salles joignables": """% Salles joignables =
DIVIDE (
    COUNTROWS ( FILTER ( 'DimVenue', NOT ISBLANK ( 'DimVenue'[contact] ) ) ),
    COUNTROWS ( 'DimVenue' ),
    0
) * 100""",
        "bénéficiaires récurrents": """Bénéficiaires récurrents =
COUNTROWS (
    FILTER (
        VALUES ( DimBeneficiary[id_beneficiary] ),
        CALCULATE ( COUNT ( DimReservation[id_reservation] ), DimReservation[status] = "confirmed" ) > 1
    )
)""",
        "chiffre d’affaires total": """CA (TND) =
CALCULATE ( SUM ( Fact_RentabiliteFinanciere[final_price] ), DimReservation[status] = "confirmed" )""",
        "ca hors fériés": """CA hors fériés =
CALCULATE (
    SUM ( 'Fact_RentabiliteFinanciere'[final_price] ),
    'DimDate'[is_holiday] = FALSE ()
)""",
        "ca jours fériés": """CA jours fériés =
CALCULATE (
    SUM ( 'Fact_RentabiliteFinanciere'[final_price] ),
    'DimDate'[is_holiday] = TRUE ()
)""",
        "cac": """CAC (TND / bénéficiaire) =
DIVIDE ( SUM ( DimMarketingSpend[marketing_spend] ), SUM ( DimMarketingSpend[new_beneficiaries] ), 0 )""",
        "diversité des catégories réservées": """Diversité catégories % =
VAR Dist =
    CALCULATE ( DISTINCTCOUNT ( DimServiceCategory[category_name] ), Fact_PerformanceCommerciale )
VAR Tot = COUNTROWS ( ALL ( DimServiceCategory[category_name] ) )
RETURN
    DIVIDE ( Dist, Tot, 0 ) * 100""",
        "écart ca férié": """Écart CA férié = [CA jours fériés] - [CA hors fériés]""",
        "fréquence résa bénéficiaire": """Fréquence résa / bénéficiaire =
DIVIDE (
    CALCULATE (
        DISTINCTCOUNT ( Fact_PerformanceCommerciale[id_reservation] ),
        DimReservation[status] = "confirmed"
    ),
    DISTINCTCOUNT ( Fact_PerformanceCommerciale[id_beneficiary] ),
    BLANK ()
)""",
        "panier moyen (tnd)": """Panier moyen (TND) =
CALCULATE ( AVERAGE ( Fact_RentabiliteFinanciere[final_price] ), DimReservation[status] = "confirmed" )""",
        "ltv simplifié": """LTV simplifié = [Panier moyen (TND)] * [Fréquence résa / bénéficiaire]""",
        "montant commissions (tnd)": """Montant commissions (TND) =
CALCULATE (
    SUMX (
        'Fact_RentabiliteFinanciere',
        'Fact_RentabiliteFinanciere'[final_price] - 'Fact_RentabiliteFinanciere'[service_price]
    ),
    'DimReservation'[status] = "confirmed"
)""",
        "nb catégories catalogue": """Nb catégories catalogue =
DISTINCTCOUNT ( 'DimServiceCategory'[category_name] )""",
        "nb catégories potentielles": """Nb Catégories potentielles =
VAR CategoriesMarche = VALUES ( DimTendanceMarche[category_name] )
VAR CategoriesEventZilla = VALUES ( DimServiceCategory[category_name] )
RETURN
    COUNTROWS ( EXCEPT ( CategoriesMarche, CategoriesEventZilla ) )""",
        "nb réclamations": """Nb Réclamations = COUNTROWS ( 'DimComplaint' )""",
        "nombre de réservations": """Nombre de Réservations = COUNT ( DimReservation[id_reservation] )""",
        "nombre de salles suggérables": """Nombre de salles suggérables = COUNTROWS ( DimVenue )""",
        "nombre de visiteurs": """Nombre de Visiteurs = SUM ( Fact_PerformanceCommerciale[nb_visitors] )""",
        "note moyenne prestataires": """Note moyenne prestataires = AVERAGE ( 'Fact_SatisfactionClient'[rating] )""",
        "nps": """NPS =
VAR Total = COUNTROWS ( Fact_SatisfactionClient )
VAR Promoteurs = CALCULATE ( COUNTROWS ( Fact_SatisfactionClient ), Fact_SatisfactionClient[rating] >= 4 )
VAR Detracteurs = CALCULATE ( COUNTROWS ( Fact_SatisfactionClient ), Fact_SatisfactionClient[rating] <= 2 )
RETURN ( DIVIDE ( Promoteurs, Total, 0 ) - DIVIDE ( Detracteurs, Total, 0 ) ) * 100""",
        "part sous marché": """Part sous marché =
VAR TotalConfirme = CALCULATE ( COUNTROWS ( Fact_RentabiliteFinanciere ), DimReservation[status] = "confirmed" )
VAR SousMarche =
    CALCULATE (
        COUNTROWS ( Fact_RentabiliteFinanciere ),
        DimReservation[status] = "confirmed",
        Fact_RentabiliteFinanciere[final_price]
            < 0.85 * Fact_RentabiliteFinanciere[benchmark_avg_price]
    )
RETURN DIVIDE ( SousMarche, TotalConfirme, 0 )""",
        "part alignée marché": """Part alignée marché =
VAR TotalConfirme = CALCULATE ( COUNTROWS ( Fact_RentabiliteFinanciere ), DimReservation[status] = "confirmed" )
VAR Aligne =
    CALCULATE (
        COUNTROWS ( Fact_RentabiliteFinanciere ),
        DimReservation[status] = "confirmed",
        Fact_RentabiliteFinanciere[final_price] >= 0.85 * Fact_RentabiliteFinanciere[benchmark_avg_price]
            && Fact_RentabiliteFinanciere[final_price] <= 1.15 * Fact_RentabiliteFinanciere[benchmark_avg_price]
    )
RETURN DIVIDE ( Aligne, TotalConfirme, 0 )""",
        "part au-dessus marché": """Part au-dessus marché =
VAR TotalConfirme = CALCULATE ( COUNTROWS ( Fact_RentabiliteFinanciere ), DimReservation[status] = "confirmed" )
VAR AuDessus =
    CALCULATE (
        COUNTROWS ( Fact_RentabiliteFinanciere ),
        DimReservation[status] = "confirmed",
        Fact_RentabiliteFinanciere[final_price] > 1.15 * Fact_RentabiliteFinanciere[benchmark_avg_price]
    )
RETURN DIVIDE ( AuDessus, TotalConfirme, 0 )""",
        "rang opportunité marché": """Rang Opportunité Marché =
IF (
    [Nb Catégories potentielles] > 0,
    RANKX (
        ALL ( DimTendanceMarche[category_name] ),
        CALCULATE ( MAX ( DimTendanceMarche[event_count_observed] ) ),,
        DESC,
        DENSE
    ),
    BLANK ()
)""",
        "taux annulation": """Taux annulation =
DIVIDE (
    CALCULATE ( COUNT ( DimReservation[id_reservation] ), DimReservation[status] = "cancelled" ),
    COUNT ( DimReservation[id_reservation] ),
    0
)""",
        "taux commission %": """Taux commission % =
VAR MargeTotale = [Montant commissions (TND)]
VAR CA_Confirme =
    CALCULATE (
        SUM ( 'Fact_RentabiliteFinanciere'[final_price] ),
        'DimReservation'[status] = "confirmed"
    )
RETURN
    DIVIDE ( MargeTotale, CA_Confirme, 0 ) * 100""",
        "taux d'acceptation": """Taux d'acceptation =
DIVIDE (
    CALCULATE ( COUNT ( DimReservation[id_reservation] ), DimReservation[status] = "confirmed" ),
    COUNT ( DimReservation[id_reservation] ),
    0
)""",
        "taux de conversion": """Taux de conversion =
DIVIDE (
    SUM ( Fact_PerformanceCommerciale[nb_reservations_site] ),
    SUM ( Fact_PerformanceCommerciale[nb_visitors] ),
    0
)""",
        "taux réclamations / 100 résa": """Taux réclamations / 100 résa =
DIVIDE ( COUNTROWS ( 'DimComplaint' ), DISTINCTCOUNT ( 'DimReservation'[id_reservation] ), 0 ) * 100""",
        "taux réservation jours fériés nombre réservations": """Taux réservation jours fériés =
DIVIDE (
    CALCULATE ( [Nombre de Réservations], DimDate[is_holiday] = TRUE () ),
    [Nombre de Réservations],
    0
)""",
        "taux résolution réclamations": """Taux résolution réclamations =
DIVIDE (
    CALCULATE ( COUNTROWS ( DimComplaint ), DimComplaint[status] = "closed" ),
    COUNTROWS ( DimComplaint ),
    0
)""",
        "taux rétention bénéficiaires %": """Taux rétention bénéficiaires % =
DIVIDE ( [Bénéficiaires récurrents], DISTINCTCOUNT ( DimBeneficiary[id_beneficiary] ), BLANK () ) * 100""",
        "top n catégories à ajouter": """Top N catégories à ajouter =
VAR TopSelectionne = 5
RETURN
    IF ( [Rang Opportunité Marché] <= TopSelectionne, [Nb Catégories potentielles], BLANK () )""",
    }


_B = _dax_blocks()

# Paquets DAX (mesures dépendantes regroupées pour le document)
DAX_COMPOSITE: dict[str, str] = {
    "ltv": _B["panier moyen (tnd)"] + "\n\n" + _B["fréquence résa bénéficiaire"] + "\n\n" + _B["ltv simplifié"],
    "taux rétention bénéficiaires %": _B["bénéficiaires récurrents"] + "\n\n" + _B["taux rétention bénéficiaires %"],
    "taux commission %": _B["montant commissions (tnd)"] + "\n\n" + _B["taux commission %"],
    "impact des jours fériés sur le ca": _B["ca jours fériés"] + "\n\n" + _B["ca hors fériés"] + "\n\n" + _B["écart ca férié"],
    "impact jours fériés sur le ca": _B["ca jours fériés"] + "\n\n" + _B["ca hors fériés"] + "\n\n" + _B["écart ca férié"],
    "top n catégories à ajouter": _B["nb catégories potentielles"] + "\n\n" + _B["rang opportunité marché"] + "\n\n" + _B["top n catégories à ajouter"],
    "parts réservations sous / alignées / au-dessus marché": _B["part sous marché"]
    + "\n\n"
    + _B["part alignée marché"]
    + "\n\n"
    + _B["part au-dessus marché"],
}


def fix_taux_commission_typo(s: str) -> str:
    return s.replace("CA_confirme", "CA_Confirme")


# Clés = norm_key(...) pour cohérence avec les libellés du tableau (apostrophe ’ → ')
DAX_FROM_LISTE: dict[str, str] = {}
for _k, _v in _B.items():
    DAX_FROM_LISTE[norm_key(_k)] = fix_taux_commission_typo(_v)
for _k, _v in {
    "nombre total de réservations": _B["nombre de réservations"],
    "panier moyen": _B["panier moyen (tnd)"],
    "part réservations sous marché": _B["part sous marché"],
    "part réservations alignées marché": _B["part alignée marché"],
    "part réservations au-dessus marché": _B["part au-dessus marché"],
    "cac": _B["cac"],
    "nps": _B["nps"],
    "nombre de réclamations": _B["nb réclamations"],
}.items():
    DAX_FROM_LISTE[norm_key(_k)] = fix_taux_commission_typo(_v)
for _k, _v in DAX_COMPOSITE.items():
    DAX_FROM_LISTE[norm_key(_k)] = fix_taux_commission_typo(_v)


_VISUAL_RAW: dict[str, str] = {
    "vue d’ensemble visiteurs → réservations": "**Entonnoir :** étapes = `Nombre de Visiteurs` → `SUM(nb_reservations_site)` → `[Nombre de Réservations]` ou confirmées ; valeurs absolues sur une axe unique.",
    "répartition des motifs de réclamation (si données)": "**Barres :** **X** = `DimComplaint[subject]` ; **Y** = `COUNTROWS(DimComplaint)`. Pareto = ligne cumul %.",
    "distribution des scores de recommandation": "**Histogramme :** **X** = `Fact_SatisfactionClient[rating]` (1–5) ; **Y** = nombre de lignes.",
    "lien prix vs plaintes (analyse)": "**Nuage de points :** agrégé par `DimBeneficiary` ; **X** = panier moyen (`Fact_RentabiliteFinanciere`) ; **Y** = `COUNTROWS(DimComplaint)`.",
    "répartition des salles par gouvernorat": "**Barres / carte :** grouper `DimVenue` par attribut géographique (colonne région/gouvernorat du fichier venues).",
}
VISUAL_DETAIL = {norm_key(k): v for k, v in _VISUAL_RAW.items()}


def lookup_dax(kpi: str) -> str:
    k = norm_key(kpi)
    k = KPI_ALIASES_LOOKUP.get(k, k)
    if k in DAX_FROM_LISTE:
        return DAX_FROM_LISTE[k]
    if k in VISUAL_DETAIL:
        return (
            "(Pas de mesure DAX unique dans Liste Des Kpis.pdf — vue composite ou analyse ; "
            "voir colonne « Configuration visuel Power BI ».)"
        )
    return (
        "— Voir `Liste Des Kpis.pdf` ou compléter selon le modèle ; KPI non mappé automatiquement."
    )


def lookup_visual(kpi: str) -> str:
    k = norm_key(kpi)
    if k in VISUAL_DETAIL:
        return VISUAL_DETAIL[k]
    return (
        "**Carte KPI** ou **courbes :** **X** = `DimDate[full_date]` ; **Y** = mesure concernée. Segmentation : `DimServiceCategory[category_name]` ou `DimEvent[event_type]`."
    )


def predictive_dax(formule_cdc: str, kpi_libelle: str) -> str:
    return (
        f"-- ═══ KPI PRÉDICTIF (phase ML — non prévu dans les dashboards Power BI livrés actuellement) ═══\n"
        f"-- Libellé : {kpi_libelle}\n"
        f"-- Référence CDC / Tableau 1 (description ou base série) : {formule_cdc.strip()}\n"
        f"-- Implémentation : exporter séries vers ML, réimporter prédictions (ex. table Fact_PredictionML liée à DimDate) ou service Azure ; ne pas dupliquer ici la logique statistique en DAX.\n"
        f"[{kpi_libelle}] (prédictif ML) = BLANK ()"
    )


def main():
    if not SRC_MD.is_file():
        raise SystemExit(
            f"Fichier manquant : {SRC_MD} — exécuter d'abord : python scripts/build_dashboards_table2_with_formulas.py"
        )

    headers, data = parse_src_table(SRC_MD)
    if not headers or len(headers) < 6:
        raise SystemExit("Tableau source invalide (colonnes attendues : 6).")

    note_liste = (
        f"**DAX officiels** : `Liste Des Kpis.pdf` présent : {LISTE_KPIS_PDF.is_file()}."
        if LISTE_KPIS_PDF.is_file()
        else "**Attention :** `Liste Des Kpis.pdf` introuvable dans le dossier ; le script utilise la copie intégrée des mesures."
    )

    new_headers = headers + ["Périmètre", "Mesure DAX (Liste Des Kpis + dépendances)", "Configuration visuel Power BI (détail)"]

    out_lines: list[list[str]] = []
    for row in data:
        if len(row) < 6:
            continue
        kpi = row[3]
        formule_cdc = row[4]
        pred = is_predictive_row(row)
        if pred:
            dax = predictive_dax(formule_cdc, strip_md_bold(kpi))
            perim = "Prédictif (ML ultérieur)"
            vis = (
                "**Lignes violettes dans le PDF/HTML :** conserver pour documentation ; **ne pas** placer tel quel sur les dashboards opérationnels. "
                "Visuel cible plus tard : courbe historique + série prévue (données ML)."
            )
        else:
            dax = lookup_dax(kpi)
            perim = "Dashboard (standard)"
            vis = lookup_visual(kpi)

        dax_one = re.sub(r"\s+", " ", dax).strip() if pred else dax.replace("\n", " ").strip()
        out_lines.append(row + [perim, dax_one, vis])

    md_lines = [
        "# EventZilla — Dashboards : DAX (`Liste Des Kpis.pdf`) + visuels (complet, prédictifs identifiés)",
        "",
        MODEL_NOTE,
        "",
        note_liste,
        "",
        "**Légende :** colonne **Périmètre** = `Dashboard (standard)` ou **`Prédictif (ML ultérieur)`** (lignes dash. 3 / « Anticipation » / « (projection) »).",
        "",
        "| " + " | ".join(new_headers) + " |",
        "|" + "|".join(["---"] * len(new_headers)) + "|",
    ]
    for r in out_lines:
        esc = [c.replace("|", "\\|") for c in r]
        md_lines.append("| " + " | ".join(esc) + " |")

    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    css = """
    body { font-family: 'Segoe UI', system-ui, sans-serif; font-size: 7.5px; margin: 14px; color: #1e293b; }
    h1 { font-size: 14px; color: #0f766e; }
    .note { background: #f0fdfa; border-left: 4px solid #0d9488; padding: 10px; font-size: 8.5px; margin-bottom: 12px; }
    .legend { background: #faf5ff; border: 1px solid #c4b5fd; padding: 8px 10px; border-radius: 8px; margin-bottom: 12px; font-size: 8px; }
    table { border-collapse: collapse; width: 100%; table-layout: fixed; }
    th { background: #0f766e; color: #fff; padding: 5px 3px; text-align: left; font-size: 6.5px; vertical-align: top; }
    td { border: 1px solid #cbd5e1; padding: 4px 3px; vertical-align: top; word-wrap: break-word; font-size: 6.5px; }
    tr.row-standard:nth-child(even) td { background: #f8fafc; }
    tr.row-predictive td { background: #ede9fe !important; border-color: #a78bfa; color: #4c1d95; }
    tr.row-predictive th { background: #7c3aed; }
    pre, .dax { font-family: Consolas, monospace; font-size: 6px; white-space: pre-wrap; margin: 0; }
    @media print {
      body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
      tr.row-predictive td { background: #ede9fe !important; }
    }
    """
    th = "".join(f"<th>{html_lib.escape(h)}</th>" for h in new_headers)
    trs = []
    n_base = len(headers)
    for r in out_lines:
        pred = r[n_base] == "Prédictif (ML ultérieur)"
        trc = "row-predictive" if pred else "row-standard"
        cells = []
        for i, c in enumerate(r):
            if i >= n_base + 1:
                cells.append(f'<td><pre class="dax">{html_lib.escape(c)}</pre></td>')
            elif i == n_base:
                cells.append(f"<td><strong>{html_lib.escape(c)}</strong></td>")
            else:
                cells.append(f"<td>{inline_html(c)}</td>")
        trs.append(f'<tr class="{trc}">' + "".join(cells) + "</tr>")

    legend = (
        "<div class='legend'><strong>Prédictif (fond violet) :</strong> réservé à une phase "
        "<strong>Machine Learning</strong> — non inclus dans les rapports Power BI livrés maintenant. "
        "Les mesures DAX affichent <code>BLANK()</code> ou une table de prévisions à brancher plus tard.</div>"
    )
    html = f"""<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'><title>EventZilla DAX & visuels</title>
    <style>{css}</style></head><body>
    <h1>Tableau dashboards — DAX (Liste Des Kpis) + visuels</h1>
    <div class='note'>{html_lib.escape(MODEL_NOTE)}</div>
    {legend}
    <table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table>
    </body></html>"""
    OUT_HTML.write_text(html, encoding="utf-8")

    if try_edge_pdf(OUT_HTML, OUT_PDF):
        print("PDF OK:", OUT_PDF)
    else:
        print("PDF : ouvrir le HTML et Imprimer.")

    print("OK:", OUT_MD)
    print("OK:", OUT_HTML)


if __name__ == "__main__":
    main()

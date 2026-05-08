# -*- coding: utf-8 -*-
"""
Fusionne le Tableau 2 (dashboards par décideur) avec les formules du Tableau 1.
Exclut les KPIs prédictifs (dashboards « Anticipation », projections / dash. 3).
Génère Markdown, HTML et PDF (Edge headless si disponible).
"""
from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_EVENTZILLA = ROOT / "docs" / "eventzilla"
DELIVERABLES = ROOT / "deliverables"
MD_SOURCE = DOCS_EVENTZILLA / "EventZilla_Dashboards_KPIs_Objectifs.md"
OUT_MD = DELIVERABLES / "EventZilla_Dashboards_Table2_Avec_Formules_Sans_Predictif.md"
OUT_HTML = DELIVERABLES / "EventZilla_Dashboards_Table2_Avec_Formules_Sans_Predictif.html"
OUT_PDF = DELIVERABLES / "EventZilla_Dashboards_Table2_Avec_Formules_Sans_Predictif.pdf"
OUT_MD_COMPLET = DELIVERABLES / "EventZilla_Dashboards_Table2_Avec_Formules_Complet.md"

# Lignes overview (dash. 0) — communes Marketing / Financier / Relation Client
OVERVIEW_TITLE = "**Overview général — santé business et expérience client**"
OVERVIEW_DECIDEUR = "Vue transversale (Marketing + Financier + Relation Client)"
OVERVIEW_ROWS: list[tuple[str, str, str]] = [
    ("Chiffre d’affaires total", "`SUM(final_price)`", "**Carte KPI** (grand format) + variation vs période précédente"),
    ("Nombre total de réservations", "`COUNT(id_reservation)`", "**Carte KPI** + **courbe sparkline**"),
    ("Taux de conversion", "`(COUNT(reservations_confirmées) / SUM(visitors)) × 100`", "**Carte KPI** + **courbe**"),
    ("Panier moyen", "`AVG(final_price)`", "**Carte KPI** + **indicateur de tendance**"),
    ("CAC", "`SUM(marketing_spend) / SUM(new_beneficiaries)`", "**Carte KPI** + **jauge** (cible)"),
    ("Taux de rétention des bénéficiaires", "`(COUNT(beneficiaries_recurrents) / COUNT(total_beneficiaries)) × 100`", "**Carte KPI** + **courbe**"),
    ("Taux d’annulation", "`(COUNT(reservations_annulées) / COUNT(total_reservations)) × 100`", "**Carte KPI** + **courbe**"),
    ("NPS", "`% promoteurs - % détracteurs`", "**Carte KPI** + **jauge**"),
    ("Taux de résolution des réclamations", "`(COUNT(closed_complaints) / COUNT(total_complaints)) × 100`", "**Carte KPI** + **barre de progression**"),
]

# Alias : libellé tel qu’au Tableau 2 → libellé KPI du Tableau 1
KPI_ALIASES = {
    "part des réservations sous le marché": "Part réservations sous marché",
    "part des réservations alignées au marché": "Part réservations alignées marché",
    "part des réservations au-dessus du marché": "Part réservations au-dessus marché",
    "taux de rétention des bénéficiaires": "Taux de rétention bénéficiaires",
    "cac": "CAC (coût d’acquisition client)",
    "ltv": "LTV (lifetime value)",
    "note moyenne des prestataires": "Note moyenne prestataires",
    "montant des commissions (agrégat)": "Commissions",
    "chiffre d’affaires total (contexte rentabilité)": "Chiffre d’affaires total",
    "nombre total de réservations (contexte)": "Nombre total de réservations",
    "taux de conversion (contexte)": "Taux de conversion",
    "impact des jours fériés sur le ca": "Impact jours fériés sur CA",
    "impact jours fériés sur le ca": "Impact jours fériés sur CA",
    "taux de résolution des réclamations": "Taux de résolution réclamations",
}

# Lignes « analyse / composite » sans KPI unique dans KPIs_FINAL
FORMULA_OVERRIDES = {
    "vue d’ensemble visiteurs → réservations": (
        "Entonnoir : étapes basées sur `SUM(visitors)` et `COUNT(id_reservation)` "
        "(confirmées) — vue composite, pas une mesure DAX unique."
    ),
    "répartition des motifs de réclamation (si données)": (
        "`COUNT(id_complaint) GROUP BY motif` (attribut « motif » à confirmer au CDC)."
    ),
    "distribution des scores de recommandation": (
        "Répartition empirique des scores (histogramme) ; lien avec NPS : "
        "`% promoteurs − % détracteurs` ; promoteur `rating ≥ 4`, détracteur `rating ≤ 2`."
    ),
    "lien prix vs plaintes (analyse)": (
        "Croisement analytique `final_price` × nombre de réclamations par segment "
        "(matrice / nuage de points), pas une formule KPI unique."
    ),
    "parts réservations sous / alignées / au-dessus marché": (
        "Sous : `(COUNT(final_price < 0.85*benchmark) / COUNT(total)) × 100` — "
        "Alignées : `(COUNT(0.85*benchmark ≤ final_price ≤ 1.15*benchmark) / COUNT(total)) × 100` — "
        "Au-dessus : `(COUNT(final_price > 1.15*benchmark) / COUNT(total)) × 100`."
    ),
    "catégories déjà couvertes": (
        "Complement / dénombrement des catégories plateforme — "
        "à lier au référentiel catégories (*précision CDC*)."
    ),
}


def norm_key(s: str) -> str:
    s = s.strip().strip("*").lower()
    s = re.sub(r"\s+", " ", s)
    return s


def strip_md_bold(s: str) -> str:
    return re.sub(r"\*\*([^*]+)\*\*", r"\1", s).strip()


def parse_md_table(section_text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in section_text.splitlines():
        line = line.rstrip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|\s*:?-+", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def extract_section(raw: str, header: str) -> str:
    i = raw.find(header)
    if i < 0:
        raise ValueError(f"Section introuvable: {header}")
    start = raw.find("\n", i) + 1
    j = raw.find("\n## ", start)
    if j < 0:
        return raw[start:]
    return raw[start:j]


def build_formula_map(table1_rows: list[list[str]], *, include_predictive: bool = False) -> dict[str, str]:
    """KPI -> formule Tableau 1 ; si include_predictive, inclut aussi les blocs PRÉDICTIF."""
    m: dict[str, str] = {}
    for cells in table1_rows[1:]:  # skip header
        if len(cells) < 5:
            continue
        gimsi = cells[1]
        if not include_predictive and ("PRÉDICTIF" in gimsi.upper() or "PRÉDICTIF" in gimsi):
            continue
        kpi = cells[3]
        formula = cells[4]
        if kpi not in m:
            m[kpi] = formula
        elif m[kpi] != formula and len(formula) > len(m[kpi]):
            m[kpi] = formula
    return m


def is_predictive_row(cells: list[str]) -> bool:
    if len(cells) < 5:
        return True
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


def resolve_formula(kpi_raw: str, formula_by_kpi: dict[str, str]) -> str:
    kpi_clean = strip_md_bold(kpi_raw)
    nk = norm_key(kpi_clean)
    if nk in FORMULA_OVERRIDES:
        return FORMULA_OVERRIDES[nk]
    canon = KPI_ALIASES.get(nk, kpi_clean)
    if norm_key(canon) in FORMULA_OVERRIDES:
        return FORMULA_OVERRIDES[norm_key(canon)]
    if canon in formula_by_kpi:
        return formula_by_kpi[canon]
    # derniers essais : correspondance insensible à la casse
    for k, f in formula_by_kpi.items():
        if norm_key(k) == norm_key(canon):
            return f
    return "— (à rapprocher du Tableau 1 / CDC)"


def overview_lines(formula_by_kpi: dict[str, str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for kpi, _formule_sql, visuel in OVERVIEW_ROWS:
        f = resolve_formula(kpi, formula_by_kpi)
        rows.append(
            [OVERVIEW_DECIDEUR, "0", OVERVIEW_TITLE, kpi, f, visuel]
        )
    return rows


def inline_format(s: str) -> str:
    if not s:
        return ""
    out: list[str] = []
    rest = s
    while rest:
        m = re.search(r"\*\*([^*]+)\*\*|`([^`]+)`", rest)
        if not m:
            out.append(html.escape(rest))
            break
        out.append(html.escape(rest[: m.start()]))
        if m.group(1) is not None:
            out.append("<strong>" + html.escape(m.group(1)) + "</strong>")
        else:
            out.append("<code>" + html.escape(m.group(2)) + "</code>")
        rest = rest[m.end() :]
    return "".join(out)


def table_to_html(headers: list[str], data_rows: list[list[str]]) -> str:
    thead = "<thead><tr>" + "".join(f"<th>{inline_format(h)}</th>" for h in headers) + "</tr></thead>"
    body: list[str] = []
    for i, row in enumerate(data_rows, 1):
        alt = "row-alt" if i % 2 == 0 else ""
        tds = "".join(f"<td>{inline_format(c)}</td>" for c in row)
        body.append(f'<tr class="{alt}">{tds}</tr>')
    return f'<table class="data-table">{thead}<tbody>{"".join(body)}</tbody></table>'


def try_edge_pdf(html_path: Path, pdf_path: Path) -> bool:
    edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    if not edge.is_file():
        edge = Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")
    if not edge.is_file():
        return False
    url = "file:///" + str(html_path.resolve()).replace("\\", "/")
    try:
        subprocess.run(
            [
                str(edge),
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_path}",
                url,
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return pdf_path.is_file()
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return False


def main() -> int:
    raw = MD_SOURCE.read_text(encoding="utf-8")
    t1_text = extract_section(raw, "## Tableau 1 —")
    t2_text = extract_section(raw, "## Tableau 2 —")

    t1_rows = parse_md_table(t1_text)
    t2_rows = parse_md_table(t2_text)
    formula_by_kpi = build_formula_map(t1_rows, include_predictive=False)
    formula_by_kpi_pred = build_formula_map(t1_rows, include_predictive=True)

    header2 = t2_rows[0] if t2_rows else []
    # Nouvelles colonnes : insérer Formule après KPI
    # Décideur | Dash | Titre | KPI | Visuel -> + Formule après KPI
    new_headers = header2[:4] + ["Formule (KPIs_FINAL / CDC)"] + header2[4:]

    out_rows: list[list[str]] = overview_lines(formula_by_kpi)
    for cells in t2_rows[1:]:
        if is_predictive_row(cells):
            continue
        kpi = cells[3]
        formule = resolve_formula(kpi, formula_by_kpi)
        new_line = cells[:4] + [formule] + cells[4:]
        out_rows.append(new_line)

    out_rows_complet: list[list[str]] = overview_lines(formula_by_kpi_pred)
    for cells in t2_rows[1:]:
        if len(cells) < 5:
            continue
        kpi = cells[3]
        formule = resolve_formula(kpi, formula_by_kpi_pred)
        new_line = cells[:4] + [formule] + cells[4:]
        out_rows_complet.append(new_line)

    # Markdown
    md_lines = [
        "# EventZilla — Dashboards par décideur avec formules (hors KPIs prédictifs)",
        "",
        "**Périmètre :** même contenu que le Tableau 2 du document « Improved », "
        "avec une colonne **Formule** issue du Tableau 1 (KPIs_FINAL).",
        "",
        "**Exclusions :** les lignes des dashboards « **Anticipation** » (projections) sont retirées ; "
        "elles seront traitées ultérieurement dans la partie **machine learning**.",
        "",
        "| " + " | ".join(new_headers) + " |",
        "|" + "|".join(["---"] * len(new_headers)) + "|",
    ]
    for r in out_rows:
        md_lines.append("| " + " | ".join(r) + " |")

    md_lines += [
        "",
        "---",
        "",
        "*Note : commission et Catégories couvertes — formules à finaliser avec le CDC (cf. document source).*",
    ]
    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    md_complet = [
        "# EventZilla — Dashboards par décideur avec formules (**complet**, y compris KPIs prédictifs)",
        "",
        "**Contenu :** tableau 2 + **Overview** (dash. 0) + **tous** les dashboards y compris « Anticipation » (dash. 3). "
        "Les lignes prédictives sont destinées à une **phase ML** ultérieure ; elles ne sont pas requises dans les rapports Power BI livrés immédiatement.",
        "",
        "| " + " | ".join(new_headers) + " |",
        "|" + "|".join(["---"] * len(new_headers)) + "|",
    ]
    for r in out_rows_complet:
        md_complet.append("| " + " | ".join(r) + " |")
    md_complet += [
        "",
        "---",
        "",
        "*KPIs prédictifs : formules issues du Tableau 1 (PRÉDICTIF) ; DAX de prévoir via export ML ou table de prévisions.*",
    ]
    OUT_MD_COMPLET.write_text("\n".join(md_complet), encoding="utf-8")

    css = """
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700&display=swap');
    :root { --teal-dark:#0f766e; --teal:#0d9488; --accent:#f97316; --ink:#0f172a; --bg:#f1f5f9; --card:#fff; --border:#e2e8f0; }
    * { box-sizing:border-box; }
    body { font-family:'DM Sans',system-ui,sans-serif; color:var(--ink); background:var(--bg); margin:0; font-size:10pt; line-height:1.45; }
    .cover { min-height:100vh; display:flex; flex-direction:column; justify-content:center; padding:40px 48px;
      background:linear-gradient(125deg,#0f766e 0%,#0d9488 40%,#2dd4bf 100%); color:#fff; page-break-after:always; }
    .cover h1 { font-size:2rem; margin:0 0 12px; }
    .cover p { max-width:620px; opacity:0.95; font-size:1.02rem; }
    .wrap { max-width:1280px; margin:0 auto; padding:24px 18px 48px; }
    .notice { background:linear-gradient(90deg,#fff7ed,#fff); border-left:4px solid var(--accent); padding:14px 18px; border-radius:0 10px 10px 0; margin-bottom:22px; font-size:0.92rem; color:#44403c; }
    .section { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:20px 18px 24px; margin-bottom:22px; }
    .section h2 { margin:0 0 12px; color:var(--teal-dark); font-size:1.15rem; border-bottom:2px solid #99f6e4; padding-bottom:8px; }
    .data-table { width:100%; border-collapse:collapse; font-size:6.8pt; margin-top:12px; }
    .data-table th { background:linear-gradient(180deg,#0f766e,#0d9488); color:#fff; font-weight:600; padding:8px 5px; text-align:left; border:1px solid #0d9488; }
    .data-table td { border:1px solid var(--border); padding:5px 5px; vertical-align:top; }
    .data-table tr.row-alt td { background:#f8fafc; }
    @media print { body { -webkit-print-color-adjust:exact; print-color-adjust:exact; background:#fff; } .data-table { font-size:6pt; } .data-table thead { display:table-header-group; } }
    """

    html_doc = [
        "<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>EventZilla — Dashboards & formules (sans prédictif)</title>",
        f"<style>{css}</style></head><body>",
        "<div class='cover'><h1>Dashboards par décideur</h1>",
        "<p>Chaque ligne : décideur, n° de rapport, titre, KPI ou vue analytique, <strong>formule concrète</strong>, visuel Power BI. "
        "Les KPIs <strong>prédictifs</strong> (anticipation / projections) sont exclus — prévus pour la phase <strong>ML</strong> ultérieure.</p></div>",
        "<div class='wrap'>",
        "<div class='notice'><strong>Périmètre Power BI :</strong> deux rapports par décideur (les anciens tableaux de bord « anticipation » ne sont pas livrés dans ce lot).</div>",
        '<div class="section"><h2>Tableau — Décideur, dashboard, KPI, formule, visuel</h2>',
        table_to_html(new_headers, out_rows),
        "</div></div></body></html>",
    ]
    OUT_HTML.write_text("".join(html_doc), encoding="utf-8")

    if try_edge_pdf(OUT_HTML, OUT_PDF):
        print("PDF:", OUT_PDF)
    else:
        print("PDF non généré (Edge introuvable) — ouvrir le HTML et Imprimer > PDF.")

    print("OK:", OUT_MD)
    print("OK:", OUT_MD_COMPLET)
    print("OK:", OUT_HTML)
    return 0


if __name__ == "__main__":
    sys.exit(main())

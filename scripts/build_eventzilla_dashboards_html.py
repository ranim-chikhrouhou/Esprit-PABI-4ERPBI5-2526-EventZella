# -*- coding: utf-8 -*-
"""Build styled HTML from EventZilla_Dashboards_KPIs_Objectifs.md — open in browser, Print to PDF."""
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_EVENTZILLA = ROOT / "docs" / "eventzilla"
DELIVERABLES = ROOT / "deliverables"
MD_PATH = DOCS_EVENTZILLA / "EventZilla_Dashboards_KPIs_Objectifs.md"
OUT_HTML = DELIVERABLES / "EventZilla_Dashboards_KPIs_Objectifs.html"


def inline_format(s):
    """Échappe le HTML et applique **gras** et `code` (segments simples)."""
    if not s:
        return ""
    out = []
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


def md_table_to_html(table_lines):
    rows = []
    for line in table_lines:
        line = line.rstrip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|\s*:?-+", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return ""
    thead = "<thead><tr>" + "".join(f"<th>{inline_format(c)}</th>" for c in rows[0]) + "</tr></thead>"
    body = []
    for i, cells in enumerate(rows[1:], 1):
        alt = "row-alt" if i % 2 == 0 else ""
        body.append(
            f'<tr class="{alt}">' + "".join(f"<td>{inline_format(c)}</td>" for c in cells) + "</tr>"
        )
    return f'<table class="data-table">{thead}<tbody>' + "\n".join(body) + "</tbody></table>"


def extract_table_after_header(chunk_lines):
    """Return (paragraph_html, table_html) from lines starting after title line."""
    paras = []
    table_lines = []
    mode = "para"
    for line in chunk_lines:
        if line.strip().startswith("|"):
            mode = "table"
        if mode == "table":
            if line.strip().startswith("|"):
                table_lines.append(line)
        else:
            s = line.strip()
            if not s or s.startswith("|") or s == "---":
                continue
            if s.startswith("**") and s.endswith("**") and s.count("**") == 2:
                paras.append(f"<p class='lead'>{inline_format(s)}</p>")
            elif s.startswith("*") and s.endswith("*") and not s.startswith("**"):
                paras.append(f"<p class='muted'><em>{html.escape(s.strip('*'))}</em></p>")
            elif s.startswith("- "):
                paras.append(("bullet", s[2:]))
            else:
                paras.append(("text", s))
    # merge bullets
    out = []
    bullets = []
    for p in paras:
        if isinstance(p, str):
            out.append(p)
        elif p[0] == "bullet":
            bullets.append(p[1])
        elif p[0] == "text":
            if bullets:
                out.append("<ul>" + "".join(f"<li>{inline_format(b)}</li>" for b in bullets) + "</ul>")
                bullets = []
            out.append(f"<p>{inline_format(p[1])}</p>")
    if bullets:
        out.append("<ul>" + "".join(f"<li>{inline_format(b)}</li>" for b in bullets) + "</ul>")
    tbl = md_table_to_html(table_lines)
    return "".join(out), tbl


def main():
    raw = MD_PATH.read_text(encoding="utf-8")
    # Drop leading # title — we'll use cover
    raw = re.sub(r"^# .+\n+", "", raw, count=1, flags=re.M)

    chunks = re.split(r"\n(?=## )", raw.strip())
    intro_chunks = []
    tableau1 = None
    tableau2 = None
    footer_chunks = []

    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        first = ch.split("\n", 1)[0]
        if first.startswith("## Tableau 1"):
            tableau1 = ch
        elif first.startswith("## Tableau 2"):
            tableau2 = ch
        elif first.startswith("## Export PDF") or first.startswith("## "):
            if "Export PDF" in first or "formules non couvertes" in ch:
                footer_chunks.append(ch)
            else:
                intro_chunks.append(ch)
        else:
            intro_chunks.insert(0, ch)

    css = """
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700&display=swap');
    :root {
      --teal: #0d9488;
      --teal-dark: #0f766e;
      --accent: #f97316;
      --ink: #0f172a;
      --muted: #64748b;
      --bg: #f1f5f9;
      --card: #ffffff;
      --border: #e2e8f0;
    }
    * { box-sizing: border-box; }
    body { font-family: 'DM Sans', system-ui, sans-serif; color: var(--ink); background: var(--bg); margin: 0; font-size: 10.5pt; line-height: 1.45; }
    .cover {
      min-height: 100vh; display: flex; flex-direction: column; justify-content: center;
      padding: 48px 56px;
      background: linear-gradient(125deg, #0f766e 0%, #0d9488 40%, #2dd4bf 100%);
      color: #fff; page-break-after: always;
    }
    .cover .badge { display: inline-block; background: rgba(255,255,255,0.22); padding: 8px 16px; border-radius: 999px; font-size: 0.78rem; margin-bottom: 18px; letter-spacing: 0.02em; }
    .cover h1 { font-size: 2.5rem; margin: 0 0 14px; font-weight: 700; letter-spacing: -0.03em; }
    .cover .sub { font-size: 1.12rem; max-width: 560px; opacity: 0.95; line-height: 1.55; }
    .cover .meta { margin-top: 36px; font-size: 0.9rem; opacity: 0.88; line-height: 1.6; }
    .wrap { max-width: 1180px; margin: 0 auto; padding: 28px 20px 56px; }
    .toc {
      background: var(--card); border-radius: 14px; border: 1px solid var(--border);
      padding: 22px 26px; margin-bottom: 28px; box-shadow: 0 4px 24px rgba(15,118,110,0.08);
      page-break-after: always;
    }
    .toc h2 { margin: 0 0 14px; color: var(--teal-dark); font-size: 1.15rem; }
    .toc ol { margin: 0; padding-left: 22px; color: #475569; }
    .section {
      background: var(--card); border-radius: 14px; border: 1px solid var(--border);
      padding: 22px 24px 26px; margin-bottom: 26px;
      box-shadow: 0 2px 16px rgba(15,23,42,0.04);
    }
    .section h2 {
      margin: 0 0 14px; font-size: 1.22rem; color: var(--teal-dark);
      padding-bottom: 10px; border-bottom: 2px solid #99f6e4;
    }
    .lead { font-weight: 600; margin: 10px 0; color: var(--ink); }
    .muted { color: var(--muted); font-size: 0.95rem; margin: 6px 0 14px; }
    p { margin: 8px 0; color: #334155; }
    ul { margin: 8px 0 14px; padding-left: 22px; color: #334155; }
    .data-table { width: 100%; border-collapse: collapse; font-size: 8.2pt; margin-top: 14px; }
    .data-table th {
      background: linear-gradient(180deg, #0f766e, #0d9488); color: #fff; font-weight: 600;
      padding: 9px 7px; text-align: left; border: 1px solid #0d9488;
    }
    .data-table td { border: 1px solid var(--border); padding: 6px 7px; vertical-align: top; }
    .data-table tr.row-alt td { background: #f8fafc; }
    .footer-note {
      margin-top: 20px; padding: 14px 16px; background: linear-gradient(90deg, #fff7ed, #fff);
      border-left: 4px solid var(--accent); border-radius: 0 10px 10px 0; font-size: 0.88rem; color: #57534e;
    }
    @media print {
      body { -webkit-print-color-adjust: exact; print-color-adjust: exact; background: #fff; }
      .wrap { padding: 12px; max-width: 100%; }
      .toc { page-break-after: always; }
      .data-table { font-size: 7.2pt; }
      .data-table thead { display: table-header-group; }
      .data-table tr { page-break-inside: avoid; }
    }
    """

    parts = []
    parts.append("<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>")
    parts.append("<title>EventZilla — Objectifs, KPIs & Dashboards</title>")
    parts.append(f"<style>{css}</style></head><body>")
    parts.append("""<div class="cover">
      <div class="badge">BI EventZilla • Objectifs & KPIs • Power BI</div>
      <h1>Objectifs, KPIs & dashboards</h1>
      <p class="sub">Document de synthèse : chaîne objectifs → indicateurs → formules (KPIs_FINAL), puis cartographie des rapports par décideur avec visuels Power BI suggérés.</p>
      <div class="meta">Impression recommandée : navigateur → Ctrl+P → Enregistrer au format PDF (activer « Graphiques d’arrière-plan »).<br>Fichier source : EventZilla_Dashboards_KPIs_Objectifs.md</div>
    </div>""")

    parts.append("<div class='wrap'>")
    parts.append("""<div class="toc"><h2>Sommaire</h2><ol>
      <li>Introduction et structure</li>
      <li>Tableau 1 — Objectifs globaux → GIMSI → objectifs opérationnels → KPI → formule</li>
      <li>Tableau 2 — Dashboards (3 par décideur), KPI et visuel associé</li>
      <li>Notes d’export</li>
    </ol></div>""")

    # Intro sections (everything before Tableau 1)
    for ch in intro_chunks:
        lines = ch.splitlines()
        if lines and lines[0].startswith("## "):
            title = lines[0][3:].strip()
            body_lines = lines[1:]
        else:
            title = "Contexte et sources"
            body_lines = lines[:]
        para_html, _ = extract_table_after_header(body_lines)
        parts.append(f'<div class="section"><h2>{html.escape(title)}</h2>{para_html}</div>')

    for name, block in [("Tableau 1", tableau1), ("Tableau 2", tableau2)]:
        if not block:
            continue
        lines = block.splitlines()
        title = lines[0][3:].strip() if lines[0].startswith("## ") else name
        body_lines = lines[1:]
        para_html, tbl_html = extract_table_after_header(body_lines)
        parts.append(f'<div class="section"><h2>{html.escape(title)}</h2>{para_html}{tbl_html}</div>')

    for ch in footer_chunks:
        lines = ch.splitlines()
        title = lines[0][3:].strip() if lines[0].startswith("## ") else "Note"
        rest = "\n".join(lines[1:])
        parts.append(f'<div class="section"><h2>{html.escape(title)}</h2><p>{html.escape(rest.strip())}</p></div>')

    parts.append("""<div class="footer-note"><strong>Rappel :</strong> la formule « taux de commission » est incomplète dans KPIs_FINAL.pdf — à finaliser avec le CDC. Les projections utilisent une logique de prévision sur séries historiques (paramétrage BI / ML).</div>""")
    parts.append("</div></body></html>")

    OUT_HTML.write_text("".join(parts), encoding="utf-8")
    print("OK:", OUT_HTML)
    return 0


if __name__ == "__main__":
    sys.exit(main())

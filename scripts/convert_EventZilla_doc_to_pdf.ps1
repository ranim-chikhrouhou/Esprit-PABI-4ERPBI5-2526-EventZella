#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Convertit EventZilla_Dashboards_KPIs_Objectifs.md en PDF via Pandoc.

.PREREQ
  - Pandoc : https://pandoc.org/installing.html
  - Moteur PDF : option A) pdflatex (MiKTeX / TeX Live) ; option B) wkhtmltopdf ; option C) --pdf-engine=typst si disponible

.EXAMPLE
  .\scripts\convert_EventZilla_doc_to_pdf.ps1
  .\scripts\convert_EventZilla_doc_to_pdf.ps1 -Engine xelatex
#>

param(
  [ValidateSet('pdflatex', 'xelatex', 'lualatex')]
  [string]$Engine = 'pdflatex'
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $ScriptDir
$md = Join-Path $root 'docs\eventzilla\EventZilla_Dashboards_KPIs_Objectifs.md'
$pdf = Join-Path $root 'deliverables\EventZilla_Dashboards_KPIs_Objectifs.pdf'

if (-not (Test-Path $md)) {
  Write-Error "Fichier introuvable : $md"
}

$pandoc = Get-Command pandoc -ErrorAction SilentlyContinue
if (-not $pandoc) {
  Write-Host @"
Pandoc n'est pas dans le PATH.
Installation : https://pandoc.org/installing.html

Sans Pandoc, vous pouvez :
  1) Ouvrir le .md dans VS Code / Cursor → Aperçu Markdown → Imprimer → PDF
  2) Coller le contenu sur https://www.markdowntopdf.com/ (service tiers)
"@
  exit 1
}

Write-Host "Conversion : $md -> $pdf (moteur : $Engine)"
& pandoc $md -o $pdf --pdf-engine=$Engine -V geometry:margin=2cm -V lang=fr
if ($LASTEXITCODE -ne 0) {
  Write-Host @"

Si le moteur LaTeX manque, installez MiKTeX ou utilisez :
  choco install pandoc miktex
ou essayez une autre engine :
  .\scripts\convert_EventZilla_doc_to_pdf.ps1 -Engine xelatex
"@
  exit $LASTEXITCODE
}

Write-Host "PDF cree : $pdf"

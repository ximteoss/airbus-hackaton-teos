from __future__ import annotations

import html
from pathlib import Path
from src.schemas import AuditReport


def generate_html_report(report: AuditReport, output_path: str | Path) -> Path:
    total = len(report.findings)
    n_ok = sum(1 for f in report.findings if f.verdict == "Cumple")
    n_ko = sum(1 for f in report.findings if f.verdict == "No cumple")
    n_rev = sum(1 for f in report.findings if f.verdict == "Revisar")

    rows_html: list[str] = []
    for f in report.findings:
        badge_cls = {
            "Cumple": "badge-ok",
            "No cumple": "badge-ko",
            "Revisar": "badge-rev",
        }[f.verdict]

        clauses_badges = " ".join(
            f'<span class="clause-tag">{html.escape(c)}</span>' for c in f.applicable_clauses
        ) or '<span class="muted">Sin asignar</span>'

        snippets_html = "".join(
            f"<div class='snippet'><strong>{html.escape(cid)}:</strong> {html.escape(txt[:450])}</div>"
            for cid, txt in f.clause_snippets.items()
        )
        missing_html = (
            "<ul>" + "".join(f"<li>{html.escape(m)}</li>" for m in f.missing_elements) + "</ul>"
            if f.missing_elements
            else ""
        )

        rows_html.append(f"""
        <tr data-status="{html.escape(f.verdict)}" data-search="{html.escape((f.req_id + ' ' + f.req_title + ' ' + ' '.join(f.applicable_clauses) + ' ' + f.justification).lower())}">
          <td><strong>{html.escape(f.req_id)}</strong><div class="muted">{html.escape(f.req_title)}</div></td>
          <td>{clauses_badges}</td>
          <td><span class="badge {badge_cls}">{html.escape(f.verdict)}</span><div class="muted">Conf: {f.confidence:.0%}</div></td>
          <td>
            <div>{html.escape(f.justification)}</div>
            {missing_html}
            <details><summary>Ver texto del requisito y cláusula citada</summary>
              <div class="snippet"><strong>Requisito:</strong> {html.escape(f.req_text)}</div>
              {snippets_html}
            </details>
          </td>
          <td><span class="muted">{html.escape(f.human_status)}</span></td>
        </tr>
        """)

    consistency_html = "".join(
        f"<li><strong>[{html.escape(i.severity.upper())} - {html.escape(i.issue_type)}]</strong> "
        f"({', '.join(html.escape(r) for r in i.involved_req_ids)}): {html.escape(i.description)}</li>"
        for i in report.consistency.issues
    ) or "<li>No se detectaron contradicciones entre requisitos.</li>"

    page = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <title>Auditoría Normativa - {html.escape(report.document_name)}</title>
  <style>
    :root {{ --bg:#0d1117; --card:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#a371f7; }}
    body {{ font-family: system-ui, -apple-system, sans-serif; background:var(--bg); color:var(--text); margin:0; padding:2rem; }}
    .wrap {{ max-width:1280px; margin:auto; }}
    h1 {{ margin:0 0 .25rem; font-size:1.6rem; }}
    .sub {{ color:var(--muted); margin-bottom:1.5rem; }}
    .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.5rem; }}
    .kpi {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:1rem; }}
    .kpi .num {{ font-size:2rem; font-weight:700; }}
    .ok {{ color:#3fb950; }} .ko {{ color:#f85149; }} .rev {{ color:#d29922; }}
    .controls {{ display:flex; gap:.75rem; margin-bottom:1rem; flex-wrap:wrap; }}
    input, select {{ background:var(--card); color:var(--text); border:1px solid var(--border); border-radius:8px; padding:.55rem .8rem; }}
    input {{ flex:1; min-width:240px; }}
    table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--border); border-radius:10px; overflow:hidden; }}
    th, td {{ text-align:left; padding:.85rem; border-bottom:1px solid var(--border); vertical-align:top; }}
    th {{ background:#1f242d; font-size:.85rem; text-transform:uppercase; color:var(--muted); }}
    .badge {{ display:inline-block; padding:.2rem .6rem; border-radius:999px; font-weight:600; font-size:.8rem; }}
    .badge-ok {{ background:#1a3a24; color:#3fb950; }}
    .badge-ko {{ background:#3d1b1e; color:#f85149; }}
    .badge-rev {{ background:#3b2e13; color:#d29922; }}
    .clause-tag {{ display:inline-block; background:#241c35; color:var(--accent); border:1px solid #483273; padding:.15rem .45rem; border-radius:6px; font-size:.8rem; margin:0 .25rem .25rem 0; }}
    .muted {{ color:var(--muted); font-size:.82rem; margin-top:.2rem; }}
    details {{ margin-top:.5rem; }} summary {{ cursor:pointer; color:var(--accent); font-size:.85rem; }}
    .snippet {{ background:#0d1117; border-left:3px solid var(--accent); padding:.5rem .75rem; margin-top:.4rem; font-size:.83rem; }}
    .panel {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:1rem 1.25rem; margin-top:1.5rem; }}
  </style>
</head>
<body>
<div class="wrap">
  <h1>Informe de Auditoría Normativa Agéntica</h1>
  <div class="sub">Documento: <strong>{html.escape(report.document_name)}</strong> · Normativa: {html.escape(report.regulation_name)} · Cobertura normativa: {report.rule_coverage_pct}%</div>

  <div class="kpis">
    <div class="kpi"><div class="num">{total}</div><div class="muted">Requisitos auditados</div></div>
    <div class="kpi"><div class="num ok">{n_ok}</div><div class="muted">Cumple</div></div>
    <div class="kpi"><div class="num ko">{n_ko}</div><div class="muted">No cumple</div></div>
    <div class="kpi"><div class="num rev">{n_rev}</div><div class="muted">Requiere revisión</div></div>
  </div>

  <div class="controls">
    <input id="q" type="text" placeholder="Filtrar por REQ-ID, cláusula (ej: IS.I.OR.245) o texto..." oninput="filterTable()" />
    <select id="st" onchange="filterTable()">
      <option value="">Todos los estados</option>
      <option value="Cumple">Cumple</option>
      <option value="No cumple">No cumple</option>
      <option value="Revisar">Revisar</option>
    </select>
  </div>

  <table>
    <thead><tr><th>Requisito</th><th>Cláusula aplicable</th><th>Estado</th><th>Justificación y evidencia</th><th>HITL</th></tr></thead>
    <tbody id="tbody">{"".join(rows_html)}</tbody>
  </table>

  <div class="panel">
    <h3>Análisis de Consistencia del Documento</h3>
    <p class="muted">{html.escape(report.consistency.homogeneity_summary)}</p>
    <ul>{consistency_html}</ul>
  </div>
</div>
<script>
function filterTable() {{
  const q = document.getElementById('q').value.toLowerCase();
  const st = document.getElementById('st').value;
  for (const tr of document.querySelectorAll('#tbody tr')) {{
    const okSt = !st || tr.dataset.status === st;
    const okQ = !q || tr.dataset.search.includes(q);
    tr.style.display = (okSt && okQ) ? '' : 'none';
  }}
}}
</script>
</body>
</html>"""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
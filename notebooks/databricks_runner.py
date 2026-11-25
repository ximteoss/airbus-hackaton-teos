# Databricks notebook source
# MAGIC %md
# MAGIC # Airbus Defence & Space — Auditoría Normativa Agéntica (IndesIAhack 2026)
# MAGIC Pipeline end-to-end: **Ingesta (EASA Part-IS + VLM) → RAG Híbrido / Azure AI Search → LangGraph (Auditor + Trazabilidad + Crítico + Consistencia) → HITL → Informe HTML + Evaluación Gold Set**.

# COMMAND ----------
# MAGIC %pip install -q langgraph langchain-core langchain-chroma langchain-huggingface langchain-openai langchain-google-genai pydantic pydantic-settings rank-bm25 lxml pypdf python-docx openpyxl sentence-transformers databricks-langchain azure-search-documents

# COMMAND ----------
dbutils.library.restartPython()  # type: ignore[name-defined]

# COMMAND ----------
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
endpoints = list(w.serving_endpoints.list())
print(f"Endpoints disponibles ({len(endpoints)}):")
for ep in endpoints:
    print(f"  - {ep.name} | estado: {ep.state.ready if ep.state else 'desconocido'}")

# COMMAND ----------
# Widgets interactivos en el panel lateral para cambiar parámetros el Día 1 sin tocar código
dbutils.widgets.dropdown("llm_provider", "databricks", ["databricks", "azure_openai", "openai", "gemini"], "1. Proveedor LLM")  # type: ignore[name-defined]
dbutils.widgets.text("llm_model", "databricks-gpt-oss-120b", "2. Endpoint / Modelo LLM")  # type: ignore[name-defined]
dbutils.widgets.dropdown("retriever_backend", "local_hybrid", ["local_hybrid", "azure_search"], "3. Backend RAG")  # type: ignore[name-defined]
dbutils.widgets.text("reg_path", "data/regulations/easa_part_is.xml", "4. Ruta normativa (.xml/.pdf)")  # type: ignore[name-defined]
dbutils.widgets.text("doc_path", "data/input_docs/sample_part_is_reqs.md", "5. Documento a auditar")  # type: ignore[name-defined]
dbutils.widgets.text("gold_path", "data/gold_set/sample_gold.json", "6. Ruta Gold Set (opcional)")  # type: ignore[name-defined]
dbutils.widgets.dropdown("enable_vlm", "false", ["false", "true"], "7. Activar VLM para diagramas")  # type: ignore[name-defined]

# COMMAND ----------
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

# 1. Localizar raíz del repo SIN cambiar el cwd
candidates = [Path.cwd(), *Path.cwd().parents]
repo_root = next((p for p in candidates if (p / "src").is_dir()), None)
if repo_root is None:
    raise FileNotFoundError(f"No encuentro 'src' desde {Path.cwd()}")

repo_str = str(repo_root.resolve())
if repo_str in sys.path:
    sys.path.remove(repo_str)
sys.path.insert(0, repo_str)

sys.path_importer_cache.clear()
importlib.invalidate_caches()
print(f"1) REPO_ROOT OK: {repo_str}")

# 2. Fijar variables de entorno ANTES de importar src
os.environ["LLM_PROVIDER"] = dbutils.widgets.get("llm_provider")  # type: ignore[name-defined]
os.environ["LLM_MODEL"] = dbutils.widgets.get("llm_model")  # type: ignore[name-defined]
os.environ["RETRIEVER_BACKEND"] = dbutils.widgets.get("retriever_backend")  # type: ignore[name-defined]
os.environ["CHROMA_PERSIST_DIR"] = "/tmp/chroma_db"
# Embeddings locales en CPU: sin rate limits 429 y multilingües ES <-> EN
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["EMBEDDING_MODEL"] = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
os.environ["TOP_K_CLAUSES"] = "6"
os.environ["ENABLE_VLM"] = dbutils.widgets.get("enable_vlm")  # type: ignore[name-defined]
os.environ["VLM_MODEL"] = "databricks-llama-4-maverick"

# Limpiar colección previa de Chroma para evitar choques de dimensión
shutil.rmtree("/tmp/chroma_db", ignore_errors=True)

if "src.config" in sys.modules:
    importlib.reload(sys.modules["src.config"])

import mlflow
mlflow.langchain.autolog()

from langgraph.types import Command
from src.agents.graph import build_document_audit_graph
from src.eval.evaluate_gold import evaluate_against_gold
from src.ingestion.doc_extractor import extract_requirements_from_file
from src.ingestion.easa_parser import parse_easa_pdf, parse_easa_xml
from src.reporting.html_report import generate_html_report
from src.retrieval.retriever import build_retriever
from src.schemas import AuditReport

print("2) Imports de src + MLflow Tracing OK")


def _resolve(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else repo_root / p


# 3. Cargar normativa y requisitos
reg_path = _resolve(dbutils.widgets.get("reg_path"))  # type: ignore[name-defined]
doc_path = _resolve(dbutils.widgets.get("doc_path"))  # type: ignore[name-defined]

clauses = parse_easa_xml(reg_path) if reg_path.suffix.lower() == ".xml" else parse_easa_pdf(reg_path)
reqs = extract_requirements_from_file(doc_path)
print(f"3) Cláusulas cargadas: {len(clauses)} | Requisitos: {len(reqs)}")

# 4. Indexar y compilar grafo
retriever = build_retriever()
retriever.index_clauses(clauses)
graph = build_document_audit_graph(retriever)
print("4) Retriever y grafo listos")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Paso 1 — Ejecución del grafo agéntico hasta el punto de revisión humana (HITL)

# COMMAND ----------
thread_config = {"configurable": {"thread_id": f"dbx-audit-{doc_path.stem}"}}

paused_result = graph.invoke(
    {
        "document_name": doc_path.name,
        "requirements": reqs,
        "findings": [],
        "consistency": None,
        "rule_coverage_pct": 0.0,
        "hitl_enabled": True,
    },
    config=thread_config,
)

if "__interrupt__" in paused_result:
    items = paused_result["__interrupt__"][0].value["items_for_review"]
    print(f"Grafo pausado (HITL). {len(items)} hallazgos pendientes de aprobación:")
    for item in items:
        print(f"  - [{item['req_id']}] {item['verdict']} ({item['confidence']:.0%}) | {item['clauses']}")
        print(f"    {item['justification']}\n")
else:
    print("Ejecución completada sin interrupción pendiente.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Paso 2 — Revisión humana (HITL) y cierre del informe
# MAGIC Modifica `HUMAN_OVERRIDES` si quieres corregir algún veredicto en directo durante la demo (o déjalo con `_action: approve_all` para aprobar todo).

# COMMAND ----------
HUMAN_OVERRIDES = {
    "_action": "approve_all",
    # Ejemplo de edición humana en vivo ante el jurado:
    # "REQ-002": {"human_status": "edited", "verdict": "Revisar", "human_note": "Pendiente de validar excepción con seguridad"},
}

final_result = (
    graph.invoke(Command(resume=HUMAN_OVERRIDES), config=thread_config)
    if "__interrupt__" in paused_result
    else paused_result
)

report = AuditReport(
    document_name=doc_path.name,
    findings=final_result["findings"],
    consistency=final_result["consistency"],
    rule_coverage_pct=final_result["rule_coverage_pct"],
)

out_dir = repo_root / "data/outputs"
out_dir.mkdir(parents=True, exist_ok=True)
json_path = out_dir / f"{doc_path.stem}_report.json"
html_path = out_dir / f"{doc_path.stem}_report.html"

json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
generate_html_report(report, html_path)

print(f"Informe guardado en {json_path} y {html_path}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Paso 3 — Dashboard interactivo embebido en Databricks

# COMMAND ----------
displayHTML(html_path.read_text(encoding="utf-8"))  # type: ignore[name-defined]

# COMMAND ----------
# MAGIC %md
# MAGIC ### Paso 4 — Evaluación objetiva frente al Gold Set (F1 y Trazabilidad)

# COMMAND ----------
gold_path = _resolve(dbutils.widgets.get("gold_path"))  # type: ignore[name-defined]
if gold_path.exists():
    metrics = evaluate_against_gold(json_path, gold_path)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
else:
    print(f"No se encontró gold set en {gold_path}. Sube el gold set del Día 1 para calcular F1 y trazabilidad.")
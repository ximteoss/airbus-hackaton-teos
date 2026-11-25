# Airbus Defence & Space — Auditoría Normativa Agéntica (IndesIAhack 2026)

Solución de IA agéntica en Python y **LangGraph** para verificar automáticamente el cumplimiento normativo de documentos técnicos frente a **EASA Part-IS** (Reglamentos UE 2022/1645 y 2023/203), garantizando trazabilidad directa `requisito -> cláusula`, detección de inconsistencias internas, revisión *Human-in-the-loop* (HITL) e informe HTML interactivo.

## Arquitectura del pipeline

1. **Ingesta normativa y documental (`src/ingestion/`):** Parser nativo para *Easy Access Rules* de EASA en formato XML (WordprocessingML con estilos `Heading3IR`, `Heading4AMC`, `Heading4GM`) y fallback PDF.
2. **RAG Híbrido IR-First (`src/retrieval/`):** Búsqueda híbrida local (**Chroma + BM25 con RRF**) priorizando *Implementing Rules* (`IR`), con adaptador listo para **Azure AI Search**.
3. **Orquestación agéntica (`src/agents/graph.py`):**
   - **Auditor:** Evalúa cumplimiento (`Cumple` / `No cumple` / `Revisar`) con salida estructurada Pydantic y reformulación automática de consulta.
   - **Guardrail de trazabilidad:** Validación determinista de que toda cláusula citada existe en el contexto recuperado.
   - **Crítico (bucle de reflexión):** Revisa falsos positivos/negativos y calibra la confianza.
   - **Analizador de consistencia:** Detecta contradicciones, ambigüedades y homogeneidad de detalle entre requisitos.
   - **Human-in-the-loop (`interrupt`):** Pausa el grafo para aprobación o edición humana antes de exportar.
4. **Evaluación y reporting (`src/eval/`, `src/reporting/`):** Cálculo de Precision, Recall, F1 y trazabilidad frente a *gold set*, más dashboard HTML autocontenido con filtros.

## Estructura del repositorio

```text
airbus-hackaton-teos/
├── main.py                         # CLI local (demo, auditoría y HITL)
├── notebooks/databricks_runner.py  # Notebook multi-celda listo para Databricks
├── src/
│   ├── config.py                   # Switch de proveedores (agy / databricks / azure_openai / gemini)
│   ├── schemas.py                  # Contratos Pydantic v2
│   ├── ingestion/                  # Parsers EASA XML/PDF y extractor de requisitos
│   ├── retrieval/                  # RAG híbrido local y adaptador Azure AI Search
│   ├── agents/                     # Prompts y grafo LangGraph
│   ├── eval/                       # Benchmark objetivo contra gold set
│   └── reporting/                  # Generador de informe HTML interactivo
└── data/                           # Normativa pública, ejemplos y gold set sintético
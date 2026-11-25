from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from pypdf import PdfReader

from src.config import get_llm
from src.ingestion.vlm_extractor import extract_diagram_requirements
from src.schemas import Requirement

# Soporta prefijos habituales en ingeniería: REQ-001, SR-01, SEC_04, SYS-REQ-12, R-05
REQ_HEADER_RE = re.compile(
    r"(?m)^(?:#+\s*|\*\*)?((?:SYS[-_])?(?:REQ|SR|SEC|CYB|IS|R)[-_]?\d{2,5})(?:\*\*)?(?:\s*[:\-–|]\s*(.*))?$"
)


class _ExtractedBatch(BaseModel):
    requirements: list[Requirement] = Field(default_factory=list)


def _pick_col(row: dict[str, str], aliases: tuple[str, ...]) -> str:
    for key, val in row.items():
        norm = key.strip().lower()
        if norm in aliases and val is not None:
            return str(val).strip()
    return ""


def _rows_to_requirements(rows: list[dict[str, str]]) -> list[Requirement]:
    reqs: list[Requirement] = []
    for idx, row in enumerate(rows, start=1):
        req_id = _pick_col(row, ("req_id", "id", "code", "codigo", "código", "identifier", "req"))
        title = _pick_col(row, ("title", "titulo", "título", "name", "nombre", "summary", "resumen"))
        text = _pick_col(row, ("text", "texto", "description", "descripcion", "descripción", "requirement", "requisito"))
        req_type = _pick_col(row, ("req_type", "type", "tipo", "category", "categoria", "categoría")) or "security"

        if not text and title:
            text = title
        if not text:
            continue

        reqs.append(
            Requirement(
                req_id=req_id or f"REQ-{idx:03d}",
                title=title or text[:70],
                text=text,
                req_type=req_type,
            )
        )
    return reqs


def _extract_from_csv(file_path: Path) -> list[Requirement]:
    with file_path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t") if sample else csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        return _rows_to_requirements([dict(r) for r in reader])


def _extract_from_xlsx(file_path: Path) -> list[Requirement]:
    import openpyxl  # type: ignore[import-not-found]

    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)
    headers = [str(h or "").strip() for h in next(rows_iter, ())]
    dict_rows = [
        {headers[i]: ("" if cell is None else str(cell)) for i in range(min(len(headers), len(r)))}
        for r in rows_iter
    ]
    return _rows_to_requirements(dict_rows)


def _extract_from_docx(file_path: Path) -> list[Requirement]:
    import docx  # type: ignore[import-not-found]

    doc = docx.Document(str(file_path))

    # 1. Intentar extraer primero de tablas estructuradas (muy común en pliegos)
    table_rows: list[dict[str, str]] = []
    for table in doc.tables:
        if len(table.rows) < 2:
            continue
        headers = [c.text.strip() for c in table.rows[0].cells]
        for row in table.rows[1:]:
            cells = [c.text.strip() for c in row.cells]
            table_rows.append({headers[i]: cells[i] for i in range(min(len(headers), len(cells)))})

    reqs = _rows_to_requirements(table_rows)
    if reqs:
        return reqs

    # 2. Si no hay tablas de requisitos, parsear párrafos
    full_text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    return extract_requirements_from_text(full_text)


def _extract_with_llm_fallback(text: str) -> list[Requirement]:
    """Agente Extractor: estructura requisitos cuando el documento no sigue un formato fijo."""
    extractor_llm = get_llm().with_structured_output(_ExtractedBatch)
    system = (
        "Eres un ingeniero de sistemas extrayendo requisitos técnicos de un documento.\n"
        "Identifica cada requisito individual y devuelve su `req_id` (usa el del documento si existe, "
        "o genera REQ-001, REQ-002...), un `title` breve, el `text` íntegro del requisito y su `req_type` "
        "(security, functional, design, governance)."
    )

    chunk_size = 12000
    all_reqs: list[Requirement] = []
    seen_ids: set[str] = set()

    for start in range(0, len(text), chunk_size):
        chunk = text[start : start + chunk_size]
        batch: _ExtractedBatch = extractor_llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=f"DOCUMENTO:\n{chunk}"),
        ])
        for r in batch.requirements:
            if r.req_id not in seen_ids:
                seen_ids.add(r.req_id)
                all_reqs.append(r)

    return all_reqs


def extract_requirements_from_text(text: str, allow_llm_fallback: bool = True) -> list[Requirement]:
    matches = list(REQ_HEADER_RE.finditer(text))
    requirements: list[Requirement] = []

    for idx, match in enumerate(matches):
        req_id = match.group(1).strip()
        title = (match.group(2) or "").strip()
        body_start = match.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = " ".join(text[body_start:body_end].split())

        if not body and title:
            body = title
        if not body:
            continue

        requirements.append(
            Requirement(
                req_id=req_id,
                title=title or body[:70],
                text=body,
                req_type="security",
            )
        )

    if not requirements and allow_llm_fallback and len(text.strip()) > 30:
        print("[Extractor] No se detectaron prefijos estándar; activando extracción por Agente LLM...", flush=True)
        return _extract_with_llm_fallback(text)

    return requirements


def extract_requirements_from_file(path: str | Path) -> list[Requirement]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".json":
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        items = raw["requirements"] if isinstance(raw, dict) and "requirements" in raw else raw
        return [Requirement(**item) for item in items]

    if suffix == ".csv":
        return _extract_from_csv(file_path)

    if suffix in {".xlsx", ".xlsm"}:
        return _extract_from_xlsx(file_path)

    if suffix == ".docx":
        return _extract_from_docx(file_path) + extract_diagram_requirements(file_path)

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return extract_requirements_from_text(text) + extract_diagram_requirements(file_path)

    text = file_path.read_text(encoding="utf-8")
    return extract_requirements_from_text(text)
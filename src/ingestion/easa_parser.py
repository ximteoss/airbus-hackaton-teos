from __future__ import annotations

import re
from pathlib import Path
from lxml import etree
from pypdf import PdfReader

from src.schemas import RegulationClause

# Captura códigos de EASA Part-IS, incluyendo múltiples subapartados: IS.I.OR.205, AMC1 IS.I.OR.200(a)(1), GM2 IS.D.OR.205(b)
CLAUSE_PATTERN = re.compile(
    r"\b((?:(?:AMC|GM)\d*\s+)?IS\.[ID]\.(?:OR|AR)\.\d+(?:\([a-z0-9]+\))*)"
)

HEADER_PATTERN = re.compile(
    r"^((?:(?:AMC|GM)\d*\s+)?IS\.[ID]\.(?:OR|AR)\.\d+(?:\([a-z0-9]+\))*)\s*(.*)$"
)


def _infer_level(clause_id: str, style: str = "") -> str:
    s = style.lower()
    if "amc" in s or clause_id.upper().startswith("AMC"):
        return "AMC"
    if "gm" in s or clause_id.upper().startswith("GM"):
        return "GM"
    return "IR"


def _get_pstyle(p_elem: etree._Element) -> str:
    for child in p_elem.iter():
        if isinstance(child.tag, str) and etree.QName(child).localname.lower() == "pstyle":
            for k, v in child.attrib.items():
                if k.lower().endswith("val"):
                    return v
    return ""


def _parse_wordprocessing_ml(root: etree._Element, regulation_name: str) -> list[RegulationClause]:
    """Parsea el XML oficial de EASA (WordprocessingML) guiándose por estilos Heading3IR / Heading4AMC / Heading4GM."""
    clauses_by_id: dict[str, RegulationClause] = {}

    current_id: str | None = None
    current_title: str = ""
    current_level: str = "IR"
    current_lines: list[str] = []

    def _flush() -> None:
        nonlocal current_id, current_title, current_level, current_lines
        if not current_id:
            return
        body = " ".join(current_lines).strip()
        if len(body) >= 40:
            existing = clauses_by_id.get(current_id)
            if existing is None or len(body) > len(existing.text):
                clauses_by_id[current_id] = RegulationClause(
                    clause_id=current_id,
                    regulation=regulation_name,
                    title=current_title[:200],
                    level=current_level,
                    text=body[:6000],
                )
        current_id = None
        current_title = ""
        current_level = "IR"
        current_lines = []

    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue
        if etree.QName(elem).localname.lower() != "p":
            continue

        style = _get_pstyle(elem)
        style_low = style.lower()
        if "toc" in style_low:
            continue

        para_text = " ".join(" ".join(elem.itertext()).split())
        if not para_text:
            continue

        if style_low.startswith("heading"):
            match = HEADER_PATTERN.match(para_text)
            if match:
                _flush()
                current_id = " ".join(match.group(1).split())
                current_title = (match.group(2) or "").strip()
                current_level = _infer_level(current_id, style)
                current_lines = [para_text]
            else:
                # Nuevo capítulo, anexo o apéndice sin código Part-IS: cerramos la cláusula anterior
                _flush()
        elif current_id is not None:
            current_lines.append(para_text)

    _flush()
    return list(clauses_by_id.values())


def parse_easa_xml(xml_path: str | Path, regulation_name: str = "EASA Part-IS") -> list[RegulationClause]:
    """Extrae cláusulas atómicas de un XML oficial de EASA (WordprocessingML o XML estructurado)."""
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    has_word_p = any(
        isinstance(el.tag, str) and etree.QName(el).localname.lower() == "p"
        for el in root.iter()
    )
    if has_word_p:
        word_clauses = _parse_wordprocessing_ml(root, regulation_name)
        if word_clauses:
            return word_clauses

    # Fallback para XML estructurado estándar
    clauses_by_id: dict[str, RegulationClause] = {}
    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue
        tag = etree.QName(elem).localname.lower()
        if tag not in {"topic", "section", "article", "requirement", "div", "rule"}:
            continue

        full_text = " ".join(" ".join(elem.itertext()).split())
        if len(full_text) < 40:
            continue

        match = CLAUSE_PATTERN.search(full_text[:200])
        if not match:
            continue

        clause_id = " ".join(match.group(1).split())
        title = ""
        for child in elem:
            if isinstance(child.tag, str) and etree.QName(child).localname.lower() in {"title", "heading", "name"}:
                title = " ".join(" ".join(child.itertext()).split())
                break

        if clause_id not in clauses_by_id or len(full_text) < len(clauses_by_id[clause_id].text):
            clauses_by_id[clause_id] = RegulationClause(
                clause_id=clause_id,
                regulation=regulation_name,
                title=title[:200],
                level=_infer_level(clause_id),
                text=full_text[:6000],
            )

    return list(clauses_by_id.values())


def parse_easa_pdf(pdf_path: str | Path, regulation_name: str = "EASA Part-IS") -> list[RegulationClause]:
    """Fallback robusto para extraer cláusulas de un PDF de EASA Part-IS."""
    reader = PdfReader(str(pdf_path))
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    matches = list(CLAUSE_PATTERN.finditer(full_text))
    clauses: dict[str, RegulationClause] = {}

    for idx, match in enumerate(matches):
        clause_id = " ".join(match.group(1).split())
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        chunk = " ".join(full_text[start:end].split())

        if len(chunk) < 50:
            continue

        first_line = chunk.split(".")[0][:160]
        if clause_id not in clauses or len(chunk) > len(clauses[clause_id].text):
            clauses[clause_id] = RegulationClause(
                clause_id=clause_id,
                regulation=regulation_name,
                title=first_line,
                level=_infer_level(clause_id),
                text=chunk[:6000],
            )

    return list(clauses.values())
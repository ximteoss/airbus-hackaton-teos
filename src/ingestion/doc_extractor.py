from __future__ import annotations

import json
import re
from pathlib import Path
from pypdf import PdfReader

from src.schemas import Requirement

REQ_HEADER_RE = re.compile(r"(?m)^(?:#+\s*)?(REQ[-_]\d+)(?:\s*[:\-–]\s*(.*))?$")


def extract_requirements_from_file(path: str | Path) -> list[Requirement]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".json":
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        items = raw["requirements"] if isinstance(raw, dict) and "requirements" in raw else raw
        return [Requirement(**item) for item in items]

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = file_path.read_text(encoding="utf-8")

    return extract_requirements_from_text(text)


def extract_requirements_from_text(text: str) -> list[Requirement]:
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

        requirements.append(
            Requirement(
                req_id=req_id,
                title=title or body[:70],
                text=body,
                req_type="security",
            )
        )

    return requirements
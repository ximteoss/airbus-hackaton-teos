from __future__ import annotations

import base64
import zipfile
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from pypdf import PdfReader

from src.config import settings
from src.schemas import Requirement

MIN_IMAGE_BYTES = 15_000  # Ignora logos, iconos y viñetas pequeñas
MAX_IMAGES_PER_DOC = 6    # Evita disparar latencia en documentos con decenas de capturas


class _DiagramFindings(BaseModel):
    requirements: list[Requirement] = Field(default_factory=list)


def _get_vlm():
    if settings.llm_provider == "databricks":
        from databricks_langchain import ChatDatabricks  # type: ignore[import-not-found]
        return ChatDatabricks(endpoint=settings.vlm_model, temperature=0.0)
    if settings.llm_provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=settings.vlm_model,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            temperature=0.0,
        )
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.vlm_model, temperature=0.0, api_key=settings.openai_api_key or None)
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.vlm_model,
            temperature=0.0,
            google_api_key=settings.google_api_key or None,
        )
    return None


def _guess_mime(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "image/png"


def _extract_images(file_path: Path) -> list[tuple[str, str, bytes]]:
    """Devuelve lista de (etiqueta, mime_type, bytes) para imágenes de PDF o DOCX."""
    suffix = file_path.suffix.lower()
    images: list[tuple[str, str, bytes]] = []

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        for page_num, page in enumerate(reader.pages, start=1):
            for img_idx, img in enumerate(getattr(page, "images", []), start=1):
                data = img.data
                if len(data) < MIN_IMAGE_BYTES:
                    continue
                images.append((f"Página {page_num} - Fig {img_idx}", _guess_mime(img.name), data))

    elif suffix == ".docx":
        with zipfile.ZipFile(file_path, "r") as zf:
            media_files = sorted(
                n for n in zf.namelist()
                if n.startswith("word/media/") and Path(n).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            )
            for idx, name in enumerate(media_files, start=1):
                data = zf.read(name)
                if len(data) < MIN_IMAGE_BYTES:
                    continue
                images.append((f"Figura DOCX {idx} ({Path(name).name})", _guess_mime(name), data))

    return images[:MAX_IMAGES_PER_DOC]


def extract_diagram_requirements(file_path: Path) -> list[Requirement]:
    """Analiza diagramas incrustados con un VLM y los convierte en requisitos auditables."""
    if not settings.enable_vlm:
        return []

    images = _extract_images(file_path)
    if not images:
        return []

    vlm = _get_vlm()
    if vlm is None:
        print("[VLM] Proveedor LLM actual no soporta visión; se omiten figuras.", flush=True)
        return []

    structured_vlm = vlm.with_structured_output(_DiagramFindings)
    system_prompt = (
        "Eres un auditor de ciberseguridad aeronáutica analizando un diagrama técnico extraído de una especificación.\n"
        "Extrae entre 1 y 3 requisitos/afirmaciones de arquitectura observables en la imagen relevantes para EASA Part-IS "
        "(segmentación de red, cifrado, gestión de logs/retención, acceso de terceros/cloud, respuesta a incidentes, "
        "roles o controles de acceso).\n"
        "Usa req_id con formato REQ-FIG-001, REQ-FIG-002... y describe hechos concretos visibles en el diagrama. "
        "Si la imagen es puramente decorativa (portada, logo), devuelve una lista vacía."
    )

    extracted: list[Requirement] = []
    counter = 1

    for label, mime, raw_bytes in images:
        print(f"[VLM] Analizando {label} con {settings.vlm_model}...", flush=True)
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        msg = HumanMessage(
            content=[
                {"type": "text", "text": f"Analiza esta figura ({label}) y extrae requisitos de seguridad observables."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]
        )
        try:
            res: _DiagramFindings = structured_vlm.invoke([SystemMessage(content=system_prompt), msg])
            for r in res.requirements:
                extracted.append(
                    r.model_copy(
                        update={
                            "req_id": f"REQ-FIG-{counter:03d}",
                            "title": f"[{label}] {r.title}",
                            "req_type": "diagram",
                        }
                    )
                )
                counter += 1
        except Exception as exc:
            print(f"[VLM] Aviso: no se pudo analizar {label}: {exc}", flush=True)

    return extracted
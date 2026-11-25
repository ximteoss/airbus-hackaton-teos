from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

VerdictType = Literal["Cumple", "No cumple", "Revisar"]


class RegulationClause(BaseModel):
    clause_id: str = Field(..., description="Código único de la cláusula, ej: IS.I.OR.245")
    regulation: str = Field(default="EASA Part-IS", description="Normativa origen")
    title: str = Field(default="", description="Título de la sección o cláusula")
    level: str = Field(default="IR", description="IR (Implementing Rule), AMC o GM")
    text: str = Field(..., description="Texto íntegro de la cláusula normativa")


class Requirement(BaseModel):
    req_id: str = Field(..., description="Identificador del requisito, ej: REQ-021")
    title: str = Field(default="", description="Título breve del requisito")
    text: str = Field(..., description="Descripción completa del requisito técnico")
    req_type: str = Field(default="functional", description="Tipo: security, functional, design, etc.")
    source_page: int | None = Field(default=None, description="Página origen en el documento")


class AuditDraft(BaseModel):
    """Salida estructurada del Agente Auditor."""
    verdict: VerdictType = Field(..., description="Veredicto de cumplimiento normativo")
    applicable_clauses: list[str] = Field(
        default_factory=list,
        description="Lista EXACTA de clause_id tomados únicamente del contexto recuperado",
    )
    justification: str = Field(
        ...,
        description="Justificación técnica concisa, basada en evidencias del requisito y la cláusula",
    )
    missing_elements: list[str] = Field(
        default_factory=list,
        description="Elementos concretos exigidos por la norma que faltan o se incumplen",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza estimada [0.0, 1.0]")
    needs_query_rewrite: bool = Field(
        default=False,
        description="True si las cláusulas recuperadas no cubren bien este requisito",
    )
    suggested_query: str | None = Field(
        default=None,
        description="Consulta reformulada si needs_query_rewrite es True",
    )


class CriticReview(BaseModel):
    """Salida estructurada del Agente Crítico (bucle de reflexión)."""
    approved: bool = Field(..., description="True si el veredicto y las citas son sólidos")
    issues_found: list[str] = Field(
        default_factory=list,
        description="Debilidades en el razonamiento, citas incorrectas o sesgos detectados",
    )
    feedback: str = Field(
        default="",
        description="Instrucción concreta para que el Auditor mejore su evaluación",
    )
    adjusted_confidence: float = Field(..., ge=0.0, le=1.0)


class AuditFinding(BaseModel):
    """Hallazgo consolidado y trazable para el informe y evaluación."""
    req_id: str
    req_title: str
    req_text: str
    verdict: VerdictType
    applicable_clauses: list[str]
    clause_snippets: dict[str, str] = Field(default_factory=dict)
    justification: str
    missing_elements: list[str] = Field(default_factory=list)
    confidence: float
    reflection_iterations: int = 0
    traceability_verified: bool = True
    human_status: Literal["pending", "approved", "rejected", "edited"] = "pending"
    human_note: str = ""


class ConsistencyIssue(BaseModel):
    """Inconsistencia o contradicción detectada entre requisitos del documento."""
    issue_type: Literal["contradiction", "ambiguity", "detail_homogeneity"]
    involved_req_ids: list[str]
    description: str
    severity: Literal["alta", "media", "baja"]


class ConsistencyReport(BaseModel):
    issues: list[ConsistencyIssue] = Field(default_factory=list)
    homogeneity_summary: str = ""


class AuditReport(BaseModel):
    document_name: str
    regulation_name: str = "EASA Part-IS"
    findings: list[AuditFinding]
    consistency: ConsistencyReport
    rule_coverage_pct: float = 0.0
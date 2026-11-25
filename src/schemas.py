from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class RegulationClause(BaseModel):
    clause_id: str = Field(..., description="Código exacto de la cláusula (ej: IS.I.OR.205)")
    regulation: str = Field(default="EASA Part-IS", description="Nombre de la normativa")
    title: str = Field(default="", description="Título o epígrafe de la cláusula")
    level: str = Field(default="IR", description="Nivel normativo: IR, AMC o GM")
    text: str = Field(..., description="Texto normativo íntegro de la cláusula")


class Requirement(BaseModel):
    req_id: str = Field(..., description="Identificador del requisito (ej: REQ-001)")
    title: str = Field(default="", description="Título breve del requisito")
    text: str = Field(..., description="Texto completo del requisito técnico")
    req_type: str = Field(default="security", description="Categoría del requisito")


class AuditDraft(BaseModel):
    """Salida estructurada del Agente Auditor."""

    verdict: Literal["Cumple", "No cumple", "Revisar"] = Field(
        ..., description="Veredicto de cumplimiento normativo"
    )
    applicable_clauses: list[str] = Field(
        default_factory=list,
        description="Lista EXACTA de clause_id tomados únicamente del contexto recuperado",
    )
    justification: str = Field(
        ..., description="Justificación técnica concisa, basada en evidencias del requisito y la cláusula"
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
        default=None, description="Consulta reformulada si needs_query_rewrite es True"
    )

    @field_validator("applicable_clauses", "missing_elements", mode="before")
    @classmethod
    def _none_to_empty_list(cls, v: Any) -> Any:
        return [] if v is None else v

    @field_validator("needs_query_rewrite", mode="before")
    @classmethod
    def _none_to_false(cls, v: Any) -> Any:
        return False if v is None else v


class CriticReview(BaseModel):
    """Salida estructurada del Agente Crítico."""

    approved: bool = Field(..., description="True si el borrador es riguroso y trazable")
    feedback: str = Field(default="", description="Feedback concreto si approved es False")
    adjusted_confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza calibrada [0.0, 1.0]")

    @field_validator("feedback", mode="before")
    @classmethod
    def _none_to_empty_str(cls, v: Any) -> Any:
        return "" if v is None else v


class AuditFinding(BaseModel):
    req_id: str
    req_title: str = ""
    req_text: str
    verdict: Literal["Cumple", "No cumple", "Revisar"]
    applicable_clauses: list[str] = Field(default_factory=list)
    clause_snippets: dict[str, str] = Field(default_factory=dict)
    justification: str
    missing_elements: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reflection_iterations: int = 1
    traceability_verified: bool = True
    human_status: Literal["pending", "approved", "edited", "rejected"] = "pending"
    human_note: str = ""


class ConsistencyIssue(BaseModel):
    issue_type: Literal["contradiction", "ambiguity", "detail_homogeneity"]
    involved_req_ids: list[str] = Field(default_factory=list)
    description: str
    severity: Literal["alta", "media", "baja"] = "media"

    @field_validator("involved_req_ids", mode="before")
    @classmethod
    def _none_to_empty_list(cls, v: Any) -> Any:
        return [] if v is None else v


class ConsistencyReport(BaseModel):
    issues: list[ConsistencyIssue] = Field(default_factory=list)
    homogeneity_summary: str = ""

    @field_validator("issues", mode="before")
    @classmethod
    def _none_to_empty_list(cls, v: Any) -> Any:
        return [] if v is None else v


class AuditReport(BaseModel):
    document_name: str
    regulation_name: str = "EASA Part-IS"
    findings: list[AuditFinding] = Field(default_factory=list)
    consistency: ConsistencyReport | None = None
    rule_coverage_pct: float = 0.0
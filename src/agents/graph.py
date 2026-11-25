from __future__ import annotations

from typing import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agents.prompts import (
    AUDITOR_SYSTEM_PROMPT,
    CONSISTENCY_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
)
from src.config import get_llm, settings
from src.retrieval.retriever import BaseRegulationRetriever
from src.schemas import (
    AuditDraft,
    AuditFinding,
    ConsistencyReport,
    CriticReview,
    RegulationClause,
    Requirement,
)


# ---------------------------------------------------------------------------
# 1. Subgrafo de auditoría por requisito (RAG + Reflexión + Trazabilidad)
# ---------------------------------------------------------------------------
class SingleReqState(TypedDict):
    requirement: Requirement
    query: str
    retrieved_clauses: list[RegulationClause]
    draft: AuditDraft | None
    critic: CriticReview | None
    feedback: str
    traceability_ok: bool
    iterations: int
    rewrites: int


def _format_clauses(clauses: list[RegulationClause]) -> str:
    return "\n\n".join(
        f"### [{c.clause_id}] ({c.level}) {c.title}\n{c.text}" for c in clauses
    )


def build_single_req_subgraph(retriever: BaseRegulationRetriever):
    llm = get_llm()
    auditor_llm = llm.with_structured_output(AuditDraft)
    critic_llm = llm.with_structured_output(CriticReview)

    def retrieve_node(state: SingleReqState) -> dict:
        clauses = retriever.search(state["query"], top_k=settings.top_k_clauses)
        return {"retrieved_clauses": clauses}

    def auditor_node(state: SingleReqState) -> dict:
        req = state["requirement"]
        clauses_ctx = _format_clauses(state["retrieved_clauses"])
        feedback_block = (
            f"\n\nFEEDBACK DEL CRÍTICO (corrige estos puntos):\n{state['feedback']}"
            if state["feedback"]
            else ""
        )

        user_prompt = (
            f"REQUISITO [{req.req_id}] - {req.title}\n"
            f"Texto: {req.text}\n\n"
            f"CLÁUSULAS NORMATIVAS RECUPERADAS:\n{clauses_ctx}"
            f"{feedback_block}"
        )
        draft: AuditDraft = auditor_llm.invoke([
            SystemMessage(content=AUDITOR_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        return {"draft": draft, "iterations": state["iterations"] + 1}

    def rewrite_query_node(state: SingleReqState) -> dict:
        draft = state["draft"]
        new_query = (draft.suggested_query if draft and draft.suggested_query else state["query"])
        return {"query": new_query, "rewrites": state["rewrites"] + 1}

    def traceability_guard_node(state: SingleReqState) -> dict:
        """Verificación determinista: las cláusulas citadas DEBEN existir en el contexto recuperado."""
        draft = state["draft"]
        valid_ids = {c.clause_id for c in state["retrieved_clauses"]}
        if not draft or not draft.applicable_clauses:
            return {
                "traceability_ok": False,
                "feedback": "No citaste ninguna cláusula normativa válida del contexto recuperado.",
            }

        invalid = [cid for cid in draft.applicable_clauses if cid not in valid_ids]
        if invalid:
            return {
                "traceability_ok": False,
                "feedback": (
                    f"Citas inválidas o alucinadas: {invalid}. "
                    f"Usa exclusivamente códigos de esta lista: {sorted(valid_ids)}."
                ),
            }
        return {"traceability_ok": True}

    def critic_node(state: SingleReqState) -> dict:
        req = state["requirement"]
        draft = state["draft"]
        clauses_ctx = _format_clauses(state["retrieved_clauses"])

        user_prompt = (
            f"REQUISITO [{req.req_id}]: {req.text}\n\n"
            f"CLÁUSULAS REALES:\n{clauses_ctx}\n\n"
            f"BORRADOR DEL AUDITOR:\n{draft.model_dump_json(indent=2)}"
        )
        review: CriticReview = critic_llm.invoke([
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        return {"critic": review, "feedback": review.feedback}

    def route_after_auditor(state: SingleReqState) -> str:
        draft = state["draft"]
        if draft and draft.needs_query_rewrite and state["rewrites"] < 1:
            return "rewrite_query"
        return "traceability_guard"

    def route_after_traceability(state: SingleReqState) -> str:
        if not state["traceability_ok"] and state["iterations"] < settings.max_reflection_loops:
            return "auditor"
        return "critic"

    def route_after_critic(state: SingleReqState) -> str:
        critic = state["critic"]
        low_conf = critic is not None and critic.adjusted_confidence < settings.confidence_threshold
        not_approved = critic is not None and not critic.approved
        if (not_approved or low_conf) and state["iterations"] < settings.max_reflection_loops:
            return "auditor"
        return END

    sg = StateGraph(SingleReqState)
    sg.add_node("retrieve", retrieve_node)
    sg.add_node("auditor", auditor_node)
    sg.add_node("rewrite_query", rewrite_query_node)
    sg.add_node("traceability_guard", traceability_guard_node)
    sg.add_node("critic", critic_node)

    sg.add_edge(START, "retrieve")
    sg.add_edge("retrieve", "auditor")
    sg.add_conditional_edges(
        "auditor",
        route_after_auditor,
        {"rewrite_query": "rewrite_query", "traceability_guard": "traceability_guard"},
    )
    sg.add_edge("rewrite_query", "retrieve")
    sg.add_conditional_edges(
        "traceability_guard",
        route_after_traceability,
        {"auditor": "auditor", "critic": "critic"},
    )
    sg.add_conditional_edges(
        "critic",
        route_after_critic,
        {"auditor": "auditor", END: END},
    )

    return sg.compile()


# ---------------------------------------------------------------------------
# 2. Grafo Maestro (Supervisor del documento + Consistencia + HITL)
# ---------------------------------------------------------------------------
class DocumentAuditState(TypedDict):
    document_name: str
    requirements: list[Requirement]
    findings: list[AuditFinding]
    consistency: ConsistencyReport | None
    rule_coverage_pct: float
    hitl_enabled: bool


def build_document_audit_graph(retriever: BaseRegulationRetriever):
    req_subgraph = build_single_req_subgraph(retriever)
    consistency_llm = get_llm().with_structured_output(ConsistencyReport)

    def audit_all_requirements_node(state: DocumentAuditState) -> dict:
        findings: list[AuditFinding] = []
        covered_clauses: set[str] = set()
        total = len(state["requirements"])

        for idx, req in enumerate(state["requirements"], start=1):
            print(f"[{idx}/{total}] Auditando {req.req_id} ({req.title})...", flush=True)
            init_state: SingleReqState = {
                "requirement": req,
                "query": f"{req.title}. {req.text}",
                "retrieved_clauses": [],
                "draft": None,
                "critic": None,
                "feedback": "",
                "traceability_ok": False,
                "iterations": 0,
                "rewrites": 0,
            }
            res = req_subgraph.invoke(init_state)
            draft: AuditDraft = res["draft"]
            critic: CriticReview | None = res["critic"]
            clause_map = {c.clause_id: c.text for c in res["retrieved_clauses"]}

            valid_clauses = [cid for cid in draft.applicable_clauses if cid in clause_map]
            covered_clauses.update(valid_clauses)
            print(f"    -> {draft.verdict} | {valid_clauses} (iters: {res['iterations']})", flush=True)

            findings.append(
                AuditFinding(
                    req_id=req.req_id,
                    req_title=req.title,
                    req_text=req.text,
                    verdict=draft.verdict,
                    applicable_clauses=valid_clauses,
                    clause_snippets={cid: clause_map[cid] for cid in valid_clauses},
                    justification=draft.justification,
                    missing_elements=draft.missing_elements,
                    confidence=critic.adjusted_confidence if critic else draft.confidence,
                    reflection_iterations=res["iterations"],
                    traceability_verified=res["traceability_ok"],
                )
            )

        total_rules = len(retriever.get_all_clause_ids())
        coverage = round((len(covered_clauses) / total_rules) * 100, 2) if total_rules else 0.0
        return {"findings": findings, "rule_coverage_pct": coverage}

    def consistency_node(state: DocumentAuditState) -> dict:
        print("[Consistencia] Analizando contradicciones y homogeneidad del documento...", flush=True)
        reqs_dump = "\n".join(
            f"- [{r.req_id}] {r.title}: {r.text}" for r in state["requirements"]
        )
        report: ConsistencyReport = consistency_llm.invoke([
            SystemMessage(content=CONSISTENCY_SYSTEM_PROMPT),
            HumanMessage(content=f"DOCUMENTO A REVISAR:\n{reqs_dump}"),
        ])
        return {"consistency": report}

    def hitl_review_node(state: DocumentAuditState) -> dict:
        if not state.get("hitl_enabled", True):
            approved = [f.model_copy(update={"human_status": "approved"}) for f in state["findings"]]
            return {"findings": approved}

        to_review = [
            {
                "req_id": f.req_id,
                "verdict": f.verdict,
                "clauses": f.applicable_clauses,
                "confidence": f.confidence,
                "justification": f.justification,
            }
            for f in state["findings"]
            if f.verdict in ("No cumple", "Revisar") or f.confidence < settings.confidence_threshold
        ]

        human_feedback = interrupt({
            "instruction": "Aprueba, rechaza o corrige los hallazgos antes de cerrar el informe.",
            "items_for_review": to_review,
        })

        overrides = human_feedback if isinstance(human_feedback, dict) else {}
        updated_findings: list[AuditFinding] = []
        for f in state["findings"]:
            patch = overrides.get(f.req_id, {})
            if not isinstance(patch, dict):
                patch = {}
            status = patch.get("human_status", "approved")
            verdict = patch.get("verdict", f.verdict)
            note = patch.get("human_note", "")
            updated_findings.append(
                f.model_copy(update={"human_status": status, "verdict": verdict, "human_note": note})
            )
        return {"findings": updated_findings}

    graph = StateGraph(DocumentAuditState)
    graph.add_node("audit_all", audit_all_requirements_node)
    graph.add_node("consistency", consistency_node)
    graph.add_node("hitl_review", hitl_review_node)

    graph.add_edge(START, "audit_all")
    graph.add_edge("audit_all", "consistency")
    graph.add_edge("consistency", "hitl_review")
    graph.add_edge("hitl_review", END)

    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("src.schemas", "Requirement"),
            ("src.schemas", "AuditFinding"),
            ("src.schemas", "ConsistencyReport"),
            ("src.schemas", "ConsistencyIssue"),
        ]
    )
    return graph.compile(checkpointer=MemorySaver(serde=serializer))
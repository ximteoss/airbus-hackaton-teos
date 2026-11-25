from __future__ import annotations

import argparse
import json
from pathlib import Path
from langgraph.types import Command

from src.agents.graph import build_document_audit_graph
from src.ingestion.doc_extractor import extract_requirements_from_file
from src.ingestion.easa_parser import parse_easa_pdf, parse_easa_xml
from src.reporting.html_report import generate_html_report
from src.retrieval.retriever import build_retriever
from src.schemas import AuditReport, RegulationClause, Requirement


def _build_demo_data() -> tuple[list[RegulationClause], list[Requirement]]:
    clauses = [
        RegulationClause(
            clause_id="IS.I.OR.205",
            title="Information security risk assessment",
            level="IR",
            text=(
                "The organisation shall identify all of its elements which could be exposed to information security risks, "
                "identify risk scenarios, and assess their likelihood and severity using defined severity criteria."
            ),
        ),
        RegulationClause(
            clause_id="IS.I.OR.230",
            title="Information security external reporting scheme",
            level="IR",
            text=(
                "The organisation shall implement an information security reporting system specifying mandatory notification "
                "channels and timelines (initial notification within 72 hours) to the competent authority."
            ),
        ),
        RegulationClause(
            clause_id="IS.I.OR.240",
            title="Personnel requirements",
            level="IR",
            text=(
                "The organisation shall ensure that personnel performing information security tasks have the necessary "
                "competence, qualifications, and periodic security awareness training."
            ),
        ),
        RegulationClause(
            clause_id="IS.I.OR.245",
            title="Record-keeping",
            level="IR",
            text=(
                "The organisation shall maintain audit logs and information security records ensuring integrity, "
                "availability, and a minimum retention period of 5 years."
            ),
        ),
    ]
    reqs = [
        Requirement(
            req_id="REQ-014",
            title="Formación del personal",
            text="Todo el personal con funciones de ciberseguridad deberá contar con competencias acreditadas y recibir formación anual obligatoria.",
        ),
        Requirement(
            req_id="REQ-021",
            title="Registro de auditoría",
            text="El sistema registrará los eventos de acceso y cambios de configuración en logs centralizados protegidos contra manipulación.",
        ),
        Requirement(
            req_id="REQ-032",
            title="Evaluación de riesgos",
            text="Se realizará una evaluación periódica de activos expuestos a riesgos de seguridad de la información.",
        ),
        Requirement(
            req_id="REQ-047",
            title="Notificación de incidentes",
            text="Los incidentes de seguridad críticos se notificarán a la autoridad competente mediante el portal seguro oficial en un plazo máximo de 24 horas.",
        ),
    ]
    return clauses, reqs


def run_pipeline(
    clauses: list[RegulationClause],
    reqs: list[Requirement],
    doc_name: str,
    hitl: bool = True,
) -> AuditReport:
    retriever = build_retriever()
    retriever.index_clauses(clauses)

    graph = build_document_audit_graph(retriever)
    config = {"configurable": {"thread_id": f"audit-{doc_name}"}}

    result = graph.invoke(
        {
            "document_name": doc_name,
            "requirements": reqs,
            "findings": [],
            "consistency": None,
            "rule_coverage_pct": 0.0,
            "hitl_enabled": hitl,
        },
        config=config,
    )

    # Si HITL está activo, LangGraph se pausa en interrupt(); simulamos o pedimos aprobación en CLI
    if hitl and "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n=== HUMAN-IN-THE-LOOP: HALLAZGOS PENDIENTES DE REVISIÓN ===")
        for item in payload["items_for_review"]:
            print(f"- {item['req_id']} | {item['verdict']} | {item['clauses']} | {item['justification']}")
        input("\nPulsa ENTER para aprobar los hallazgos y cerrar el informe...")
        result = graph.invoke(Command(resume={"_action": "approve_all"}), config=config)

    report = AuditReport(
        document_name=doc_name,
        findings=result["findings"],
        consistency=result["consistency"],
        rule_coverage_pct=result["rule_coverage_pct"],
    )

    out_dir = Path("data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{Path(doc_name).stem}_report.json"
    html_path = out_dir / f"{Path(doc_name).stem}_report.html"

    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    generate_html_report(report, html_path)

    print(f"\nJSON generado: {json_path}")
    print(f"HTML generado: {html_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoría normativa agéntica - Airbus IndesIAhack 2026")
    parser.add_argument("--demo", action="store_true", help="Ejecuta prueba completa con datos sintéticos Part-IS")
    parser.add_argument("--reg", type=str, help="Ruta a normativa EASA (.xml o .pdf)")
    parser.add_argument("--doc", type=str, help="Ruta al documento de requisitos (.pdf, .md, .json)")
    parser.add_argument("--no-hitl", action="store_true", help="Ejecución desatendida (útil para benchmarking gold set)")
    args = parser.parse_args()

    if args.demo:
        clauses, reqs = _build_demo_data()
        run_pipeline(clauses, reqs, doc_name="demo_airbus_part_is", hitl=not args.no_hitl)
        return

    if not args.reg or not args.doc:
        parser.error("Indica --demo o bien --reg <normativa> y --doc <documento>")

    reg_path = Path(args.reg)
    clauses = parse_easa_xml(reg_path) if reg_path.suffix.lower() == ".xml" else parse_easa_pdf(reg_path)
    reqs = extract_requirements_from_file(args.doc)
    run_pipeline(clauses, reqs, doc_name=Path(args.doc).name, hitl=not args.no_hitl)


if __name__ == "__main__":
    main()
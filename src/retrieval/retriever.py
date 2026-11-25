from __future__ import annotations

import re
from abc import ABC, abstractmethod
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import get_embeddings, settings
from src.schemas import RegulationClause


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9\._\-]+", text.lower())


class BaseRegulationRetriever(ABC):
    @abstractmethod
    def index_clauses(self, clauses: list[RegulationClause]) -> None:
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[RegulationClause]:
        ...

    @abstractmethod
    def get_all_clause_ids(self) -> set[str]:
        ...


class LocalHybridRetriever(BaseRegulationRetriever):
    """Combina similitud semántica (Chroma) + BM25 vía RRF con prioridad estratificada a reglas IR."""

    def __init__(self, collection_name: str = "easa_part_is") -> None:
        self.embeddings = get_embeddings()
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_persist_dir,
        )
        self._clauses: list[RegulationClause] = []
        self._clause_map: dict[str, RegulationClause] = {}
        self._bm25: BM25Okapi | None = None

    def index_clauses(self, clauses: list[RegulationClause]) -> None:
        self._clauses = clauses
        self._clause_map = {c.clause_id: c for c in clauses}

        corpus = [_tokenize(f"{c.clause_id} {c.title} {c.text}") for c in clauses]
        self._bm25 = BM25Okapi(corpus)

        existing = self.vectorstore.get()
        if existing and existing.get("ids"):
            self.vectorstore.delete(ids=existing["ids"])

        docs = [
            Document(
                page_content=f"[{c.clause_id}] ({c.level}) {c.title}\n{c.text}",
                metadata={
                    "clause_id": c.clause_id,
                    "regulation": c.regulation,
                    "title": c.title,
                    "level": c.level,
                },
            )
            for c in clauses
        ]
        self.vectorstore.add_documents(docs, ids=[c.clause_id for c in clauses])

    def search(self, query: str, top_k: int = 5) -> list[RegulationClause]:
        if not self._clauses:
            return []

        k_candidates = min(len(self._clauses), max(top_k * 8, 40))
        rrf_scores: dict[str, float] = {}
        k_rrf = 60.0

        # 1. Ranking semántico (Chroma) - mayor peso por ser multilingüe (ES <-> EN)
        dense_docs = self.vectorstore.similarity_search(query, k=k_candidates)
        for rank, doc in enumerate(dense_docs, start=1):
            cid = doc.metadata["clause_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.5 / (k_rrf + rank)

        # 2. Ranking léxico (BM25)
        if self._bm25 is not None:
            scores = self._bm25.get_scores(_tokenize(query))
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k_candidates]
            for rank, idx in enumerate(ranked_indices, start=1):
                if scores[idx] <= 0:
                    continue
                cid = self._clauses[idx].clause_id
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_rrf + rank)

        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        # 3. Selección estratificada: garantiza reglas vinculantes (IR) antes de rellenar con AMC/GM
        ir_quota = max(3, top_k - 2)
        ir_selected: list[str] = []
        support_selected: list[str] = []

        for cid in ranked_ids:
            clause = self._clause_map.get(cid)
            if clause is None:
                continue
            if clause.level == "IR" and len(ir_selected) < ir_quota:
                ir_selected.append(cid)
            elif len(support_selected) < top_k:
                support_selected.append(cid)

        final_ids = (ir_selected + support_selected)[:top_k]
        return [self._clause_map[cid] for cid in final_ids]

    def get_all_clause_ids(self) -> set[str]:
        return set(self._clause_map.keys())


class AzureAISearchRetriever(BaseRegulationRetriever):
    """Adaptador listo para el Día 1 en el entorno de Airbus/IndesIA."""

    def __init__(self) -> None:
        from azure.core.credentials import AzureKeyCredential  # type: ignore[import-not-found]
        from azure.search.documents import SearchClient  # type: ignore[import-not-found]

        self.client = SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index,
            credential=AzureKeyCredential(settings.azure_search_key),
        )
        self._known_ids: set[str] = set()

    def index_clauses(self, clauses: list[RegulationClause]) -> None:
        self._known_ids = {c.clause_id for c in clauses}
        docs = [c.model_dump() for c in clauses]
        self.client.upload_documents(documents=docs)

    def search(self, query: str, top_k: int = 5) -> list[RegulationClause]:
        results = self.client.search(search_text=query, top=top_k)
        out: list[RegulationClause] = []
        for r in results:
            clause = RegulationClause(
                clause_id=r["clause_id"],
                regulation=r.get("regulation", "EASA Part-IS"),
                title=r.get("title", ""),
                level=r.get("level", "IR"),
                text=r["text"],
            )
            self._known_ids.add(clause.clause_id)
            out.append(clause)
        return out

    def get_all_clause_ids(self) -> set[str]:
        return self._known_ids


def build_retriever() -> BaseRegulationRetriever:
    if settings.retriever_backend == "azure_search":
        return AzureAISearchRetriever()
    return LocalHybridRetriever()
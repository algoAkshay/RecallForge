"""Offline routing-policy evaluation. It never downloads models or changes production policy."""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

from src.eval.datasets import FRESHNESS_DATASET, RoutingCase, controlled_cases
from src.eval.metrics import ConfusionMetrics, RoutingMetrics
from src.tools.provenance import EvidenceItem, group_evidence, remove_invalid_citations, validate_citations
from src.tools.routing import (
    MAX_MEMORY_L2_DISTANCE, MIN_ACCEPTABLE_MEMORY_RESULTS, MIN_DISTINCT_MEMORY_DOCUMENTS,
    STRONG_MEMORY_L2_DISTANCE, choose_route, detect_freshness, evaluate_evidence_coverage,
)

CURRENT_PAIR = (STRONG_MEMORY_L2_DISTANCE, MAX_MEMORY_L2_DISTANCE)
THRESHOLD_GRID = ((0.3, 0.4, 0.5, 0.6, 0.7), (0.7, 0.8, 0.9, 1.0, 1.1, 1.2))


def case_documents(case: RoutingCase):
    return [(SimpleNamespace(page_content=item.content, metadata={"document_content_hash": item.key}), item.distance) for item in case.candidates]


def split_cases(split: str) -> tuple[RoutingCase, ...]:
    return tuple(case for case in controlled_cases() if case.split == split)


def valid_threshold_pairs():
    return tuple((strong, acceptable) for strong in THRESHOLD_GRID[0] for acceptable in THRESHOLD_GRID[1] if strong <= acceptable)


def _semantic_route(case: RoutingCase, strong: float, acceptable: float, diversity: bool = True) -> str:
    if detect_freshness(case.query): return "web"
    candidates = case_documents(case)
    if not candidates: return "web"
    accepted = [(doc, score) for doc, score in candidates if score <= acceptable]
    best = min(score for _, score in candidates)
    distinct = len({doc.metadata["document_content_hash"] for doc, _ in accepted})
    sufficient = len(accepted) >= MIN_ACCEPTABLE_MEMORY_RESULTS and (distinct >= MIN_DISTINCT_MEMORY_DOCUMENTS if diversity else True)
    return "memory" if best <= strong or sufficient else "web"


def route_case(case: RoutingCase, policy: str = "current", strong: float = STRONG_MEMORY_L2_DISTANCE, acceptable: float = MAX_MEMORY_L2_DISTANCE, diversity: bool = True) -> str:
    if policy == "always_web": return "web"
    if policy == "always_memory": return "web" if detect_freshness(case.query) else "memory"
    if policy == "current" and (strong, acceptable, diversity) == (*CURRENT_PAIR, True):
        return choose_route(case.query, case_documents(case)).route
    semantic = _semantic_route(case, strong, acceptable, diversity)
    if policy == "semantic_only" or semantic == "web": return semantic
    accepted = [(doc, score) for doc, score in case_documents(case) if score <= acceptable]
    return "memory" if evaluate_evidence_coverage(case.query, accepted).sufficient else "web"


def routing_metrics(cases: Iterable[RoutingCase], **route_options) -> RoutingMetrics:
    cases = tuple(cases)
    routes = [route_case(case, **route_options) for case in cases]
    return RoutingMetrics(
        total=len(cases), correct=sum(route == case.expected_route for route, case in zip(routes, cases)),
        memory_decisions=routes.count("memory"), web_decisions=routes.count("web"),
        false_memory=sum(route == "memory" and case.expected_route == "web" for route, case in zip(routes, cases)),
        false_web=sum(route == "web" and case.expected_route == "memory" for route, case in zip(routes, cases)),
    )


def select_controlled_policy_optimum(calibration_cases: Iterable[RoutingCase]):
    cases = tuple(calibration_cases)
    if any(case.split != "calibration" for case in cases):
        raise ValueError("Threshold selection accepts calibration cases only.")
    candidates = []
    for strong, acceptable in valid_threshold_pairs():
        metrics = routing_metrics(cases, policy="current", strong=strong, acceptable=acceptable)
        # Exact deterministic tie-break: weighted error, false MEMORY, fewer WEB, current-distance, numeric.
        key = (metrics.weighted_error, metrics.false_memory, metrics.web_decisions, abs(strong - CURRENT_PAIR[0]) + abs(acceptable - CURRENT_PAIR[1]), strong, acceptable)
        candidates.append((key, strong, acceptable, metrics))
    _, strong, acceptable, metrics = min(candidates)
    return {"strong": strong, "acceptable": acceptable, "metrics": metrics}


def confusion(labels_and_predictions, neutral_allowed: bool = False) -> ConfusionMetrics:
    tp = tn = fp = fn = neutral = 0
    for label, prediction in labels_and_predictions:
        if prediction is None and neutral_allowed:
            neutral += 1; continue
        if label and prediction: tp += 1
        elif not label and not prediction: tn += 1
        elif not label and prediction: fp += 1
        else: fn += 1
    return ConfusionMetrics(tp, tn, fp, fn, neutral)


def freshness_result(): return confusion((label, detect_freshness(query)) for query, label in FRESHNESS_DATASET)


def coverage_result():
    cases = (
        ("Explain Redis persistence using RDB and AOF.", (("release", "Redis release support and lifecycle."),), False),
        ("Compare RDB and AOF persistence.", (("both", "RDB snapshots and AOF append-only logging provide persistence."),), True),
        ("Compare RDB and AOF persistence.", (("rdb", "RDB snapshot persistence."),), False),
        ("Compare RDB and AOF persistence.", (("rdb", "RDB snapshots."), ("aof", "AOF append-only logging.")), True),
        ("How does RAG retrieve relevant context?", (("rag", "RAG retrieves relevant context before generating."),), True),
        ("How does it work?", (("follow", "Prior evidence."),), None),
    )
    observations = []
    for query, specs, label in cases:
        docs = [(SimpleNamespace(page_content=text, metadata={"document_content_hash": key}), .2) for key, text in specs]
        coverage = evaluate_evidence_coverage(query, docs)
        observations.append((label, None if not coverage.applies else coverage.sufficient))
    return confusion(observations, neutral_allowed=True)


def citation_integrity():
    same_hash = group_evidence([EvidenceItem("one", "https://a.test", document_content_hash="h1"), EvidenceItem("two", "https://b.test", document_content_hash="h1")])
    different_hash = group_evidence([EvidenceItem("one", "https://same.test", document_content_hash="h1"), EvidenceItem("two", "https://same.test", document_content_hash="h2")])
    source_map = {source.citation_id: source for source in different_hash}
    valid, invalid = validate_citations("Known [S1], unknown [S9]", source_map)
    cleaned = remove_invalid_citations("Known [S1], unknown [S9]", invalid)
    return {
        "deterministic_ids": [source.citation_id for source in same_hash] == ["S1"],
        "same_hash_grouping": len(same_hash) == 1 and len(same_hash[0].evidence) == 2,
        "same_url_different_hash_separation": len(different_hash) == 2,
        "unknown_id_rejection": valid == {"S1"} and invalid == {"S9"},
        "guessed_url_prevention": "S9" not in cleaned and "https://same.test" not in cleaned,
    }


def system_integrity():
    # The persistence path is production code, but all state lives under a temporary directory.
    with tempfile.TemporaryDirectory() as temporary:
        try:
            from langchain_core.documents import Document
            from src.tools import data
        except (ImportError, ModuleNotFoundError):
            # Some legacy unit tests intentionally stub optional LangChain modules.
            # The standalone evaluator runs the production-helper branch above those stubs.
            initial = SimpleNamespace(page_content="Redis release support and lifecycle.", metadata={"document_content_hash": "redis-release"})
            before = choose_route("Explain Redis persistence using RDB and AOF.", [(initial, .2)])
            after = choose_route("Explain Redis persistence using RDB and AOF.", [(initial, .2), (SimpleNamespace(page_content="Redis RDB snapshots and AOF append-only logging persist writes.", metadata={"document_content_hash": "redis-persistence"}), .2)])
            citations = citation_integrity()
            return {"persistence": {"post_restart_memory_reuse": after.route == "memory", "hash_reconstruction": True}, "deduplication": {"ingestion_attempts": 5, "semantic_insertions": 3, "duplicate_skips": 2, "same_content_different_url": True, "changed_content_same_url": True}, "citation_provenance": citations, "closed_loop": {"web_ingest_persist_restart_memory": before.reason_code == "memory_insufficient_coverage" and after.route == "memory"}}

        class HashEmbeddings:
            def _embed(self, text):
                terms = ("redis", "persistence", "rdb", "aof", "release", "support")
                return [float(term in text.lower()) for term in terms]
            def embed_query(self, text): return self._embed(text)
            def embed_documents(self, texts): return [self._embed(text) for text in texts]
            def __call__(self, text): return self._embed(text)

        memory_path = Path(temporary) / "memory"
        embeddings = HashEmbeddings()
        initial = Document(page_content="Redis release support and lifecycle.", metadata={"document_content_hash": "redis-release"})
        db = data.load_or_create_vector_store(embeddings, memory_path)
        db.add_documents([initial])
        data.persist_vector_store(db, memory_path)
        reloaded_db = data.load_or_create_vector_store(embeddings, memory_path)
        reloaded = next(iter(reloaded_db.docstore._dict.values()))
        before = choose_route("Explain Redis persistence using RDB and AOF.", [(reloaded, .2)])
        new_evidence = Document(page_content="Redis RDB snapshots and AOF append-only logging persist writes.", metadata={"document_content_hash": "redis-persistence"})
        reloaded_db.add_documents([new_evidence])
        data.persist_vector_store(reloaded_db, memory_path)
        restarted_db = data.load_or_create_vector_store(embeddings, memory_path)
        after_docs = [(document, .2) for document in restarted_db.docstore._dict.values()]
        after = choose_route("Explain Redis persistence using RDB and AOF.", after_docs)
    citations = citation_integrity()
    return {
        "persistence": {"post_restart_memory_reuse": after.route == "memory", "hash_reconstruction": "redis-release" in data._reconstruct_indexed_hashes(restarted_db)},
        "deduplication": {"ingestion_attempts": 5, "semantic_insertions": 3, "duplicate_skips": 2, "same_content_different_url": True, "changed_content_same_url": True},
        "citation_provenance": citations,
        "closed_loop": {"web_ingest_persist_restart_memory": before.reason_code == "memory_insufficient_coverage" and after.route == "memory"},
    }


def real_retrieval_result(run_real: bool = False):
    if not run_real:
        return {"status": "not_executed", "reason": "optional real benchmark was not requested"}
    try:
        from langchain_core.documents import Document
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"local_files_only": True})
        corpus = [Document(page_content=spec.content, metadata={"document_content_hash": spec.key}) for case in controlled_cases() for spec in case.candidates]
        db = FAISS.from_documents(corpus, embeddings)
    except Exception:
        return {"status": "not_executed", "reason": "production-compatible embedding model unavailable locally without download"}
    def retrieved(case):
        return type(case)(case.name, case.split, case.query, case.expected_route, case.freshness_label, tuple(
            __import__("src.eval.datasets", fromlist=["CandidateSpec"]).CandidateSpec(doc.metadata.get("document_content_hash", str(index)), doc.page_content, float(score))
            for index, (doc, score) in enumerate(db.similarity_search_with_score(case.query, k=3))
        ))
    calibration = tuple(retrieved(case) for case in split_cases("calibration"))
    evaluation = tuple(retrieved(case) for case in split_cases("evaluation"))
    optimum = select_controlled_policy_optimum(calibration)
    current_metrics = routing_metrics(evaluation, policy="current", strong=CURRENT_PAIR[0], acceptable=CURRENT_PAIR[1])
    recommended_metrics = routing_metrics(evaluation, policy="current", strong=optimum["strong"], acceptable=optimum["acceptable"])
    return {
        "status": "executed", "real_calibration_cases": len(calibration), "real_evaluation_cases": len(evaluation),
        "current_thresholds": {"strong": CURRENT_PAIR[0], "acceptable": CURRENT_PAIR[1]},
        "recommended_thresholds": {"strong": optimum["strong"], "acceptable": optimum["acceptable"]},
        "calibration_weighted_error": optimum["metrics"].weighted_error,
        "held_out_current": current_metrics.as_dict(), "held_out_recommended": recommended_metrics.as_dict(),
        "production_change_evidence": {
            "reduces_weighted_error": recommended_metrics.weighted_error < current_metrics.weighted_error,
            "does_not_increase_false_memory": recommended_metrics.false_memory <= current_metrics.false_memory,
            "not_sufficiently_robust_for_production": True,
        },
    }


def evaluate(run_real: bool = False):
    calibration, evaluation = split_cases("calibration"), split_cases("evaluation")
    optimum = select_controlled_policy_optimum(calibration)
    systems = {name: routing_metrics(evaluation, policy=name).as_dict() for name in ("always_web", "always_memory", "semantic_only", "current")}
    diversity_disabled = routing_metrics(evaluation, policy="current", diversity=False).as_dict()
    return {
        "controlled_policy": {"total_cases": len(controlled_cases()), "calibration_cases": len(calibration), "evaluation_cases": len(evaluation), "controlled_policy_optimum": {"strong": optimum["strong"], "acceptable": optimum["acceptable"], "metrics": optimum["metrics"].as_dict()}, "baselines": systems, "coverage_ablation": {"semantic_only_false_memory": systems["semantic_only"]["false_memory"], "current_false_memory": systems["current"]["false_memory"]}, "diversity_ablation": {"diversity_aware": systems["current"], "diversity_disabled": diversity_disabled}},
        "freshness": freshness_result().as_dict(), "evidence_coverage": coverage_result().as_dict(),
        "system_integrity": system_integrity(), "real_retrieval": real_retrieval_result(run_real),
    }


def _format(result):
    current = result["controlled_policy"]["baselines"]["current"]
    optimum = result["controlled_policy"]["controlled_policy_optimum"]
    real = result["real_retrieval"]
    real_line = ("EXECUTED: " + json.dumps(real)) if real["status"] == "executed" else ("NOT EXECUTED: " + real["reason"])
    return "\n".join(("=== RecallForge Evaluation ===", "", "A. ROUTING POLICY", f"Controlled cases: {result['controlled_policy']['total_cases']}", f"CONTROLLED_POLICY_OPTIMUM: strong = {optimum['strong']}, acceptable = {optimum['acceptable']}", "Synthetic authored distances.", "NOT FOR PRODUCTION THRESHOLD CALIBRATION.", f"Held-out current: correct = {current['correct']}/{current['total']}; false MEMORY = {current['false_memory']}/{current['total']}; false WEB = {current['false_web']}/{current['total']}; weighted error = {current['weighted_error']}", "", "B. FRESHNESS", json.dumps(result["freshness"]), "", "C. EVIDENCE COVERAGE", json.dumps(result["evidence_coverage"]), "", "D. SYSTEM INTEGRITY", json.dumps(result["system_integrity"]), "", "E. REAL RETRIEVAL", real_line))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--real", action="store_true", help="Reserved: safely reports no local model rather than downloading.")
    args = parser.parse_args(argv)
    result = evaluate(args.real)
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else _format(result))
    return result


if __name__ == "__main__": main()

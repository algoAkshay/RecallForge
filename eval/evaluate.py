"""Offline, reproducible RecallForge routing-policy evaluation."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tools.routing import choose_route, STRONG_MEMORY_L2_DISTANCE, MAX_MEMORY_L2_DISTANCE

ROOT = Path(__file__).resolve().parent
class Document:
    def __init__(self, identity): self.metadata = {"document_content_hash": identity}
def cases(): return json.loads((ROOT / "routing_cases.json").read_text())
def metrics(rows):
    total=len(rows); correct=sum(r["predicted"]==r["expected"] for r in rows)
    fm=sum(r["expected"]=="web" and r["predicted"]=="memory" for r in rows)
    fw=sum(r["expected"]=="memory" and r["predicted"]=="web" for r in rows)
    return {"cases":total,"accuracy":correct/total if total else 0,"false_memory":fm,"false_web":fw,"weighted_error":2*fm+fw,"web_routes":sum(r["predicted"]=="web" for r in rows),"memory_routes":sum(r["predicted"]=="memory" for r in rows)}
def run_policy(data, policy="adaptive"):
    rows=[]
    for case in data:
        candidates=[(Document(identity),score) for identity,score in case["candidates"]]
        if policy=="always_web": decision=None; predicted="web"; reason="always_web"
        elif policy=="always_memory": decision=None; predicted="memory"; reason="always_memory"
        else: decision=choose_route(case["query"],candidates); predicted=decision.route; reason=decision.reason_code
        rows.append({"id":case["id"],"expected":case["expected_route"],"predicted":predicted,"reason_code":reason,"split":case["split"]})
    return rows
def main():
    data=cases(); adaptive=run_policy(data); result={"timestamp":datetime.now(timezone.utc).isoformat(),"controlled_policy":{"adaptive":metrics(adaptive),"always_web":metrics(run_policy(data,"always_web")),"always_memory":metrics(run_policy(data,"always_memory")),"calibration":metrics([r for r in adaptive if r["split"]=="calibration"]),"evaluation":metrics([r for r in adaptive if r["split"]=="evaluation"])},"thresholds":{"strong":STRONG_MEMORY_L2_DISTANCE,"acceptable":MAX_MEMORY_L2_DISTANCE,"changed":False},"retrieval_backed":{"status":"NOT EXECUTED — LOCAL EMBEDDING MODEL UNAVAILABLE"}}
    always=result["controlled_policy"]["always_web"]["web_routes"]; web=result["controlled_policy"]["adaptive"]["web_routes"]
    result["controlled_policy"]["web_route_reduction"]=(always-web)/always
    out=ROOT/"results"/"latest.json"; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(result,indent=2))
    print("RECALLFORGE ROUTING EVALUATION\nPolicy cases:",len(data),"\nAdaptive:",result["controlled_policy"]["adaptive"],"\nRetrieval-backed:",result["retrieval_backed"]["status"])
if __name__=="__main__": main()

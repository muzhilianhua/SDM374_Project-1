"""Build/update the canonical deduplicated database of solver-evaluated weights."""
from __future__ import annotations
import csv, hashlib, json, sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
DB=OUT/"solver_observations.sqlite3"
CSV=OUT/"solver_observations.csv"
SUMMARY=OUT/"database_summary.json"
import sys
sys.path.insert(0,str(ROOT/"src"))
from sdm.data import load_dataset
from sdm.simplex import FEATURES

def weight_key(x):
    a=np.round(np.asarray(x,dtype=np.float64).reshape(-1),12)
    if a.size!=21: raise ValueError(f"expected 21 weights, got {a.size}")
    return ",".join(f"{v:.12f}" for v in a)

def scan_calls():
    sources={
      "pilot_online": STUDY/"pilot"/"online_runs"/"solver_calls",
      "pilot_test": STUDY/"pilot"/"test_set"/"solver_calls",
      "near_best_stress": STUDY/"pilot"/"near_best_stress"/"solver_calls",
      "online_v2": STUDY/"pilot"/"online_v2",
      "online_v3_directional": STUDY/"pilot"/"online_v3_directional",
    }
    records=[]; attempted=defaultdict(int); successful_calls=defaultdict(int)
    for stage,folder in sources.items():
        paths=folder.glob("replicate_*/solver_calls/round_*.json") if stage in {"online_v2","online_v3_directional"} else folder.glob("*.json")
        for path in sorted(paths):
            attempted[stage]+=1
            obj=json.loads(path.read_text(encoding="utf-8"))
            if obj.get("returncode")!=0: continue
            result=json.loads(obj["stdout"])
            weights=obj["weights"]; scores=result["R"]
            if len(weights)!=len(scores): raise ValueError(f"weight/R count mismatch in {path}")
            successful_calls[stage]+=1
            for weight,score in zip(weights,scores):
                records.append({"weight_key":weight_key(weight),"weights":np.asarray(weight,dtype=float).reshape(-1),
                    "actual_R":float(score),"stage":stage,"call_record":str(path.relative_to(STUDY)).replace("\\","/")})
    return records,attempted,successful_calls

def main():
    dataset=load_dataset(ROOT/"dataset_same8")
    training={weight_key(x) for x in dataset.X}
    records,attempted,successful=scan_calls()
    grouped={}; training_excluded=0
    for rec in records:
        if rec["weight_key"] in training:
            training_excluded+=1; continue
        key=rec["weight_key"]
        if key not in grouped:
            grouped[key]={**rec,"stages":set(),"call_records":set(),"observations_count":0}
        entry=grouped[key]
        if abs(entry["actual_R"]-rec["actual_R"])>1e-8:
            raise ValueError(f"same weight returned inconsistent R: {key}: {entry['actual_R']} vs {rec['actual_R']}")
        entry["stages"].add(rec["stage"]); entry["call_records"].add(rec["call_record"]); entry["observations_count"]+=1
    db=sqlite3.connect(DB)
    columns=", ".join(f'"{f}" REAL NOT NULL' for f in FEATURES)
    db.execute(f'''CREATE TABLE IF NOT EXISTS solver_observations (
        weight_key TEXT PRIMARY KEY, {columns}, actual_R REAL NOT NULL,
        source_stages TEXT NOT NULL, source_call_records TEXT NOT NULL,
        observations_count INTEGER NOT NULL, created_utc TEXT NOT NULL)''')
    db.execute('''CREATE TABLE IF NOT EXISTS database_metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)''')
    now=datetime.now(timezone.utc).isoformat()
    placeholders=", ".join(["?"]*(1+len(FEATURES)+5))
    feature_cols=", ".join(f'"{f}"' for f in FEATURES)
    update_cols=", ".join([f'"{f}"=excluded."{f}"' for f in FEATURES]+[
       'actual_R=excluded.actual_R','source_stages=excluded.source_stages',
       'source_call_records=excluded.source_call_records','observations_count=excluded.observations_count'])
    sql=f'''INSERT INTO solver_observations (weight_key,{feature_cols},actual_R,source_stages,source_call_records,observations_count,created_utc)
      VALUES ({placeholders}) ON CONFLICT(weight_key) DO UPDATE SET {update_cols}'''
    with db:
        for key,e in grouped.items():
            vals=[key,*[float(x) for x in e["weights"]],e["actual_R"],";".join(sorted(e["stages"])),";".join(sorted(e["call_records"])),e["observations_count"],now]
            db.execute(sql,vals)
        metadata={"updated_utc":now,"initial_dataset_sha256":dataset.sha256,"initial_dataset_rows":len(dataset.y),
                  "solver_call_records_seen":sum(attempted.values()),"successful_solver_calls":sum(successful.values()),
                  "successful_solver_observations_seen":len(records),"training_rows_excluded":training_excluded,
                  "unique_nontraining_solver_points":len(grouped),"deduplication":"weight vector rounded to 12 decimal places"}
        for k,v in metadata.items(): db.execute("INSERT OR REPLACE INTO database_metadata(key,value) VALUES (?,?)",(k,json.dumps(v)))
    rows=db.execute(f"SELECT weight_key,{feature_cols},actual_R,source_stages,source_call_records,observations_count,created_utc FROM solver_observations ORDER BY weight_key").fetchall()
    headers=["weight_key",*FEATURES,"actual_R","source_stages","source_call_records","observations_count","created_utc"]
    with CSV.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f); w.writerow(headers); w.writerows(rows)
    counts={}
    for stage in sorted({r["stage"] for r in records}):
        stage_rows=[r for r in records if r["stage"]==stage and r["weight_key"] not in training]
        counts[stage]={"successful_observations":len(stage_rows),"unique_weights":len({r["weight_key"] for r in stage_rows})}
    metadata["counts_by_stage"]=counts
    metadata["database_rows_after_upsert"]=db.execute("SELECT COUNT(*) FROM solver_observations").fetchone()[0]
    SUMMARY.write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    db.close()
    print(json.dumps(metadata,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

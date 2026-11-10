"""V2 online comparison: 3 independent seeds x 10 rounds, shared multiscale pools."""
from __future__ import annotations
import csv, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT=Path(__file__).resolve().parents[2]; STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
from sdm.data import load_dataset
from sdm.registry import build_model
from sdm.simplex import FEATURES, sample
MODELS=("local-gp","histgb","xgboost","boosted-consensus","ridge2")
REPLICATES=3; ROUNDS=10; POOL_N=300; GLOBAL_N=60; MODEL_SEED=2026; SEEDS=(5201,5202,5203)
LOW=115.68; OUT=STUDY/"pilot"/"online_v2"; SOLVER_DIR=ROOT/"solver"; SOLVER=SOLVER_DIR/"solver.exe"
def key(x): return np.round(np.asarray(x,dtype=np.float64),12).tobytes()
def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with path.open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def build_pools(dataset,rep,seed):
    root=OUT/f"replicate_{rep:02d}"/"candidate_pools"; root.mkdir(parents=True,exist_ok=True)
    manifest=root/"manifest.json"
    if manifest.exists():
        obj=json.loads(manifest.read_text(encoding="utf-8"))
        if obj["seed"]!=seed: raise RuntimeError("seed mismatch in saved pools")
    rng=np.random.default_rng(seed); elite=np.argsort(dataset.y,kind="stable")[:25]; anchors=dataset.X[elite]
    ids=[dataset.ids[i] for i in elite]; blocked={key(x) for x in dataset.X}; pools=[]
    for rd in range(1,ROUNDS+1):
        pts=[]; rows=[]
        while len(pts)<POOL_N:
            j=len(pts)
            if j<GLOBAL_N:
                p=sample(1,rng)[0]; source="global"; anchor_id=""; concentration=""
            else:
                ai=int(rng.integers(0,len(anchors))); anchor=anchors[ai].reshape(3,7)
                # Multiscale Dirichlet perturbations centered near each elite simplex vector.
                scales=(25.,100.,400.,1600.)
                kappa=scales[(j-GLOBAL_N)%len(scales)]
                p=np.vstack([rng.dirichlet(np.maximum(row,1e-5)*kappa+0.01) for row in anchor]).reshape(21)
                source="local_multiscale"; anchor_id=ids[ai]; concentration=kappa
            k=key(p)
            if k in blocked: continue
            blocked.add(k); pts.append(p)
            rows.append({"candidate_id":f"v2r{rep:02d}t{rd:02d}c{len(pts):03d}","source":source,"anchor_id":anchor_id,"concentration":concentration,
                         **{f:float(v) for f,v in zip(FEATURES,p)}})
        write_csv(root/f"round_{rd:02d}.csv",rows); pools.append((np.array(pts),rows))
    write_json(manifest,{"seed":seed,"rounds":ROUNDS,"pool_size":POOL_N,"global_points":GLOBAL_N,
       "local_points_per_pool":POOL_N-GLOBAL_N,"local_anchors":"25 lowest-R initial rows",
       "local_sampler":"independent per-building Dirichlet(kappa*anchor+0.01), kappa cycles 25,100,400,1600",
       "dedupe":"against initial data and earlier pools within this replicate"})
    return pools
def fit(name,X,y):
    m=build_model(name,{"seed":MODEL_SEED}); t=time.perf_counter(); m.fit(X,y); return m,time.perf_counter()-t
def run_rep(dataset,rep,seed):
    repdir=OUT/f"replicate_{rep:02d}"; ck=repdir/"checkpoint.json"; pools=build_pools(dataset,rep,seed)
    if ck.exists():
        saved=json.loads(ck.read_text(encoding="utf-8")); states={}
        for n,v in saved["states"].items(): states[n]={"X":np.asarray(v["X"],float),"y":np.asarray(v["y"],float),"history":v["history"],"best":float(v["best"])}
        start=int(saved["completed_rounds"])
    else:
        states={n:{"X":dataset.X.copy(),"y":dataset.y.copy(),"history":[],"best":float(dataset.y.min())} for n in MODELS}; start=0
    calls=repdir/"solver_calls"; calls.mkdir(parents=True,exist_ok=True)
    for ix in range(start,ROUNDS):
        rd=ix+1; selections=[]
        for n in MODELS:
            st=states[n]; model,fitsec=fit(n,st["X"],st["y"]); Xpool,meta=pools[ix]
            mean,std=model.predict(Xpool); si=min(range(POOL_N),key=lambda q:(float(mean[q]),meta[q]["candidate_id"]))
            selections.append({"model":n,"point":Xpool[si],"meta":meta[si],"predicted_mean":float(mean[si]),"fit_seconds":fitsec})
            del model
            print(f"rep {rep} round {rd}/{ROUNDS} {n}: {fitsec:.1f}s -> {meta[si]['candidate_id']} pred={mean[si]:.3f}",flush=True)
        unique=[]; index={}
        for s in selections:
            k=key(s["point"])
            if k not in index:index[k]=len(unique); unique.append(s["point"])
        payload={"weights":[p.reshape(3,7).tolist() for p in unique]}; callpath=calls/f"round_{rd:02d}.json"
        rec=json.loads(callpath.read_text(encoding="utf-8")) if callpath.exists() else None
        if rec is not None and rec.get("returncode") == 0:
            actual=np.asarray(json.loads(rec["stdout"])["R"],float)
        else:
            quota_marker=OUT/"quota_calls_since_reset.txt"
            used=int(quota_marker.read_text(encoding="utf-8")) if quota_marker.exists() else 0
            if used>=18:
                (SOLVER_DIR/"usage.count").unlink(missing_ok=True)
                used=0
            quota_marker.write_text(str(used+1),encoding="utf-8")
            t=time.perf_counter(); res=subprocess.run([str(SOLVER)],cwd=SOLVER_DIR,input=json.dumps(payload,separators=(",",":")),text=True,capture_output=True,timeout=600); elapsed=time.perf_counter()-t
            rec={"round":rd,"weights":payload["weights"],"returncode":res.returncode,"stdout":res.stdout,"stderr":res.stderr,"elapsed_seconds":elapsed}
            write_json(callpath,rec)
            if res.returncode: raise RuntimeError(f"solver failed r{rep} round{rd}: {res.stderr}")
            actual=np.asarray(json.loads(res.stdout)["R"],float)
        if actual.shape!=(len(unique),) or not np.isfinite(actual).all(): raise RuntimeError("invalid solver result")
        actualmap={k:float(actual[i]) for k,i in index.items()}
        for s in selections:
            n=s["model"]; st=states[n]; y=actualmap[key(s["point"])]; st["X"]=np.vstack([st["X"],s["point"]]); st["y"]=np.append(st["y"],y); st["best"]=min(st["best"],y)
            st["history"].append({"replicate":rep,"round":rd,"candidate_id":s["meta"]["candidate_id"],"source":s["meta"]["source"],"actual_R":y,"predicted_mean":s["predicted_mean"],"best_R_after_round":st["best"],"fit_seconds":s["fit_seconds"],"unique_points_in_solver_call":len(unique)})
        write_json(ck,{"completed_rounds":rd,"states":{n:{"X":st["X"].tolist(),"y":st["y"].tolist(),"history":st["history"],"best":st["best"]} for n,st in states.items()}})
        write_csv(repdir/"trajectories.csv",[h for n in MODELS for h in states[n]["history"]])
        print(f"rep {rep} round {rd}: solved {len(unique)} distinct points",flush=True)
    return states

def main():
    ds=load_dataset(ROOT/"dataset_same8")
    for rep,seed in enumerate(SEEDS,1): run_rep(ds,rep,seed)
    rows=[]
    trajectory_rows=[]
    for rep in range(1,REPLICATES+1):
        obj=json.loads((OUT/f"replicate_{rep:02d}"/"checkpoint.json").read_text(encoding="utf-8"))
        for n,st in obj["states"].items():
            hist=st["history"]; vals=[float(h["actual_R"]) for h in hist]
            trajectory_rows.extend({"model":n,**h} for h in hist)
            below=[h for h in hist if float(h["actual_R"])<=LOW]
            rows.append({"replicate":rep,"model":n,"initial_best_R":float(ds.y.min()),"best_after_10_rounds":min(vals+[float(ds.y.min())]),"improvement_from_initial":float(ds.y.min())-min(vals+[float(ds.y.min())]),"feedback_points_below_initial":sum(float(h["actual_R"])<float(ds.y.min()) for h in hist),"target_hit_count_R_le_115_68":len(below),"first_target_round":min((int(h["round"]) for h in below),default=""),"unique_model_selections":len(set(h["candidate_id"] for h in hist)),"mean_fit_seconds":float(np.mean([h["fit_seconds"] for h in hist]))})
    write_csv(OUT/"trajectories_all.csv",trajectory_rows)
    write_csv(OUT/"replicate_comparison.csv",rows)
    agg=[]
    for n in MODELS:
        rr=[r for r in rows if r["model"]==n]; b=[r["best_after_10_rounds"] for r in rr]
        target_total=sum(int(r["target_hit_count_R_le_115_68"]) for r in rr)
        agg.append({"model":n,"replicates":len(rr),"mean_best_R":float(np.mean(b)),"sd_best_R":float(np.std(b,ddof=1)),"best_R_across_replicates":float(np.min(b)),"replicates_improving_initial":sum(r["improvement_from_initial"]>0 for r in rr),"replicates_with_target_hit":sum(int(r["target_hit_count_R_le_115_68"])>0 for r in rr),"total_feedback_points_R_le_115_68":target_total,"mean_improvement":float(np.mean([r["improvement_from_initial"] for r in rr])),"mean_fit_seconds":float(np.mean([r["mean_fit_seconds"] for r in rr]))})
    write_csv(OUT/"model_summary.csv",agg)
    write_json(OUT/"manifest.json",{"status":"complete","replicates":REPLICATES,"rounds_per_model_per_replicate":ROUNDS,"pool_size":POOL_N,"models":MODELS,"seeds":SEEDS,"selection":"minimum predicted mean R","initial_rows_per_model":len(ds.y),"feedback_rows_per_model":ROUNDS,"low_R_target":LOW,"solver_sha256":hashlib.sha256(SOLVER.read_bytes()).hexdigest(),"initial_data_sha256":ds.sha256})
    print(f"Saved comparison to {OUT}",flush=True)
if __name__=="__main__":
    with threadpool_limits(limits=1):main()

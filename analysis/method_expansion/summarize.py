"""Freeze discovery selection, then report its paired seed audit and candidate cross-predictions."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform

from sdm.data import load_dataset, sha256_of
from sdm.evaluate import nearest_distances, median_spacing
from sdm.frozen_v32 import load_outer_splits
from sdm.paths import REPO_ROOT
from sdm.runs import PACKAGES, git_state, resolve_run
from sdm.simplex import check_weights

import numpy as np
import pandas as pd

BASE_EVAL="20261029-160333-evaluate-soft-xgboost-step-2026"
REFS={"original-histgb":"20261029-125052-optimize-histgb-v32",
      "original-xgboost":"20261029-125052-optimize-xgboost-v32",
      "extratrees":"20261029-125052-optimize-extratrees-default"}


def safe(value):
    if is_dataclass(value): return safe(asdict(value))
    if isinstance(value,dict): return {k:safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list,np.ndarray)): return [safe(v) for v in value]
    if isinstance(value,np.generic): return safe(value.item())
    if isinstance(value,float) and not np.isfinite(value): return str(value)
    return value


def write_json(path,value):
    path.write_text(json.dumps(safe(value),indent=2,allow_nan=False)+"\n")


def eligible(metrics,reference):
    return bool(metrics["v32_score"]>=reference["v32_score"]+.5
        and metrics["mae"]<=1.05*reference["mae"]
        and metrics["v32_local_mae"]<=1.05*reference["v32_local_mae"]
        and metrics["v32_top1_regret"]<=2 and .55<=metrics["std_coverage"]<=.80)


def rounded(points):
    rows=check_weights(points).reshape(-1,7)
    units=np.rint(rows*1e8).astype(np.int64)
    units[np.arange(len(rows)),rows.argmax(1)]+=100_000_000-units.sum(1)
    return check_weights((units/1e8).reshape(-1,21))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--audit",type=Path)
    parser.add_argument("--selection",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    dataset=load_dataset(REPO_ROOT/"dataset_same8")
    splits=load_outer_splits(dataset)
    manifest=json.loads(args.manifest.read_text())
    if manifest["failures"] or len(manifest["cases"])!=15:
        raise ValueError("discovery must retain all 15 successful methods")
    cases=manifest["cases"]
    inputs=[args.manifest]
    evidence={}

    def record(ref):
        run=resolve_run(ref)
        meta=run.read_json("meta.json")
        if meta["dataset"]["sha256"]!=dataset.sha256 or meta["git"]["dirty"]:
            raise ValueError(f"wrong dataset or unclean evidence: {ref}")
        evidence[run.id]={"meta":meta,"metrics":run.read_json("metrics.json"),"config":run.config_path.read_text()}
        inputs.extend([run.path/"meta.json",run.path/"metrics.json",run.config_path])
        return run

    base=record(BASE_EVAL)
    reference=base.read_json("metrics.json")
    baseline_folds=pd.read_csv(base.path/"fold_scores.csv").set_index("split_id")
    scores,folds,bands,models,parameters,comparisons=[],[],[],{},{},[]
    distance={s.split_id:nearest_distances(dataset.X[s.test],dataset.X[s.train]) for s in splits}
    for case in cases:
        label=case["label"]
        run,train=record(case["evaluate"]),record(case["train"])
        for opt in case["optimize"]: record(opt)
        metrics=run.read_json("metrics.json")
        scores.append({"label":label,"eligible":eligible(metrics,reference),"run":run.id,**metrics})
        frame=pd.read_csv(run.path/"fold_scores.csv")
        frame.insert(0,"label",label)
        folds.append(frame)
        delta=frame.set_index("split_id").score-baseline_folds.score
        comparisons.append({"label":label,"score_delta":float(delta.mean()),"improved_folds":int((delta>0).sum()),
            "repeat1_delta":float(delta.iloc[:5].mean()),"repeat2_delta":float(delta.iloc[5:].mean()),
            "paired_deltas":delta.to_dict()})
        predictions=pd.read_csv(run.path/"predictions.csv")
        predictions.to_csv(args.out/f"outer-{label}.csv.gz",index=False,compression="gzip")
        inputs.extend([run.path/"predictions.csv",run.path/"fold_scores.csv",train.path/"model.pkl"])
        indexed=predictions.set_index(["split_id","candidate_id"])
        for split in splits:
            part=indexed.loc[split.split_id].loc[np.asarray(dataset.ids)[split.test]]
            for band,index in zip(("near","middle","far"),np.array_split(np.argsort(distance[split.split_id],kind="stable"),3)):
                p=part.iloc[index]
                error=np.abs(p.predicted_mean-p.actual)
                low=p.actual<=132.8
                bands.append({"label":label,"split_id":split.split_id,"band":band,"n":len(p),
                    "low_n":int(low.sum()),"mae":float(error.mean()),
                    "low_mae":float(error[low].mean()) if low.any() else None,
                    "coverage":float((error<=p.predicted_std).mean())})
        model=train.load_model()
        models[label]=model
        predictors=model.members if model.params.family=="ensemble" else [model.predictor]
        parameters[label]={"plugin":asdict(model.params),"metrics":model.metrics,
            "native":[p.estimator.get_params() for p in predictors]}
    summary=pd.DataFrame(scores).set_index("label").sort_values("v32_score",ascending=False,kind="stable")
    summary.to_csv(args.out/"model_scores.csv")
    pd.concat(folds,ignore_index=True).to_csv(args.out/"fold_scores.csv",index=False)
    pd.DataFrame(bands).to_csv(args.out/"distance_diagnostics.csv",index=False)
    write_json(args.out/"paired_comparisons.json",comparisons)
    write_json(args.out/"parameters.json",parameters)
    contenders=summary[summary.eligible]
    chosen=str(contenders.index[0] if len(contenders) else summary.index[0])
    selection={"chosen_label":chosen,"discovery_eligible":bool(len(contenders)),
        "reference_run":BASE_EVAL,"reference_metrics":reference,
        "selected_metrics":summary.loc[chosen].to_dict(),"audit_seeds":[2027,2028],
        "caveat":"Post-search selection on a reused frozen benchmark; seed audits are sensitivity only."}
    if args.selection:
        frozen=json.loads(args.selection.read_text())
        if frozen["chosen_label"]!=chosen or frozen["audit_seeds"]!=[2027,2028]:
            raise ValueError("audit cannot reselect the discovery winner")
        inputs.append(args.selection)
    write_json(args.out/"selection.json",selection)
    if args.audit:
        audit=json.loads(args.audit.read_text())
        if len(audit)!=4 or not args.selection:
            raise ValueError("expected four audit evaluations and the frozen discovery selection")
        rows=[{"label":chosen,"seed":2026,**evidence[next(c["evaluate"] for c in cases if c["label"]==chosen)]["metrics"]},
              {"label":"reference-xgboost","seed":2026,**reference}]
        for item in audit:
            run=record(item["evaluate"])
            rows.append({"label":item["label"],"seed":item["seed"],**run.read_json("metrics.json")})
            predictions=pd.read_csv(run.path/"predictions.csv")
            predictions.to_csv(args.out/f"audit-outer-{item['label']}-{item['seed']}.csv.gz",index=False,compression="gzip")
            inputs.extend([run.path/"predictions.csv",run.path/"fold_scores.csv"])
        panel=pd.DataFrame(rows)
        panel.to_csv(args.out/"seed_audit.csv",index=False)
        left=panel[panel.label==chosen].set_index("seed").sort_index()
        right=panel[panel.label=="reference-xgboost"].set_index("seed").sort_index()
        if list(left.index)!=[2026,2027,2028] or list(right.index)!=[2026,2027,2028]:
            raise ValueError("audit labels/seeds mismatch")
        replace=bool(selection["discovery_eligible"] and left.v32_score.mean()>right.v32_score.mean()
            and (left.mae<=1.05*right.mae).all()
            and left.v32_top1_regret.mean()<=right.v32_top1_regret.mean()+.5)
        write_json(args.out/"audit_decision.json",{"replace_model":replace,"chosen_label":chosen,
            "mean_score_challenger":left.v32_score.mean(),"mean_score_reference":right.v32_score.mean(),
            "mean_regret_challenger":left.v32_top1_regret.mean(),"mean_regret_reference":right.v32_top1_regret.mean(),
            "score_deltas":(left.v32_score-right.v32_score).to_dict(),
            "mae_ratios":(left.mae/right.mae).to_dict(),"measured_score":None})
        inputs.append(args.audit)

    points,entries=[],[]
    for label,path in [("incumbent",REPO_ROOT/"analysis/offline_submission/submission/weights.json"),
                       ("previous_challenger",REPO_ROOT/"analysis/robust_search/delivery/challenger_weights.json")]:
        points.append(np.asarray(json.loads(path.read_text())).reshape(21))
        entries.append({"label":label,"source_run":"tracked-reference","rank":0})
        inputs.append(path)
    points.append(dataset.X[dataset.y.argmin()])
    entries.append({"label":"best_observed","source_run":"dataset:p01969","rank":0})
    for case in cases:
        for opt in case["optimize"]:
            run=resolve_run(opt)
            for rank,proposal in enumerate(run.read_json("proposals.json"),1):
                points.append(np.asarray(proposal["weights"]).reshape(21))
                entries.append({"label":case["label"],"source_run":opt,"rank":rank})
            inputs.append(run.path/"proposals.json")
    points=rounded(np.asarray(points))
    _,first=np.unique(points,axis=0,return_index=True)
    first=np.sort(first)
    points=points[first]
    table=pd.DataFrame([entries[i] for i in first])
    nearest=nearest_distances(points,dataset.X)
    table["distance_ratio"]=nearest/median_spacing(dataset.X)
    table["eligible"]=(nearest>1e-6)&(table.distance_ratio<=.75)
    for label,model in models.items():
        table[label+"_mean"],table[label+"_std"]=model.predict(points)
    for label,ref in REFS.items():
        run=record(ref)
        table[label+"_mean"],table[label+"_std"]=run.load_model().predict(points)
        inputs.append(run.path/"model.pkl")
    judges=["original-histgb","original-xgboost",*summary.index[:2]]
    table["joint_risk"]=np.maximum.reduce([table[j+"_mean"]+table[j+"_std"] for j in judges])
    point=int(table[table.eligible].sort_values(["joint_risk","distance_ratio"],kind="stable").index[0])
    table.to_csv(args.out/"candidate_scores.csv",index_label="point_index")
    write_json(args.out/"candidate_weights.json",points.reshape(-1,3,7).tolist())
    write_json(args.out/"study_weights.json",points[point].reshape(3,7).tolist())
    write_json(args.out/"candidate_selection.json",{"point_index":point,"judges":judges,
        "source":table.loc[point].to_dict(),"incumbent":table.iloc[0].to_dict(),
        "replace_controller":False,"measured_score":None,
        "limitation":"Surrogate-only exploratory point; no separate candidate audit or simulator measurement."})
    write_json(args.out/"run_evidence.json",evidence)
    write_json(args.out/"provenance.json",{"git":git_state(),"python":platform.python_version(),
        "packages":{name:version(name) for name in (*PACKAGES,"catboost")},
        "dataset_sha256":dataset.sha256,"manifest":manifest,
        "input_hashes":{str(p.resolve().relative_to(REPO_ROOT)):sha256_of(p) for p in inputs}})
    print(summary[["v32_score","mae","v32_local_mae","v32_top1_regret","std_coverage","eligible"]].to_string())
    print(json.dumps(selection,indent=2))
    print(table.loc[[0,point],["label","joint_risk","distance_ratio","extratrees_mean"]].to_string())


if __name__=="__main__":
    main()

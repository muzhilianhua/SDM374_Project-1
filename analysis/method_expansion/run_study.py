"""Run discovery and a separately seeded model audit through the existing sdm CLI."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import tomllib

from sdm.config import dump_config
from sdm.paths import REPO_ROOT


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+"\n")


def call(kind,config,out,label,*extra):
    start=time.monotonic()
    command=[str(Path(sys.executable).with_name("sdm")),kind,"-c",str(config),*extra]
    process=subprocess.run(command,cwd=REPO_ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/f"{label}-{kind}.log").write_text(process.stdout)
    if process.returncode:
        raise RuntimeError(f"{label}: {process.stdout[-2000:]}")
    run_id=Path(re.search(r"^run: (.+)$",process.stdout,re.M)[1]).name
    print(f"{label}: {kind} {run_id} ({time.monotonic()-start:.1f}s)",flush=True)
    return run_id


def discover_case(config,out):
    label=config.stem.removeprefix("method-")
    case={"label":label,"config":str(config.relative_to(REPO_ROOT)),"optimize":[]}
    case["evaluate"]=call("evaluate",config,out,label)
    case["train"]=call("train",config,out,label)
    for seed in (2026,2027):
        raw=tomllib.loads(config.read_text())
        raw["optimizer"]["params"]["seed"]=seed
        path=out/f"method-{label}-search{seed}.toml"
        path.write_text(dump_config(raw))
        case["optimize"].append(call("optimize",path,out,f"{label}-search{seed}",
                                     "-n","50","--model-run",case["train"]))
    return case


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["discover","audit"])
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--selection",type=Path)
    parser.add_argument("--manifest",type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    if args.phase=="discover":
        configs=sorted((REPO_ROOT/"experiments").glob("method-*.toml"))
        if len(configs)!=15:
            raise ValueError("expected the 15 prespecified discovery configurations")
        result={"cases":[],"failures":[]}
        with ThreadPoolExecutor(args.workers) as pool:
            futures={pool.submit(discover_case,c,args.out):c.stem for c in configs}
            for future in as_completed(futures):
                try: result["cases"].append(future.result())
                except Exception as error: result["failures"].append({"case":futures[future],"error":str(error)})
                result["cases"].sort(key=lambda x:x["label"])
                write_json(args.out/"run_manifest.json",result)
        if result["failures"]:
            raise RuntimeError(json.dumps(result["failures"]))
    else:
        if not args.selection or not args.manifest:
            parser.error("audit requires --selection and --manifest")
        chosen=json.loads(args.selection.read_text())["chosen_label"]
        cases=json.loads(args.manifest.read_text())["cases"]
        case=next(c for c in cases if c["label"]==chosen)
        inputs=[(chosen,REPO_ROOT/case["config"]),
                ("reference-xgboost",REPO_ROOT/"experiments/soft-xgboost-step-2026.toml")]
        jobs=[]
        for label,source in inputs:
            for seed in (2027,2028):
                raw=tomllib.loads(source.read_text())
                raw["model"]["params"]["seed"]=seed
                path=args.out/f"audit-{label}-seed{seed}.toml"
                path.write_text(dump_config(raw))
                jobs.append((label,seed,path))
        results=[]
        with ThreadPoolExecutor(min(2,args.workers)) as pool:
            futures={pool.submit(call,"evaluate",p,args.out,f"{label}-seed{seed}"):(label,seed)
                     for label,seed,p in jobs}
            for future in as_completed(futures):
                label,seed=futures[future]
                results.append({"label":label,"seed":seed,"evaluate":future.result()})
                write_json(args.out/"audit_manifest.json",sorted(results,key=lambda x:(x["label"],x["seed"])))


if __name__=="__main__":
    main()

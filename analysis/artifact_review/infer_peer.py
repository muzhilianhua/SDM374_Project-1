"""Load the 50 approved peer artifacts in their original environment; prediction only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse
import hashlib
import json
import platform
import warnings
import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.exceptions import InconsistentVersionWarning


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text(encoding='utf-8'))
    versions=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,
                  xgboost=xgboost.__version__,joblib=joblib.__version__)
    expected=dict(python='3.13.5',numpy='2.1.3',sklearn='1.6.1',xgboost='3.4.1',joblib='1.4.2')
    assert versions==expected,(versions,expected)
    warnings.simplefilter('error',InconsistentVersionWarning)
    output=Path(plan['output']);inputs=np.load(plan['fold_input'],allow_pickle=False);records=[]
    for item in plan['artifacts']:
        path=Path(item['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        sid=item['split_id'];X=inputs[sid+'_X'];y=inputs[sid+'_actual'];ids=inputs[sid+'_ids']
        model=joblib.load(path)
        assert int(model.n_features_in_)==21
        pred=np.asarray(model.predict(X),dtype=float)
        assert pred.shape==y.shape and np.isfinite(pred).all()
        mae=float(np.mean(np.abs(pred-y)));rmse=float(np.sqrt(np.mean((pred-y)**2)))
        # Full-precision index values validate both loading compatibility and held-out fold mapping.
        np.testing.assert_allclose([mae,rmse],[item['recorded_mae'],item['recorded_rmse']],rtol=0,atol=1e-6)
        file=output/'peer_predictions'/f"{item['group']}_{sid}.npz";file.parent.mkdir(exist_ok=True)
        np.savez_compressed(file,candidate_ids=ids,actual=y,predicted=pred)
        records.append(dict(group=item['group'],split_id=sid,mae=mae,rmse=rmse,
            source_sha256=item['sha256'],prediction_file=str(file.relative_to(output)),model_class=type(model).__name__))
        print(json.dumps(dict(completed=len(records),total=50,group=item['group'],split_id=sid)),flush=True)
    assert len(records)==50
    (output/'peer_inference.json').write_text(json.dumps(dict(status='passed',versions=versions,records=records,
        model_fit_calls=0,recorded_metric_tolerance=1e-6),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()

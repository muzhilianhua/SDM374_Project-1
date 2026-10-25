import importlib.util, numpy as np, pandas as pd
spec=importlib.util.spec_from_file_location("b","bart_regression_strat82.py")
# 只复用配置，不执行脚本：直接内联最小复现检查
import contextlib, io, random
from bartpy2.sklearnmodel import SklearnModel
METRICS=("cost","carbon","peak","discomfort","degradation","ramping","resilience")
F=[f"b{b}_{m}" for b in range(1,4) for m in METRICS]
df=pd.read_csv("dataset_same8/train.csv")
X=df[F].to_numpy(float); y=df["ramping"].to_numpy(float)
P=dict(n_trees=50,alpha=0.95,beta=2.0,n_burn=500,n_samples=1000,thin=0.05,n_chains=1,n_jobs=1,store_in_sample_predictions=False)
def fit():
    np.random.seed(2026); random.seed(2026)
    m=SklearnModel(**P)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        m.fit(X[:300],y[:300])
    return m.predict(X[300:350])
a,b=fit(),fit()
print("max abs diff:", np.max(np.abs(a-b)), "identical:", np.array_equal(a,b))

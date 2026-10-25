import time, sys, warnings, contextlib, io, random
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, r"D:\HuaweiMoveData\Users\11659\Desktop\机器学习项目\Lab6")
from bartpy2.sklearnmodel import SklearnModel
METRICS=("cost","carbon","peak","discomfort","degradation","ramping","resilience")
FEATURES=[f"b{b}_{m}" for b in range(1,4) for m in METRICS]
df=pd.read_csv("dataset_same8/train.csv", float_precision="round_trip")
X=df[FEATURES].to_numpy(float); y=df["ramping"].to_numpy(float)
np.random.seed(2026); random.seed(2026)
p=dict(n_trees=50,alpha=0.95,beta=2.0,n_burn=500,n_samples=1000,thin=0.05,n_chains=1,n_jobs=1,store_in_sample_predictions=False)
s=time.time()
m=SklearnModel(**p)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    m.fit(X,y)
print("BART fit secs:", round(time.time()-s,1), "nsamples:", len(m.model_samples))
s=time.time(); pr=m.predict(X); print("predict secs:", round(time.time()-s,1))
from sklearn.metrics import mean_absolute_error
print("train MAE", mean_absolute_error(y,pr))

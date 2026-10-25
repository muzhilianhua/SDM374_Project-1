import time, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, r"D:\HuaweiMoveData\Users\11659\Desktop\机器学习项目\Lab6")
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

METRICS = ("cost","carbon","peak","discomfort","degradation","ramping","resilience")
FEATURES = [f"b{b}_{m}" for b in range(1,4) for m in METRICS]
df = pd.read_csv("dataset_same8/train.csv", float_precision="round_trip")
X=df[FEATURES].to_numpy(float); y=df["ramping"].to_numpy(float)

def t(name, f):
    s=time.time(); m=f(); 
    from sklearn.model_selection import cross_val_predict
    oof=cross_val_predict(m, X, y, cv=5, n_jobs=-1)
    print(name, "5fold OOF secs:", round(time.time()-s,1))

t("XGB", lambda: XGBRegressor(n_estimators=600,max_depth=5,learning_rate=0.05,subsample=0.9,colsample_bytree=0.9,tree_method="hist",random_state=2026,n_jobs=-1))
t("RF800", lambda: RandomForestRegressor(n_estimators=800,max_depth=None,min_samples_leaf=2,random_state=2026,n_jobs=-1))
t("HistGB", lambda: HistGradientBoostingRegressor(max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=1.0,early_stopping=False,learning_rate=0.05,max_iter=600,random_state=2026))
t("Ridge2", lambda: Pipeline([("poly",PolynomialFeatures(2,include_bias=False)),("sc",StandardScaler()),("r",Ridge(alpha=100.0))]))

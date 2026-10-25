import time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor

METRICS=("cost","carbon","peak","discomfort","degradation","ramping","resilience")
FEATURES=[f"b{b}_{m}" for b in range(1,4) for m in METRICS]
df=pd.read_csv("dataset_same8/train.csv", float_precision="round_trip")
X=df[FEATURES].to_numpy(float);y=df["ramping"].to_numpy(float)
cv=KFold(5,shuffle=True,random_state=2026)

class SPMM(BaseEstimator,RegressorMixin):
    def __init__(self,alpha_linear=100.0,alpha_kernel=1.0,gamma=0.05,degree=2):
        self.alpha_linear=alpha_linear;self.alpha_kernel=alpha_kernel;self.gamma=gamma;self.degree=degree
    def fit(self,X,y):
        X=np.asarray(X,float);y=np.asarray(y,float)
        self.scaler_=StandardScaler().fit(X);Z=self.scaler_.transform(X)
        self.poly_=PolynomialFeatures(self.degree,include_bias=False).fit(Z);P=self.poly_.transform(Z)
        self.linear_=Ridge(alpha=self.alpha_linear).fit(P,y)
        r=y-self.linear_.predict(P)
        self.kernel_=KernelRidge(kernel="rbf",alpha=self.alpha_kernel,gamma=self.gamma).fit(Z,r)
        return self
    def predict(self,X):
        X=np.asarray(X,float);Z=self.scaler_.transform(X);P=self.poly_.transform(Z)
        return self.linear_.predict(P)+self.kernel_.predict(Z)

s=time.time()
gs=GridSearchCV(XGBRegressor(subsample=0.9,colsample_bytree=0.9,tree_method="hist",random_state=2026,n_jobs=-1),
  {"max_depth":[3,5,7],"learning_rate":[0.05,0.1],"n_estimators":[300,600]},scoring="neg_mean_absolute_error",cv=cv,n_jobs=1).fit(X,y)
print("XGB best",gs.best_params_,"MAE",-gs.best_score_,"secs",round(time.time()-s,1))

s=time.time()
gs2=GridSearchCV(HistGradientBoostingRegressor(max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=1.0,early_stopping=False,random_state=2026),
  {"max_depth":[4,6,None],"learning_rate":[0.05,0.1],"max_iter":[300,600]},scoring="neg_mean_absolute_error",cv=cv,n_jobs=-1).fit(X,y)
print("HistGB best",gs2.best_params_,"MAE",-gs2.best_score_,"secs",round(time.time()-s,1))

s=time.time()
gs3=GridSearchCV(SPMM(),{"alpha_linear":[10.0,100.0,300.0],"alpha_kernel":[0.01,0.03,0.1,0.3],"gamma":[0.05,0.1,0.2]},scoring="neg_mean_absolute_error",cv=cv,n_jobs=-1).fit(X,y)
print("SPMM best",gs3.best_params_,"MAE",-gs3.best_score_,"secs",round(time.time()-s,1))

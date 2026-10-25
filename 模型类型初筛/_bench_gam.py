import time, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, r"D:\HuaweiMoveData\Users\11659\Desktop\机器学习项目\Lab6")
from scipy.interpolate import BSpline
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin
METRICS=("cost","carbon","peak","discomfort","degradation","ramping","resilience")
FEATURES=[f"b{b}_{m}" for b in range(1,4) for m in METRICS]
def _bs(x,n_knots=8,degree=3):
    x=np.asarray(x,float); xmin,xmax=float(x.min()),float(x.max())
    interior=np.quantile(x,np.linspace(0,1,n_knots+2))[1:-1]
    knots=np.concatenate([np.full(degree+1,xmin),interior,np.full(degree+1,xmax)])
    nb=len(knots)-degree-1; B=np.zeros((len(x),nb))
    for i in range(nb):
        c=np.zeros(nb); c[i]=1.0; sp=BSpline(knots,c,degree,extrapolate=False); B[:,i]=np.nan_to_num(sp(x),nan=0.0)
    return B,knots
def _d2(K):
    D=np.zeros((K-2,K))
    for i in range(K-2): D[i,i]=1;D[i,i+1]=-2;D[i,i+2]=1
    return D.T@D+1e-6*np.eye(K)
class BayesianGAM(BaseEstimator,RegressorMixin):
    def __init__(self,n_knots=8,degree=3,n_burn=1000,n_samples=2000,thin=2,a_tau=1.0,b_tau=0.1,a_sigma=1e-3,b_sigma=1e-3,seed=2026):
        self.n_knots=n_knots;self.degree=degree;self.n_burn=n_burn;self.n_samples=n_samples;self.thin=thin
        self.a_tau=a_tau;self.b_tau=b_tau;self.a_sigma=a_sigma;self.b_sigma=b_sigma;self.seed=seed
    def _build(self,X,fit=False):
        X=np.asarray(X,float); blocks=[]
        if fit: self._meta=[]
        for j in range(X.shape[1]):
            B,knots=_bs(X[:,j],self.n_knots,self.degree)
            if fit: cm=B.mean(0); self._meta.append((knots,cm))
            else:
                knots,cm=self._meta[j]; nb=len(knots)-self.degree-1; B=np.zeros((len(X),nb))
                for i in range(nb):
                    c=np.zeros(nb);c[i]=1.0;sp=BSpline(knots,c,self.degree,extrapolate=False);B[:,i]=np.nan_to_num(sp(X[:,j]),nan=0.0)
            blocks.append(B-cm)
        return np.hstack([np.ones((X.shape[0],1)),*blocks])
    def fit(self,X,y):
        rng=np.random.default_rng(self.seed); X=np.asarray(X,float);y=np.asarray(y,float);n=len(y)
        Z=self._build(X,True);p=Z.shape[1];J=X.shape[1]
        self._slices=[];col=1
        for j in range(J):
            knots,_=self._meta[j];Kj=len(knots)-self.degree-1;self._slices.append(slice(col,col+Kj));col+=Kj
        ZtZ=Z.T@Z;Zty=Z.T@y;Kp=[_d2(s.stop-s.start) for s in self._slices]
        sigma2=float(np.var(y));tau=np.ones(J);beta=np.linalg.solve(ZtZ+1e-3*np.eye(p),Zty)
        nt=self.n_burn+self.n_samples*self.thin; bk=np.zeros((self.n_samples,p));sk=np.zeros(self.n_samples);tk=np.zeros((self.n_samples,J));kept=0
        for it in range(nt):
            P=np.zeros((p,p))
            for j in range(J): s=self._slices[j];P[s,s]=tau[j]*Kp[j]
            prec=ZtZ/sigma2+P;L=np.linalg.cholesky(prec);mm=np.linalg.solve(L.T,np.linalg.solve(L,Zty/sigma2));beta=mm+np.linalg.solve(L.T,rng.standard_normal(p))
            for j in range(J):
                s=self._slices[j];bj=beta[s];quad=float(bj@Kp[j]@bj);Kj=s.stop-s.start
                tau[j]=rng.gamma(self.a_tau+0.5*Kj,1.0/(self.b_tau+0.5*quad))
            r=y-Z@beta;rss=float(r@r);iv=rng.gamma(self.a_sigma+0.5*n,1.0/(self.b_sigma+0.5*rss));sigma2=1.0/iv
            if it>=self.n_burn and (it-self.n_burn)%self.thin==0:
                bk[kept]=beta;sk[kept]=sigma2;tk[kept]=tau;kept+=1
        self.beta_samples_=bk;self.sigma2_samples_=sk;self.tau_samples_=tk;self.beta_mean_=bk.mean(0)
        return self
    def predict(self,X): return self._build(X,False)@self.beta_mean_
df=pd.read_csv("dataset_same8/train.csv", float_precision="round_trip")
X=df[FEATURES].to_numpy(float);y=df["ramping"].to_numpy(float)
s=time.time();m=BayesianGAM().fit(X,y);print("GAM fit secs:",round(time.time()-s,1));s=time.time();pr=m.predict(X);print("pred secs",round(time.time()-s,2))
from sklearn.metrics import mean_absolute_error;print("train MAE",mean_absolute_error(y,pr))

"""Approved v3.2 evaluator, with precomputed truth pairs for repeated inner selection."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

PROTOCOL_ID='sdm374.surrogate.tolerance.v3.2'
WEIGHTS=dict(local_order=35,local_mae=15,global_order=15,global_mae=15,top1_regret=20)


@dataclass(frozen=True)
class Policy:
    delta: float=.5
    fixed_cutoff: float=132.8

    def __post_init__(self):
        if self.delta!=.5 or self.fixed_cutoff!=132.8:raise ValueError('Only approved v3.2 policy is supported')


@dataclass(frozen=True)
class Context:
    cutoff: float
    mean: float
    scale: float

    def __post_init__(self):
        if self.cutoff!=132.8 or not np.isfinite([self.mean,self.scale]).all() or self.scale<=0:
            raise ValueError('Expected fixed cutoff and a positive finite training standard deviation')


def load_final_policy(path=None):
    path=Path(path) if path else Path(__file__).with_name('EVALUATION_POLICY_V3_2.json')
    data=json.loads(path.read_text(encoding='utf-8'))
    expected=dict(protocol_id=PROTOCOL_ID,status='FINAL',fixed_cutoff=132.8,delta=.5,weights=WEIGHTS,
        scale='unfiltered current training labels population standard deviation; ddof=0',
        pair_weight='min((absolute true-value gap - 0.5)/s, 1); only gaps > 0.5',
        pair_credit={'correct':1.,'tie':.5,'inverted':0.},prediction_tie_tolerance_in_training_std_units=1e-12,
        pair_skill='max(0, 2 * weighted_pair_accuracy - 1)',mae_skill='1/(1+MAE/s)',
        top1_selection='smallest prediction; exact ties broken by ascending candidate_id',
        top1_regret='actual value of selected record minus minimum actual value in current validation/test set',
        top1_skill='1/(1+top1_regret/s)',top1_regret_tolerance=0.)
    if any(data.get(k)!=v for k,v in expected.items()):raise ValueError('Configuration differs from approved v3.2')
    return Policy()


def make_context(training_y,policy):
    y=np.asarray(training_y,dtype=float)
    if y.ndim!=1 or len(y)<2 or not np.isfinite(y).all():raise ValueError('Invalid training labels')
    return Context(policy.fixed_cutoff,float(y.mean()),float(y.std(ddof=0)))


def make_pairs(y,context,policy,region):
    members=np.flatnonzero(y<=policy.fixed_cutoff) if region=='local' else np.arange(len(y))
    if region not in ('local','global'):raise ValueError(region)
    a,b=np.triu_indices(len(members),1);i,j=members[a],members[b]
    gap=np.abs(y[j]-y[i]);keep=gap>policy.delta;i,j,gap=i[keep],j[keep],gap[keep]
    return np.where(y[i]<y[j],i,j),np.where(y[i]<y[j],j,i),np.minimum((gap-policy.delta)/context.scale,1.)


def pair_accuracy(pred,pairs,scale):
    better,worse,w=pairs
    if not len(w):raise ValueError('No eligible comparison pairs; do not discard or impute this fold')
    gap=(pred[worse]-pred[better])/scale
    credit=np.where(gap>1e-12,1.,np.where(gap < -1e-12,0.,.5))
    return float(np.clip(np.dot(w,credit)/w.sum(),0.,1.))


class PreparedScore:
    def __init__(self,y,context,policy,candidate_ids):
        y=np.asarray(y,dtype=float);ids=np.asarray(candidate_ids)
        if y.ndim!=1 or ids.ndim!=1 or len(y)!=len(ids) or len(set(ids))!=len(y) or not np.isfinite(y).all():
            raise ValueError('Finite labels and unique aligned candidate IDs are required')
        if context.cutoff!=policy.fixed_cutoff:raise ValueError('Threshold mismatch')
        self.order=np.argsort(ids,kind='stable');self.y=y[self.order];self.ids=ids[self.order]
        self.context=context;self.policy=policy;self.low=self.y<=policy.fixed_cutoff
        if not self.low.any() or self.low.all():raise ValueError('Both low and other records are required')
        self.local=make_pairs(self.y,context,policy,'local');self.global_=make_pairs(self.y,context,policy,'global')
        if not len(self.local[2]) or not len(self.global_[2]):raise ValueError('Insufficient comparison evidence')
    def score(self,pred):
        pred=np.asarray(pred,dtype=float)
        if pred.shape!=self.y.shape or not np.isfinite(pred).all():raise ValueError('Invalid predictions')
        p=pred[self.order];y=self.y;s=self.context.scale;error=p-y
        local_a=pair_accuracy(p,self.local,s);global_a=pair_accuracy(p,self.global_,s)
        local_mae=float(np.mean(np.abs(error[self.low])));global_mae=float(np.mean(np.abs(error)))
        ordered=np.argsort(p,kind='stable');winner=int(ordered[0]);best=float(y.min())
        regret=float(y[winner]-best)
        points=dict(local_order=35*max(0.,2*local_a-1),local_mae=15/(1+local_mae/s),
            global_order=15*max(0.,2*global_a-1),global_mae=15/(1+global_mae/s),top1_regret=20/(1+regret/s))
        m=int(self.low.sum());hits=int(self.low[ordered[:m]].sum())
        return dict(status='FINAL_V3_2',protocol_id=PROTOCOL_ID,points=points,score=sum(points.values()),
            cutoff=self.policy.fixed_cutoff,delta=self.policy.delta,train_std=s,train_mean=self.context.mean,
            low_n=m,test_n=len(y),local_pair_accuracy=local_a,global_pair_accuracy=global_a,
            local_pair_n=len(self.local[2]),global_pair_n=len(self.global_[2]),
            local_mae=local_mae,global_mae=global_mae,local_bias=float(error[self.low].mean()),global_bias=float(error.mean()),
            top1_regret=regret,top1_candidate_id=str(self.ids[winner]),top1_predicted=float(p[winner]),
            top1_actual=float(y[winner]),oracle_best_actual=best,recall_hits=hits,recommendation_count=m,
            recall_at_low_count=hits/m)


def evaluate(y,pred,context,policy,candidate_ids):
    return PreparedScore(y,context,policy,candidate_ids).score(pred)

"""Fixed, predeclared equal blend of additive GP and the existing tree ensemble."""
from dataclasses import dataclass
import numpy as np
from ..interfaces import Surrogate
from ..registry import register_model
from .bayes_search import BayesianSearchModel
from .method_study import MethodStudy


@register_model('gp-tree-blend')
class GPTreeBlend(Surrogate):
    """Equal GP/tree mean; uncertainty is a disagreement envelope, not calibrated confidence."""
    uncertainty_kind = 'heuristic'
    @dataclass(frozen=True)
    class Params:
        seed: int = 2026
        gp_kernel: str = 'additive'

    def fit(self, X, y):
        ard = self.params.gp_kernel == 'additive-ard'
        self.gp = BayesianSearchModel(BayesianSearchModel.Params(kernel=self.params.gp_kernel,
            lengths=(.5,) if ard else (.25,.5,1.), noises=(.03,) if ard else (.005,.03),
            seed=self.params.seed)).fit(X,y)
        self.tree = MethodStudy(MethodStudy.Params(family='ensemble',ensemble_mode='stack',
                                seed=self.params.seed)).fit(X,y)
        self.metrics = {'gp_weight':.5,'tree_weight':.5,'uncertainty_is_heuristic':True,
                        'gp_seed_affects_fit':False, 'seed_varies_tree_only':True}
        return self

    def predict(self, X):
        a, sa = self.gp.predict(X)
        b, sb = self.tree.predict(X)
        return (a+b)/2, np.maximum(sa,sb)+np.abs(a-b)/2

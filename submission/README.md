# Standalone T15 policy

`policy.py` and `surrogate.py` contain the batch proposal part of the graded `submission/` package.
They do not import `sdm` or read any development run, solver cache, or research log. The only inputs
to a proposal are the 2,048 training rows and successful or failed points recorded by the current run.

## API

```python
import numpy as np
from submission.policy import BlockDescentPolicy, State

state = State(X_train=train_weights, y_train=train_r)
policy = BlockDescentPolicy(seed=2026)
batch, metadata, info = policy.propose_weights(state, n=50)
```

Each returned row is a `(3, 7)` array. The policy validates the simplex by normalizing every building,
keeps a 25/25 exploit/explore design in the first 50 candidates, and refills after deduplication until
the requested batch is full. Training rows and every point from a previous attempt are excluded within
`1e-9` in each coordinate. Failed attempts remain excluded but do not enter surrogate training.

The default constants in `config.json` are the T37 v2 settings. `Surrogate` uses a fixed feature map,
ridge trend and local RBF residual model. Its kernel hyperparameters are constants and the Gaussian
process optimizer is disabled so a refit cannot change the next batch on another machine.

`State.calls` is a list of calls; each call is a list of records with `W`, `R` (`NaN` for a failed
attempt), and `meta`. A runner can serialize those records as JSON after converting arrays to lists.
The runner owns solver invocation, checkpoint files, and the 20-call quota.

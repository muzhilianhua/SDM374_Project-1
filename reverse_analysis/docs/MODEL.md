# Model and scoring

The original course `solver.exe` contains a .NET wrapper and a PyInstaller Python
3.11 program. Its public fixed eight-week path combines CityLearn 2.5.0, an hourly
bounded controller and pretrained indoor-temperature dynamics. This release
executes the recovered task logic as source and uses the original data/model files.

## Controller inputs

The 21 values are three rows of seven controller preferences, in this order:

1. Cost, including the battery future-price adjustment.
2. Nonnegative carbon emissions.
3. Maximum of completed peak and predicted current net consumption.
4. A discomfort proxy using current indoor temperature, setpoint and cooling action.
5. Battery throughput/action-change degradation proxy.
6. Per-building change from the previous hourly net consumption.
7. Reserve-SOC deficit and unserved non-shiftable load during outages.

They are **not** the LSTM parameters. Each row is normalized by its sum. Every hour,
`ProbeAgent(mode="bounded")` enumerates battery and cooling actions subject to
power, energy, SOC, reserve, action-change and cooling-service limits. Its seven
metrics are normalized by their spans over the current candidate set. Columns with
span at most `1e-10 * max(1, max(abs(metric)))` are ignored.

The minimum weighted metric sum wins. Near ties (within
`1e-12 * max(1, abs(best))`) use the original ordered tie keys: battery action
magnitude, excess cooling, cooling action and battery action. The candidate set and
future physical state depend on previous decisions, so final R is not a linear
function of the 21 weights.

## Temperature and equipment

Each building is a `LSTMDynamicsBuilding` with its own pretrained checkpoint:
13 inputs, a 12-hour window, two LSTM layers with hidden size 16, and a 16-to-1
linear output (4,177 parameters per building). Inputs are solar irradiation,
outdoor temperature, indoor cooling setpoint, occupant count, cooling demand,
cyclic month/hour/day-type features and indoor temperature. Normalization ranges
are specified in each week's schema. The full original training dataset and
training project were not recovered.

The environment also models battery efficiency curves, power limits, SOC,
self-discharge, depth of discharge and capacity degradation, plus heat-pump COP,
photovoltaics, non-shiftable consumption, heating and domestic hot water.
`initialization.py` corrects hour-zero accounting and checks completed-hour energy
flows. `citylearn_task/temporal.py` preserves the original padding and observation
timing; these details are needed for numerical agreement.

## Final score

Each of the eight weeks has 168 hourly values and uses scenario seed 2026. The
public path sets `family="load_pv"`, `strength=0`, `disturbance_seed=0`.
First add the three buildings' net loads, then sum absolute within-week hourly
changes, then average the eight weekly scores:

```python
# nets has shape (8 weeks, 3 buildings, 168 hours)
weekly_R = np.abs(np.diff(nets.sum(axis=1), axis=1)).sum(axis=1)
R = float(weekly_R.mean())
```

There are 167 differences per week; week boundaries are not joined. This district
score is different from summing the three buildings' individual ramping scores.
R is nonnegative, but no achievable global minimum or feasibility below 100 has
been proved.

Read `source_view/evaluate.py`, then `controller.py`, `initialization.py`, and the
four modules in `source_view/citylearn_task/`. All public-path task imports now use
these readable files. The base agent's unused `predict()` branch is explicitly
omitted; the public `ProbeAgent` supplies the implemented method. Alternate
disturbance branches and the original wrapper's quota implementation are outside
the validated release scope.

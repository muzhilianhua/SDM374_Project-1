from pathlib import Path
import importlib.metadata
import json
import numpy as np

from load_bundle import install, BUNDLE

install()
original_version = importlib.metadata.version
importlib.metadata.version = lambda name: '2.5.0' if name == 'CityLearn' else original_version(name)
from evaluate import evaluate_probe
from citylearn.data import DataSet
DataSet.get_pv_sizing_data = lambda *args, **kwargs: None
DataSet.get_battery_sizing_data = lambda *args, **kwargs: None

root = Path(__file__).resolve().parent / 'resources' / 'environment' / 'inputs'
w = np.tile(np.full(7, 1/7), (3, 1))
scores = []
for i in range(1, 9):
    scenario = Path(__file__).resolve().parent / 'resources' / 'environment' / 'scenarios' / f'week_{i:02d}.json'
    result = evaluate_probe(w, str(scenario), 'load_pv', 0, 0, True)
    scores.append(result['ramping'])
    print(i, result['ramping'], result['steps'])
print('mean', float(np.mean(scores)))

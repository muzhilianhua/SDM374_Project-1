"""Check final model loading against supplied in-sample predictions; never fit."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import pandas as pd
import joblib


def main():
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temp:
        with zipfile.ZipFile(here / "packages/SDM374_final_models_v1.zip") as z:
            for name in z.namelist():
                assert (Path(temp) / name).resolve().is_relative_to(Path(temp).resolve())
            z.extractall(temp)
        package = Path(temp) / "SDM374_final_models_v1"
        manifest = json.loads((package / "MANIFEST.json").read_text("utf8"))
        data = pd.read_csv(here.parents[1] / "dataset_same8/train.csv")
        X = data[manifest["feature_columns"]].to_numpy(dtype=float)
        reference = pd.read_csv(package / "evaluation/final_insample_predictions.csv")
        checks = []
        for name, entry in manifest["models"].items():
            model = joblib.load(package / entry["model_file"])
            pred = model.predict(X)
            sub = reference[reference["config"] == name].set_index("candidate_id")
            expected = sub.loc[data.candidate_id, "y_pred_insample"].to_numpy()
            np.testing.assert_allclose(pred, expected, rtol=0, atol=1e-10)
            checks.append(dict(model=name, predictions=len(pred),
                max_absolute_difference=float(abs(pred - expected).max()), sha256=entry["model_sha256"]))
        result = dict(status="passed", scope="In-sample loading smoke test only; not generalization evaluation",
            fitting_performed=False, models=checks)
        out = here / "results/model_smoke_check.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

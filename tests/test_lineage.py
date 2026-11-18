import hashlib
import json
from pathlib import Path

import joblib
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def file_hash(path, algorithm="md5"):
    digest = hashlib.new(algorithm, usedforsecurity=False)

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def test_lineage_matches_existing_artifacts():
    lineage = load_json(ROOT / "artifacts/model_lineage.json")
    selection = load_json(ROOT / "artifacts/model_selection.json")

    with (ROOT / "dvc.lock").open(encoding="utf-8") as file:
        stages = yaml.safe_load(file)["stages"]

    assert lineage["task"] == "regression"
    assert lineage["target"] == "SalePrice"
    assert lineage["selected_model"] == selection["selected_model"]

    assert lineage["selection_metric"] == selection["selection_metric"]
    assert lineage["validation_rmse_log_price"] == pytest.approx(
        selection["validation_rmse_log_price"]
    )

    expected_artifacts = {
        "train": ("features", "data/processed/train.parquet"),
        "validation": (
            "features",
            "data/processed/validation.parquet",
        ),
        "test": ("features", "data/processed/test.parquet"),
        "model": (
            "train",
            "models/selected_house_price_model.joblib",
        ),
        "selection": (
            "train",
            "artifacts/model_selection.json",
        ),
    }

    evaluation_path = ROOT / "reports/metrics.json"

    if evaluation_path.is_file():
        expected_artifacts["test_metrics"] = (
            "evaluate",
            "reports/metrics.json",
        )
        assert lineage["test_metrics"] == load_json(evaluation_path)
    else:
        assert lineage["test_metrics"] is None

    recorded = lineage["verified_local_artifacts"]
    assert set(recorded) == set(expected_artifacts)

    # Check lineage against both DVC records and actual local files.
    for name, (stage_name, relative_path) in expected_artifacts.items():
        entries = [
            item
            for item in stages[stage_name]["outs"]
            if item["path"] == relative_path
        ]

        assert len(entries) == 1, (
            f"Expected one DVC output entry for {relative_path}"
        )

        expected_md5 = entries[0]["md5"]

        assert recorded[name]["path"] == relative_path
        assert recorded[name]["md5"] == expected_md5
        assert file_hash(ROOT / relative_path) == expected_md5

    model_path = ROOT / expected_artifacts["model"][1]

    assert lineage["model_sha256"] == file_hash(
        model_path, "sha256"
    )
    assert lineage["dvc_lock_sha256"] == file_hash(
        ROOT / "dvc.lock", "sha256"
    )

    # Load only a trusted, locally trained model artifact.
    model = joblib.load(model_path)

    assert lineage["features"] == list(model.feature_names_in_)
    assert "Id" not in lineage["features"]
    assert "SalePrice" not in lineage["features"]